"""
MCAR-Net: Multi-scale Context-Aware Residual network for 3D brain tumor segmentation.

This file contains the single model definition used for all reported experiments
(Table 1, Table 2, UCSF-PDGM external validation). Parameter names are identical to
the original training notebook, so the released checkpoints load directly.

Default configuration (MCARNet(in_channels=4, out_channels=4, base_ch=16)):
    6.57 M trainable parameters, 141.91 GMac for a 4 x 128 x 128 x 128 input
    (ptflops 0.7.4, see complexity.py).

The optional arguments `use_amcf`, `amcf_attention` and `decoder_gate` reproduce the
component-wise ablation variants of Table 3. They default to the full MCAR-Net.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Feature Boosting Attention (FBA) = channel attention followed by spatial attention
# (Section 3.5, Eqs. 6-7; Supplementary Figs. 1-2)
# ---------------------------------------------------------------------------
class ChannelAttention3D(nn.Module):
    """C_i^a(.) in Eq. 7: global avg + max pooling -> shared 1x1x1 MLP -> sigmoid."""

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention3D(nn.Module):
    """S_i^a(.) in Eq. 6: channel-wise avg + max -> k x k x k conv (k = 3) -> sigmoid."""

    def __init__(self, kernel_size=3):
        super().__init__()
        assert kernel_size in (3, 5, 7), "kernel size must be 3, 5 or 7"
        self.conv3d = nn.Conv3d(2, 1, kernel_size=kernel_size,
                                padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv3d(torch.cat([avg_out, max_out], dim=1)))


class FeatureWiseAttention3D(nn.Module):
    """FBA block: y_i = C(X) * X (Eq. 7), x_i = S(y_i) * y_i (Eq. 6).

    mode: 'both' (MCAR-Net), 'channel' or 'spatial' (Table 3 ablations).
    """

    def __init__(self, channels, reduction_ratio=16, spatial_kernel_size=3, mode="both"):
        super().__init__()
        assert mode in ("both", "channel", "spatial")
        self.mode = mode
        if mode in ("both", "channel"):
            self.channel_attention = ChannelAttention3D(channels, reduction_ratio)
        if mode in ("both", "spatial"):
            self.spatial_attention = SpatialAttention3D(spatial_kernel_size)

    def forward(self, x):
        if self.mode in ("both", "channel"):
            x = x * self.channel_attention(x)
        if self.mode in ("both", "spatial"):
            x = x * self.spatial_attention(x)
        return x


# ---------------------------------------------------------------------------
# Residual block (Sections 3.2-3.4, Eqs. 1, 3, 5)
# ---------------------------------------------------------------------------
class ResidualBlock3D(nn.Module):
    """Two 3x3x3 conv (stride 1, padding 1) + BN + ReLU, 1x1x1 conv shortcut if channels differ."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)


# ---------------------------------------------------------------------------
# Context-Gated Squeeze-and-Excitation (CGSE) (Section 3.6, Eqs. 9-13)
# ---------------------------------------------------------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation with reduction r = 16 (Supplementary Fig. 3)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1, 1)
        return x * y


class AttentionGate3D(nn.Module):
    """CGSE module.

    x : encoder skip feature X^{e'}_{(n-i)}
    g : upsampled decoder feature P_i (output of the transposed convolution)

    z'_i = W_x(x) + W_g(g)                 (1x1x1 conv + BN on each input, Eq. 13)
    g_i  = sigmoid(conv1x1x1(ReLU(z'_i)))  (spatial gate, Eq. 12)
    z_i  = x * g_i                         (gated encoder skip feature, Eq. 10)
    Y^da = SE(z_i)                         (Eq. 9)

    mode: 'ga_se' (MCAR-Net), 'ga' (gate only), 'se' (SE only) for Table 3.
    """

    def __init__(self, in_ch, gating_ch, inter_ch, mode="ga_se"):
        super().__init__()
        assert mode in ("ga_se", "ga", "se")
        self.mode = mode
        if mode in ("ga_se", "ga"):
            self.W_g = nn.Sequential(nn.Conv3d(gating_ch, inter_ch, kernel_size=1),
                                     nn.BatchNorm3d(inter_ch))
            self.W_x = nn.Sequential(nn.Conv3d(in_ch, inter_ch, kernel_size=1),
                                     nn.BatchNorm3d(inter_ch))
            self.psi = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Conv3d(inter_ch, 1, kernel_size=1),
                                     nn.Sigmoid())
        if mode in ("ga_se", "se"):
            self.se = SEBlock(in_ch)

    def forward(self, x, g):
        out = x
        if self.mode in ("ga_se", "ga"):
            out = out * self.psi(self.W_x(x) + self.W_g(g))
        if self.mode in ("ga_se", "se"):
            out = self.se(out)
        return out


class UpBlock3D(nn.Module):
    """2x2x2 transposed conv (stride 2) -> CGSE on skip -> concat -> residual block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.resblock = ResidualBlock3D(in_ch, out_ch)

    def forward(self, x, skip, attn_gate=None):
        x = self.up(x)
        if attn_gate is not None:
            skip = attn_gate(skip, x)
        return self.resblock(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# MCAR-Net
# ---------------------------------------------------------------------------
class MCARNet(nn.Module):
    """
    Args:
        in_channels  : 4 (FLAIR, T1ce, T2, T1 stacked in this order)
        out_channels : 4 (background, NCR/NET, ED, ET; ET relabelled 4 -> 3)
        base_ch      : 16 -> channels 16, 32, 64, 128, bottleneck 256
        use_amcf     : True = AMCF multi-scale fusion into the bottleneck (MCAR-Net).
                       False = bottleneck fed by max-pooled last encoder level only.
        amcf_attention : 'both' | 'channel' | 'spatial'   (Table 3)
        decoder_gate   : 'ga_se' | 'ga' | 'se' | 'none'  (Table 3; 'none' = plain skip concat)
    """

    def __init__(self, in_channels=4, out_channels=4, base_ch=16,
                 use_amcf=True, amcf_attention="both", decoder_gate="ga_se"):
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        self.use_amcf = use_amcf

        # Encoder (four residual levels)
        self.enc1 = ResidualBlock3D(in_channels, chs[0])
        self.enc2 = ResidualBlock3D(chs[0], chs[1])
        self.enc3 = ResidualBlock3D(chs[1], chs[2])
        self.enc4 = ResidualBlock3D(chs[2], chs[3])
        self.pool = nn.MaxPool3d(2)

        if use_amcf:
            # AMCF: one FBA per encoder level + scale-dependent average pooling
            # k = 2^(n-i-1), n = 5 -> 8, 4, 2, 1 (Eq. 8)
            self.at1 = FeatureWiseAttention3D(chs[0], mode=amcf_attention)
            self.at2 = FeatureWiseAttention3D(chs[1], mode=amcf_attention)
            self.at3 = FeatureWiseAttention3D(chs[2], mode=amcf_attention)
            self.at4 = FeatureWiseAttention3D(chs[3], mode=amcf_attention)
            self.ap1 = nn.AvgPool3d(8)
            self.ap2 = nn.AvgPool3d(4)
            self.ap3 = nn.AvgPool3d(2)
            bottleneck_in = chs[0] + chs[1] + chs[2] + chs[3]  # 240
        else:
            bottleneck_in = chs[3]

        # Residual bottleneck (Eqs. 3-4)
        self.bottleneck = ResidualBlock3D(bottleneck_in, chs[4])

        # Decoder with CGSE on every skip connection
        def gate(c):
            return None if decoder_gate == "none" else AttentionGate3D(c, c, c // 2, mode=decoder_gate)

        self.up4, self.att4 = UpBlock3D(chs[4], chs[3]), gate(chs[3])
        self.up3, self.att3 = UpBlock3D(chs[3], chs[2]), gate(chs[2])
        self.up2, self.att2 = UpBlock3D(chs[2], chs[1]), gate(chs[1])
        self.up1, self.att1 = UpBlock3D(chs[1], chs[0]), gate(chs[0])

        self.final_conv = nn.Conv3d(chs[0], out_channels, kernel_size=1)  # Eq. 14

    def forward(self, x):
        e1 = self.enc1(x)                 # 16  x 128^3
        e2 = self.enc2(self.pool(e1))     # 32  x 64^3
        e3 = self.enc3(self.pool(e2))     # 64  x 32^3
        e4 = self.enc4(self.pool(e3))     # 128 x 16^3

        if self.use_amcf:
            p1 = self.ap1(self.at1(e1))   # 16  x 16^3
            p2 = self.ap2(self.at2(e2))   # 32  x 16^3
            p3 = self.ap3(self.at3(e3))   # 64  x 16^3
            p4 = self.at4(e4)             # 128 x 16^3
            b_in = torch.cat([p1, p2, p3, p4], dim=1)  # 240 x 16^3
        else:
            b_in = e4

        b = self.bottleneck(self.pool(b_in))           # 256 x 8^3

        d4 = self.up4(b, e4, self.att4)
        d3 = self.up3(d4, e3, self.att3)
        d2 = self.up2(d3, e2, self.att2)
        d1 = self.up1(d2, e1, self.att1)
        return self.final_conv(d1)        # 4 x 128^3 logits


# Name used in the original notebook; kept so old code and checkpoints work unchanged.
ResAttUNet3D = MCARNet


# Table 3 component-wise ablation variants
ABLATION_VARIANTS = {
    "baseline":            dict(use_amcf=False, decoder_gate="none"),
    "amcf_channel":        dict(use_amcf=True, amcf_attention="channel", decoder_gate="none"),
    "amcf_spatial":        dict(use_amcf=True, amcf_attention="spatial", decoder_gate="none"),
    "amcf":                dict(use_amcf=True, amcf_attention="both", decoder_gate="none"),
    "amcf_cgse_se":        dict(use_amcf=True, amcf_attention="both", decoder_gate="se"),
    "amcf_cgse_ga":        dict(use_amcf=True, amcf_attention="both", decoder_gate="ga"),
    "mcar_net":            dict(use_amcf=True, amcf_attention="both", decoder_gate="ga_se"),
}


def build_model(variant="mcar_net", in_channels=4, out_channels=4, base_ch=16):
    return MCARNet(in_channels, out_channels, base_ch, **ABLATION_VARIANTS[variant])
