import sys
sys.path.append('/kaggle/input/monai/pytorch/default/1/')

import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged,
    CropForegroundd, Orientationd, RandFlipd, RandAffined,
    RandScaleIntensityd, RandShiftIntensityd, ToTensord
)
from monai.data import Dataset, DataLoader

# Define directories
train_dir = "/kaggle/input/brats-2020-new/Brats 2020 New/Train"
val_dir = "/kaggle/input/brats-2020-new/Brats 2020 New/Val"
test_dir = "/kaggle/input/brats-2020-new/Brats 2020 New/Test"

# Subdirs
train_images_dir = os.path.join(train_dir, "Images")
train_labels_dir = os.path.join(train_dir, "Mask")
val_images_dir = os.path.join(val_dir, "Images")
val_labels_dir = os.path.join(val_dir, "Mask")
test_images_dir = os.path.join(test_dir, "Images")
test_labels_dir = os.path.join(test_dir, "Mask")

# File lists
train_image_files = sorted(glob.glob(f"{train_images_dir}/*.nii"))
train_label_files = sorted(glob.glob(f"{train_labels_dir}/*.nii"))
val_image_files = sorted(glob.glob(f"{val_images_dir}/*.nii"))
val_label_files = sorted(glob.glob(f"{val_labels_dir}/*.nii"))
test_image_files = sorted(glob.glob(f"{test_images_dir}/*.nii"))
test_label_files = sorted(glob.glob(f"{test_labels_dir}/*.nii"))

# Matching check
if len(train_image_files) != len(train_label_files):
    raise ValueError(f"Train images and labels count mismatch.")
if len(val_image_files) != len(val_label_files):
    raise ValueError(f"Val images and labels count mismatch.")
if len(test_image_files) != len(test_label_files):
    raise ValueError(f"Test images and labels count mismatch.")

# Create file dictionaries
train_files = [{"image": img, "label": lbl} for img, lbl in zip(train_image_files, train_label_files)]
val_files = [{"image": img, "label": lbl} for img, lbl in zip(val_image_files, val_label_files)]
test_files = [{"image": img, "label": lbl} for img, lbl in zip(test_image_files, test_label_files)]

# Train transforms with augmentation

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    # ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0, b_max=1, clip=True),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000, b_min=0, b_max=255, clip=True),
    # CropForegroundd(keys=["image", "label"], source_key="image"),

    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

    RandAffined(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=0.3,
        rotate_range=(0.1, 0.1, 0.1),
        scale_range=(0.1, 0.1, 0.1),
        padding_mode="border"
    ),

    RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ToTensord(keys=["image", "label"]),
])


# Val/test transforms (no augmentation)
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    # ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0, b_max=1, clip=True),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000, b_min=0, b_max=255, clip=True),
    # CropForegroundd(keys=["image", "label"], source_key="image"),
    ToTensord(keys=["image", "label"]),
])

# Datasets and loaders
train_ds = Dataset(train_files, train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

val_ds = Dataset(val_files, val_transforms)
val_loader = DataLoader(val_ds, batch_size=1)

test_ds = Dataset(test_files, val_transforms)
test_loader = DataLoader(test_ds, batch_size=1)

print(f"Training samples: {len(train_files)}, Validation samples: {len(val_files)}, Testing samples: {len(test_files)}")





import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention3D(nn.Module):
    """
    3D Channel Attention Module.
    This module computes attention weights for each channel by aggregating spatial information
    and passing it through a shared Multi-Layer Perceptron (MLP).
    """
    def __init__(self, channels, reduction_ratio=16):
        """
        Initializes the ChannelAttention3D module.

        Args:
            channels (int): Number of input channels.
            reduction_ratio (int): Reduction ratio for the MLP, controlling its complexity.
        """
        super(ChannelAttention3D, self).__init__()
        # Adaptive average pooling across spatial dimensions (Depth, Height, Width)
        # This reduces each channel's spatial information to a single value.
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # Adaptive max pooling across spatial dimensions
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # Shared MLP for both average and max pooled features.
        # It consists of two 1x1x1 convolutions (acting as linear layers for 1x1x1 input)
        # with a ReLU activation in between.
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        """
        Forward pass for the ChannelAttention3D module.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Depth, Height, Width).

        Returns:
            torch.Tensor: Channel attention weights of shape (Batch, Channels, 1, 1, 1).
        """
        # Apply global average pooling to get channel-wise statistics
        avg_out = self.avg_pool(x)
        # Apply global max pooling to get channel-wise statistics
        max_out = self.max_pool(x)

        # Pass both pooled features through the shared MLP
        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)

        # Sum the outputs of the MLP and apply sigmoid to get attention weights.
        # Sigmoid ensures weights are between 0 and 1.
        attention_weights = torch.sigmoid(avg_out + max_out)
        return attention_weights

class SpatialAttention3D(nn.Module):
    """
    3D Spatial Attention Module.
    This module computes attention weights for each spatial location by aggregating
    channel-wise information and passing it through a 3D convolutional layer.
    """
    def __init__(self, kernel_size=3):
        """
        Initializes the SpatialAttention3D module.

        Args:
            kernel_size (int): Kernel size for the 3D convolution. Must be odd (3, 5, or 7).
        """
        super(SpatialAttention3D, self).__init__()
        # Ensure kernel size is odd for symmetric padding
        assert kernel_size in (3, 5, 7), 'kernel size must be 3, 5 or 7'
        padding = (kernel_size - 1) // 2 # Calculate padding to maintain spatial dimensions

        # 3D convolution to learn spatial relationships.
        # It takes concatenated average and max pooled features (2 channels)
        # and outputs a single channel attention map.
        self.conv3d = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        """
        Forward pass for the SpatialAttention3D module.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Depth, Height, Width).

        Returns:
            torch.Tensor: Spatial attention weights of shape (Batch, 1, Depth, Height, Width).
        """
        # Average pool across the channel dimension.
        # keepdim=True maintains the channel dimension for concatenation.
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Max pool across the channel dimension.
        # .values is used with torch.max to get the max values, not indices.
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate average and max pooled features along the channel dimension.
        # This creates a tensor with 2 channels.
        combined_features = torch.cat([avg_out, max_out], dim=1)

        # Pass through the 3D convolution and apply sigmoid to get spatial attention weights.
        attention_weights = torch.sigmoid(self.conv3d(combined_features))
        return attention_weights

class FeatureWiseAttention3D(nn.Module):
    """
    A 3D Feature-wise Attention module that combines Channel Attention and Spatial Attention.
    This module takes a 5D input tensor (Batch, Channels, Depth, Height, Width)
    and applies attention mechanisms to re-weight features across channels and spatial dimensions.
    The attention is applied sequentially: first channel attention, then spatial attention.
    """
    def __init__(self, channels, reduction_ratio=16, spatial_kernel_size=3):
        """
        Initializes the FeatureWiseAttention3D module.

        Args:
            channels (int): Number of input channels.
            reduction_ratio (int): Reduction ratio for the Channel Attention MLP.
            spatial_kernel_size (int): Kernel size for the Spatial Attention 3D convolution.
        """
        super(FeatureWiseAttention3D, self).__init__()
        # Initialize the Channel Attention Module
        self.channel_attention = ChannelAttention3D(channels, reduction_ratio)
        # Initialize the Spatial Attention Module
        self.spatial_attention = SpatialAttention3D(spatial_kernel_size)

    def forward(self, x):
        """
        Forward pass for the FeatureWiseAttention3D module.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Depth, Height, Width).

        Returns:
            torch.Tensor: Output tensor of the same shape, with features re-weighted by attention.
        """
        # Apply Channel Attention:
        # 1. Compute channel attention map (B, C, 1, 1, 1).
        channel_att_map = self.channel_attention(x)
        # 2. Multiply the input tensor by the channel attention map.
        #    The channel_att_map is broadcasted across D, H, W dimensions.
        x_after_channel_att = x * channel_att_map

        # Apply Spatial Attention:
        # 1. Compute spatial attention map (B, 1, D, H, W) using the output from channel attention.
        spatial_att_map = self.spatial_attention(x_after_channel_att)
        # 2. Multiply the tensor (after channel attention) by the spatial attention map.
        #    The spatial_att_map is broadcasted across the C dimension.
        x_final = x_after_channel_att * spatial_att_map

        return x_final

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
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

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class AttentionGate3D(nn.Module):
    def __init__(self, in_ch, gating_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(gating_ch, inter_ch, kernel_size=1),
            nn.BatchNorm3d(inter_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(in_ch, inter_ch, kernel_size=1),
            nn.BatchNorm3d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.se = SEBlock(in_ch)

    def forward(self, x, g):
        psi = self.psi(self.W_x(x) + self.W_g(g))
        out = x * psi
        out = self.se(out)  # Apply SE block after attention
        return out

class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.resblock = ResidualBlock3D(in_ch, out_ch)

    def forward(self, x, skip, attn_gate=None):
        x = self.up(x)
        if attn_gate is not None:
            skip = attn_gate(skip, x)
        x = torch.cat([x, skip], dim=1)
        return self.resblock(x)

class ResAttUNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, base_ch=16):
        super().__init__()
        chs = [base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16]

        # Encoder
        self.enc1 = ResidualBlock3D(in_channels, chs[0])
        self.enc2 = ResidualBlock3D(chs[0], chs[1])
        self.enc3 = ResidualBlock3D(chs[1], chs[2])
        self.enc4 = ResidualBlock3D(chs[2], chs[3])

        self.at1 = FeatureWiseAttention3D(chs[0])
        self.at2 = FeatureWiseAttention3D(chs[1])
        self.at3 = FeatureWiseAttention3D(chs[2])
        self.at4 = FeatureWiseAttention3D(chs[3])

        self.ap1 = nn.AvgPool3d(8)
        self.ap2 = nn.AvgPool3d(4)
        self.ap3 = nn.AvgPool3d(2)

        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock3D(240, chs[4])

        # Decoder
        self.up4 = UpBlock3D(chs[4], chs[3])
        self.att4 = AttentionGate3D(chs[3], chs[3], chs[3]//2)

        self.up3 = UpBlock3D(chs[3], chs[2])
        self.att3 = AttentionGate3D(chs[2], chs[2], chs[2]//2)

        self.up2 = UpBlock3D(chs[2], chs[1])
        self.att2 = AttentionGate3D(chs[1], chs[1], chs[1]//2)

        self.up1 = UpBlock3D(chs[1], chs[0])
        self.att1 = AttentionGate3D(chs[0], chs[0], chs[0]//2)

        self.final_conv = nn.Conv3d(chs[0], out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        a1 = self.at1(e1)
        a2 = self.at2(e2)
        a3 = self.at3(e3)
        # print(e1.shape)
        a4 = self.at4(e4)

        p1 = self.ap1(a1)
        p2 = self.ap2(a2)
        p3 = self.ap3(a3)

        # print(a1.shape)
        # print(p1.shape)

        bo = torch.cat([p1, p2, p3, a4], dim=1)
        # print(bo.shape)

        
        b = self.bottleneck(self.pool(bo))

        d4 = self.up4(b, e4, self.att4)
        d3 = self.up3(d4, e3, self.att3)
        d2 = self.up2(d3, e2, self.att2)
        d1 = self.up1(d2, e1, self.att1)

        return self.final_conv(d1)


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
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

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class AttentionGate3D(nn.Module):
    def __init__(self, in_ch, gating_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(gating_ch, inter_ch, kernel_size=1),
            nn.BatchNorm3d(inter_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(in_ch, inter_ch, kernel_size=1),
            nn.BatchNorm3d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.se = SEBlock(in_ch)

    def forward(self, x, g):
        psi = self.psi(self.W_x(x) + self.W_g(g))
        out = x * psi
        out = self.se(out)  # Apply SE block after attention
        return out

class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.resblock = ResidualBlock3D(in_ch, out_ch)

    def forward(self, x, skip, attn_gate=None):
        x = self.up(x)
        if attn_gate is not None:
            skip = attn_gate(skip, x)
        x = torch.cat([x, skip], dim=1)
        return self.resblock(x)

class ResAttUNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, base_ch=16):
        super().__init__()
        chs = [base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16]

        # Encoder
        self.enc1 = ResidualBlock3D(in_channels, chs[0])
        self.enc2 = ResidualBlock3D(chs[0], chs[1])
        self.enc3 = ResidualBlock3D(chs[1], chs[2])
        self.enc4 = ResidualBlock3D(chs[2], chs[3])

        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock3D(chs[3], chs[4])

        # Decoder
        self.up4 = UpBlock3D(chs[4], chs[3])
        self.att4 = AttentionGate3D(chs[3], chs[3], chs[3]//2)

        self.up3 = UpBlock3D(chs[3], chs[2])
        self.att3 = AttentionGate3D(chs[2], chs[2], chs[2]//2)

        self.up2 = UpBlock3D(chs[2], chs[1])
        self.att2 = AttentionGate3D(chs[1], chs[1], chs[1]//2)

        self.up1 = UpBlock3D(chs[1], chs[0])
        self.att1 = AttentionGate3D(chs[0], chs[0], chs[0]//2)

        self.final_conv = nn.Conv3d(chs[0], out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b, e4, self.att4)
        d3 = self.up3(d4, e3, self.att3)
        d2 = self.up2(d3, e2, self.att2)
        d1 = self.up1(d2, e1, self.att1)

        return self.final_conv(d1)

from torchinfo import summary
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResAttUNet3D(in_channels=4, out_channels=4, base_ch=16).to(device)
# summary(model, (1, 4, 128, 128, 128))


from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Optimizer with a slightly lower learning rate for fine-tuning
optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)

# Scheduler with a little more patience for fine-tuning
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',         # Reduce LR when validation iou plateaus
    factor=0.5,         # Halve the LR
    patience=3,         # Wait 3 epochs before reducing
    min_lr=1e-7
)

from monai.losses import DiceCELoss, HausdorffDTLoss

combined_loss = DiceCELoss(softmax=True) 

import torch
import torch.nn.functional as F
from monai.transforms import AsDiscrete
import numpy as np

# Define post-processing for predictions
post_pred = AsDiscrete(threshold=0.5)
post_label = AsDiscrete(threshold=0.5)

def compute_iou(preds, labels):
    """Compute mean IoU for batch predictions."""
    preds = preds > 0.5  # Convert to binary mask
    labels = labels > 0.5  # Convert to binary mask

    intersection = torch.logical_and(preds, labels).sum(dim=(2, 3, 4))  # Sum over spatial dims
    union = torch.logical_or(preds, labels).sum(dim=(2, 3, 4))  # Sum over spatial dims
    iou = intersection / (union + 1e-6)  # Avoid division by zero

    return iou.mean().item()


model_savepath = "/kaggle/working/mcar_net.pth"

# Early stopping parameters (based on val IoU)
patience = 200
best_val_iou = -float("inf")  # Initialize to a very low value
counter = 0

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_iou": [], "val_iou": []}

# Continue training for another 10 epochs
epochs = 150
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    total_correct, total_pixels = 0, 0
    iou_scores = []

    for batch in tqdm(train_loader):
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = combined_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        preds = post_pred(outputs)
        labels = post_label(labels)

        correct = (preds == labels).sum().item()
        total = torch.numel(labels)
        total_correct += correct
        total_pixels += total

        iou = compute_iou(preds, labels)
        iou_scores.append(iou)

    avg_train_loss = epoch_loss / len(train_loader)
    train_accuracy = total_correct / total_pixels * 100
    avg_train_iou = np.mean(iou_scores)

    history["train_loss"].append(avg_train_loss)
    history["train_acc"].append(train_accuracy)
    history["train_iou"].append(avg_train_iou)

    # Validation Step
    model.eval()
    val_loss = 0
    val_correct, val_pixels = 0, 0
    val_iou_scores = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            loss = combined_loss(outputs, labels)
            val_loss += loss.item()

            preds = post_pred(outputs)
            labels = post_label(labels)

            correct = (preds == labels).sum().item()
            total = torch.numel(labels)
            val_correct += correct
            val_pixels += total

            iou = compute_iou(preds, labels)
            val_iou_scores.append(iou)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_pixels * 100
    avg_val_iou = np.mean(val_iou_scores)

    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(val_accuracy)
    history["val_iou"].append(avg_val_iou)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Train IoU: {avg_train_iou:.4f} || Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | Val IoU: {avg_val_iou:.4f}")

    scheduler.step(avg_val_loss)

    # Early Stopping Check (based on val IoU)
    if avg_val_iou > best_val_iou:
        best_val_iou = avg_val_iou
        counter = 0
        torch.save(model.state_dict(), model_savepath)  # Save the best model
        print(" Model improved and saved!")
    else:
        counter += 1
        print(f"No improvement in validation IoU. Patience counter: {counter}/{patience}")

    if counter >= patience:
        print("Early stopping triggered! Training stopped.")
        break

print("Training completed! Best model updated and saved.")





