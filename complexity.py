"""
Parameter count and MACs of MCAR-Net (Section 5.1, Supplementary Table 1).

Expected output (ptflops 0.7.4, input 4 x 128 x 128 x 128):
    MCAR-Net: 6.57 M parameters, 141.91 GMac

Note: ptflops >= 0.7.5 changed its default counting backend and reports ~127.8 GMac for
the same network; the version is therefore pinned in requirements.txt.
The parameter count is the number of trainable parameter elements of the complete
network (attention and normalisation layers included; BatchNorm running statistics are
buffers and are not counted). No layer is frozen.
"""

import argparse

import torch
from ptflops import get_model_complexity_info

from model import ABLATION_VARIANTS, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="mcar_net", choices=list(ABLATION_VARIANTS))
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()

    model = build_model(args.variant)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        macs, params = get_model_complexity_info(model, (4, args.size, args.size, args.size),
                                                 as_strings=True, print_per_layer_stat=False)
    print(f"{args.variant}: trainable params {trainable / 1e6:.2f} M (total {total / 1e6:.2f} M)")
    print(f"ptflops: {params} parameters, {macs}")


if __name__ == "__main__":
    main()
