"""Extract a VGGT point cloud + intrinsics from one input image.

Writes:
  <out_dir>/points.bin   -- uint32 N || N*float32[3] xyz || N*uint8[3] rgb
  <out_dir>/meta.json    -- {focals, principal, image_h, image_w}
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vggt"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print("loading VGGT...", flush=True)
    model = VGGT.from_pretrained(args.model_path).to(device)

    print("loading image...", flush=True)
    img = Image.open(args.image).convert("RGB")
    first_frame = load_and_preprocess_images([img, img]).to(device)

    print("running VGGT inference...", flush=True)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            preds = model(first_frame)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                preds["pose_enc"], first_frame.shape[-2:]
            )
            world_points = preds["world_points"][0][0]  # HxWx3
            focals = intrinsic[0][0][:2, :2].diag().detach().cpu().numpy().tolist()
            principal = intrinsic[0][0][:2, 2].detach().cpu().numpy().tolist()

    raw = first_frame[0].detach().cpu().numpy().transpose(1, 2, 0)  # HxWx3 in [0,1]
    H, W = raw.shape[:2]

    xyz = world_points.detach().cpu().float().numpy().reshape(-1, 3).astype(np.float32)
    rgb = (raw.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)

    # Drop NaN / inf points and any obvious outliers far away
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgb = rgb[finite]

    n = xyz.shape[0]
    print(f"writing {n} points...", flush=True)
    with open(os.path.join(args.out_dir, "points.bin"), "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(xyz.tobytes(order="C"))
        f.write(rgb.tobytes(order="C"))

    meta = {
        "focals": focals,
        "principal": principal,
        "image_h": H,
        "image_w": W,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    print("done.", flush=True)


if __name__ == "__main__":
    main()
