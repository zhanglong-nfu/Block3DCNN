#!/usr/bin/env python3
"""Prepare ImageCAS for BlockCNN reproducibility.

Faithful, dependency-light reimplementation of the preprocessing in
``train/custom/dataset/imagecas_dataset.py`` (``ImageCas_Dataset``). It does NOT
depend on the withheld BlockCNN core or the ``artoo`` framework -- only on
SimpleITK + torch -- so it can be run standalone to reproduce the exact 512^3
inputs the paper uses.

Usage:
    python prepare_imagecas.py --root /path/to/imagecas_root \
        --csv imagecas_ful.csv --out prepared_512 \
        --volume_size 512 512 512 --num_classes 3 \
        --win_level 200 --win_width 800
"""
import argparse
import os
import shutil
import sys

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


def load_volume(case_id, image_dir, label_dir):
    img_path = os.path.join(image_dir, f"imagecas_{case_id}_0000.nii.gz")
    lab_path = os.path.join(label_dir, f"imagecas_{case_id}.nii.gz")
    if not (os.path.exists(img_path) and os.path.exists(lab_path)):
        return None, None
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)
    if img.ndim == 4:  # (C, D, H, W) -> (D, H, W)
        img = img[0]
    lab = sitk.GetArrayFromImage(sitk.ReadImage(lab_path)).astype(np.int64)
    if lab.ndim == 4:
        lab = lab[0]
    return img, lab


def window(vol, level, width):
    lo = level - width / 2.0
    hi = level + width / 2.0
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo)
    return vol.astype(np.float32)


def remap_label(lab, num_classes, left=(1,), right=(2,)):
    if num_classes == 2:
        return (lab > 0).astype(np.int64)
    out = np.zeros_like(lab, dtype=np.int64)
    for v in left:
        out[lab == v] = 1
    for v in right:
        out[lab == v] = 2
    return out


def resize_volume(vol, volume_size, is_label):
    t = torch.from_numpy(vol.astype(np.float32))[None, None]  # (1,1,D,H,W)
    if is_label:
        t = F.interpolate(t, size=tuple(volume_size), mode="nearest")
        return t[0, 0].round().to(torch.int64).numpy()
    t = F.interpolate(t, size=tuple(volume_size), mode="trilinear",
                      align_corners=False)
    return t[0, 0].numpy().astype(np.float32)


def load_csv(csv_file, split=None):
    want = {"train": 1, "val": 2, "test": 3}.get(split) if split else None
    cases = []
    with open(csv_file) as f:
        f.readline()  # header: image_id,type
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            if want is not None:
                try:
                    if int(parts[1].strip()) != want:
                        continue
                except ValueError:
                    continue
            cases.append(parts[0].strip())
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True,
                    help="ImageCAS root containing Dataset001_CoronaryArtery")
    ap.add_argument("--csv", required=True,
                    help="case-split CSV (image_id,type)")
    ap.add_argument("--out", required=True,
                    help="output dir for preprocessed volumes")
    ap.add_argument("--dataset_dir", default="Dataset001_CoronaryArtery")
    ap.add_argument("--volume_size", nargs=3, type=int,
                    default=[512, 512, 512])
    ap.add_argument("--num_classes", type=int, default=3, choices=[2, 3])
    ap.add_argument("--win_level", type=float, default=200.0)
    ap.add_argument("--win_width", type=float, default=800.0)
    ap.add_argument("--split", default=None, choices=["train", "val", "test"])
    args = ap.parse_args()

    image_dir = os.path.join(args.root, args.dataset_dir, "imagesTr")
    label_dir = os.path.join(args.root, args.dataset_dir, "labelsTr")
    if not (os.path.isdir(image_dir) and os.path.isdir(label_dir)):
        sys.exit(f"ERROR: expected {image_dir} and {label_dir}")

    cases = load_csv(args.csv, args.split)
    print(f"[prepare_imagecas] {len(cases)} cases (split={args.split})")

    out_img = os.path.join(args.out, "imagesTr")
    out_lab = os.path.join(args.out, "labelsTr")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lab, exist_ok=True)

    missing = 0
    done = 0
    for cid in cases:
        img, lab = load_volume(cid, image_dir, label_dir)
        if img is None:
            missing += 1
            continue
        img = window(img, args.win_level, args.win_width)
        lab = remap_label(lab, args.num_classes)
        img = resize_volume(img, args.volume_size, is_label=False)
        lab = resize_volume(lab, args.volume_size, is_label=True)
        # image (1, D, H, W); label (D, H, W) stored int8
        sitk.WriteImage(
            sitk.GetImageFromArray(img[None]),
            os.path.join(out_img, f"imagecas_{cid}_0000.nii.gz"))
        sitk.WriteImage(
            sitk.GetImageFromArray(lab.astype(np.int8)),
            os.path.join(out_lab, f"imagecas_{cid}.nii.gz"))
        done += 1
    print(f"[prepare_imagecas] wrote {done} preprocessed volumes; "
          f"{missing} missing from release")

    shutil.copy(args.csv, os.path.join(args.out, os.path.basename(args.csv)))
    print(f"[prepare_imagecas] done -> {args.out}")


if __name__ == "__main__":
    main()
