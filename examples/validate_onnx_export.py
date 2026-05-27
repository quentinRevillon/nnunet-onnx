"""
Validate that an ONNX export of a nnUNet model produces the same segmentation as PyTorch.

Steps:
  1. Export nnUNet checkpoint → model.onnx  (plans embedded in ONNX metadata)
  2. Run ONNX inference on a test image       (no plans.json needed)
  3. Run PyTorch inference on the same image
  4. Compute Dice between the two segmentations

Expected result: Dice ≥ 0.9999  (1-2 boundary voxels may differ due to float32 precision)

Usage:
    # Download sample image (spine-generic, sub-barcelona06, T2w, ~5 MB):
    wget -q -O image.nii.gz https://github.com/spine-generic/data-multi-subject/raw/main/sub-barcelona06/anat/sub-barcelona06_T2w.nii.gz

    python examples/validate_onnx_export.py \
        --checkpoint /path/to/fold_0/checkpoint_final.pth \
        --image      image.nii.gz \
        --output-dir /tmp/nnunet_onnx_test
"""

import argparse
import os
import sys
import time

import nibabel as nib
import numpy as np

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nnunet_onnx.export    import export
from nnunet_onnx.inference import infer_onnx, infer_pt


def dice(a, b):
    inter = (a & b).sum()
    denom = a.sum() + b.sum()
    return 2 * inter / denom if denom > 0 else 1.0


def main():
    parser = argparse.ArgumentParser(
        description='Export nnUNet checkpoint to ONNX and validate against PyTorch',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to fold_N/checkpoint_final.pth')
    parser.add_argument('--image', required=True,
                        help='Input NIfTI image (.nii.gz)')
    parser.add_argument('--output-dir', default='/tmp/nnunet_onnx_test',
                        help='Directory where model.onnx and segmentations are saved')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    onnx_path = os.path.join(args.output_dir, 'model.onnx')
    seg_onnx  = os.path.join(args.output_dir, 'seg_onnx.nii.gz')
    seg_pt    = os.path.join(args.output_dir, 'seg_pt.nii.gz')

    # ── Step 1: export ────────────────────────────────────────────────────────
    print('\n[1/3] Exporting nnUNet checkpoint → ONNX ...')
    export(args.checkpoint, onnx_path)

    img = nib.load(args.image)

    # ── Step 2: ONNX inference ────────────────────────────────────────────────
    print('\n[2/3] Running ONNX inference (no PyTorch) ...')
    t0  = time.perf_counter()
    seg = infer_onnx(img, onnx_path)      # plans read from ONNX metadata
    t1  = time.perf_counter()
    nib.save(seg, seg_onnx)
    print(f'  Done in {t1-t0:.1f}s  →  {seg_onnx}')

    # ── Step 3: PyTorch inference ─────────────────────────────────────────────
    print('\n[3/3] Running PyTorch inference ...')
    t2  = time.perf_counter()
    seg = infer_pt(img, args.checkpoint)  # checkpoint path only
    t3  = time.perf_counter()
    nib.save(seg, seg_pt)
    print(f'  Done in {t3-t2:.1f}s  →  {seg_pt}')

    # ── Comparison ────────────────────────────────────────────────────────────
    a = np.asarray(nib.load(seg_onnx).dataobj).astype(bool)
    b = np.asarray(nib.load(seg_pt).dataobj).astype(bool)
    d    = dice(a, b)
    diff = int((a != b).sum())

    print()
    print('─' * 40)
    print(f'  Dice        : {d:.4f}')
    print(f'  Diff voxels : {diff}')
    print('─' * 40)

    # 1-2 boundary voxels may differ due to float32 precision differences
    # between torch (PT path) and numpy (ONNX path) at voxels with probability ≈ 0.5
    if d >= 0.9999 and diff <= 5:
        print('\n  ✓  ONNX == PT\n')
    else:
        print('\n  ✗  Mismatch — check preprocessing\n')
        sys.exit(1)


if __name__ == '__main__':
    main()
