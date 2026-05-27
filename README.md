# nnunet-onnx — Export nnUNet models to ONNX

[nnUNet](https://github.com/MIC-DKFZ/nnUNet) trains state-of-the-art segmentation models but its inference stack requires PyTorch and the full nnunetv2 package (~2 GB). **nnunet-onnx** lets you export any trained nnUNet checkpoint to a single self-contained `.onnx` file and run inference with only `onnxruntime` — no PyTorch, no nnunetv2.

- **`export`** — convert `checkpoint_*.pth` → `model.onnx`. Preprocessing parameters (spacing, patch size) are read directly from the checkpoint and embedded in the ONNX metadata. No `plans.json` required.
- **`inference`** — full nnUNet preprocessing + sliding-window + postprocessing reimplemented in pure numpy/scipy. Takes any NIfTI image (any orientation), returns a binary segmentation mask in the same space.

---

## Install

```bash
conda create -n nnunet_onnx python=3.10 && conda activate nnunet_onnx
pip install git+https://github.com/quentinRevillon/nnunet-onnx.git
```

---

## Tutorial

### Step 1 — Download a test image

Same image as the [sc-crop tutorial](https://github.com/ivadomed/sc-crop):

```bash
mkdir ~/nnunet-onnx-test && cd ~/nnunet-onnx-test
curl -L https://github.com/spinalcordtoolbox/sct_tutorial_data/releases/download/r20260508/data_spinalcord-segmentation.zip -o sct_tutorial.zip
unzip sct_tutorial.zip
```

The T2 image is at `single_subject/data/t2/t2.nii.gz`.

### Step 2 — Download a nnUNet checkpoint

This example uses the [contrast-agnostic spinal cord segmentation model v3.0](https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/releases/tag/v3.0) — a publicly released nnUNet model trained on 20+ MRI datasets:

```bash
curl -L https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/releases/download/v3.0/model_contrast_agnostic_20250123.zip -o model.zip
unzip model.zip
```

The checkpoint is at:

```
model_contrast_agnostic_20250123/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
```

### Step 3 — Export the checkpoint to ONNX

```bash
python -m nnunet_onnx.export \
    --checkpoint model_contrast_agnostic_20250123/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth \
    --output     model.onnx
```

This produces a single `model.onnx` file (~115 MB). Preprocessing parameters are embedded in its metadata — nothing else is needed for inference.

### Step 4 — Segment with ONNX

```bash
nnunet-segment-onnx \
    -i single_subject/data/t2/t2.nii.gz \
    -o seg_onnx.nii.gz \
    --model model.onnx
```

### Step 5 — Segment with PyTorch and compare

```bash
nnunet-segment-pt \
    -i single_subject/data/t2/t2.nii.gz \
    -o seg_pt.nii.gz \
    --checkpoint model_contrast_agnostic_20250123/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
```

Both commands print the elapsed time, so you can directly compare. ONNX Runtime is faster per inference patch (~18% on CPU on this image, 9 patches). The gain scales with the number of patches — larger or higher-resolution images will show a bigger absolute speedup. The ONNX advantage also includes **deployment**: no PyTorch, no nnunetv2, lighter environment.

### Step 6 — Compare the two segmentations

```bash
nnunet-compare seg_onnx.nii.gz seg_pt.nii.gz
```

```
Dice        : 1.0000
Diff voxels : 0
```

---

## Python API

```python
import nibabel as nib
from nnunet_onnx import infer_onnx, infer_pt

img = nib.load('image.nii.gz')

# ONNX — no PyTorch, no nnunetv2
seg = infer_onnx(img, 'model.onnx')

# PyTorch — reference inference from the original checkpoint
seg = infer_pt(img, 'fold_0/checkpoint_best.pth')

nib.save(seg, 'seg.nii.gz')
```

Both functions accept any NIfTI image (any orientation) and return a binary segmentation mask in the same space and orientation as the input.

---

## How it works

### Export

`export.py` loads the checkpoint, reads the network architecture and preprocessing parameters from `checkpoint['init_args']`, and exports only the neural network (no preprocessing) to ONNX opset 14. The preprocessing parameters are embedded in the ONNX metadata so the model file is self-contained.

```
checkpoint_best.pth
  └─ init_args
       ├─ network architecture  →  ONNX graph
       ├─ spacing               →  ONNX metadata
       └─ patch_size            →  ONNX metadata
```

### Inference

The preprocessing matches nnUNet's pipeline exactly — validated at Dice = 1.0 against `nnUNetPredictor`:

1. Reorient to RPI
2. `crop_to_nonzero`
3. Z-score normalisation
4. Resample to target spacing (isotropic: `skimage.resize` order=3 ; anisotropic: separate-Z with `scipy.map_coordinates` order=0)
5. Sliding-window inference with Gaussian weighting — logits accumulated, not probabilities
6. Resample logits back (order=1) → softmax → threshold 0.5
7. `pad_back` → reorient to original orientation

---

## Compatibility

The preprocessing reads two fields from the ONNX metadata set at export time:
- `target_spacing` — resampling target
- `patch_size` — sliding window size

These fields are stable across nnUNet v2.x. Pin the nnUNet version used for export (`nnunetv2==X.Y.Z` in `pyproject.toml`) and distribute `model.onnx` as a versioned artefact. The original checkpoint is not needed after export.

---

## Requirements

Python ≥ 3.9. Installed by `pip install -e .`: `nibabel`, `numpy`, `scipy`, `scikit-image`, `onnxruntime`, `onnx`.

Python ≥ 3.9. All dependencies installed automatically: `nibabel`, `numpy`, `scipy`, `scikit-image`, `onnxruntime`, `onnx`, `torch`, `nnunetv2`.
