"""
nnUNet sliding-window inference: ONNX (no PyTorch) and PyTorch modes.

Both functions take a nibabel image in any orientation and return a binary
segmentation mask in the same space/orientation.

Usage:
    from nnunet_onnx.inference import infer_onnx, infer_pt

    seg = infer_onnx(nib.load('image.nii.gz'), 'model.onnx')
    nib.save(seg, 'seg.nii.gz')
"""

import glob
import json
import tempfile

import nibabel as nib
import numpy as np
from nibabel.orientations import io_orientation
from scipy.ndimage import gaussian_filter

from .postprocessing import pad_back, softmax_threshold
from .preprocessing import (
    compute_new_shape,
    crop_to_nonzero,
    get_voxel_spacing_zyx,
    load_plans,
    reorient_back,
    reorient_to_rpi,
    resample,
    zscore_normalize,
)


# ── Sliding window ────────────────────────────────────────────────────────────

def _make_gaussian_map(patch_size, sigma_scale=1.0 / 8):
    tmp = np.zeros(patch_size, dtype=np.float64)
    tmp[tuple(i // 2 for i in patch_size)] = 1
    g = gaussian_filter(tmp, [i * sigma_scale for i in patch_size], mode='constant', cval=0)
    g /= g.max()
    g[g == 0] = g[g > 0].min()
    return g.astype(np.float32)


def _compute_steps(image_size, patch_size, tile_step):
    steps = []
    for img_s, patch_s in zip(image_size, patch_size):
        if img_s <= patch_s:
            steps.append([0])
            continue
        n       = int(np.ceil((img_s - patch_s) / (patch_s * tile_step))) + 1
        max_val = img_s - patch_s
        actual  = max_val / (n - 1) if n > 1 else 99999
        steps.append([int(np.round(actual * i)) for i in range(n)])
    return steps


def _sliding_window(data, session, patch_size, tile_step):
    """Sliding window with Gaussian weighting — accumulates logits, matches nnUNet exactly."""
    D, H, W = data.shape
    pd, ph, pw = patch_size

    pad_d = max(0, pd - D); d0 = pad_d // 2; d1 = pad_d - d0
    pad_h = max(0, ph - H); h0 = pad_h // 2; h1 = pad_h - h0
    pad_w = max(0, pw - W); w0 = pad_w // 2; w1 = pad_w - w0
    if pad_d or pad_h or pad_w:
        data = np.pad(data, ((d0, d1), (h0, h1), (w0, w1)), mode='constant')
    Dp, Hp, Wp = data.shape

    gauss      = _make_gaussian_map(patch_size)
    accum      = np.zeros((2, Dp, Hp, Wp), dtype=np.float32)
    weight_map = np.zeros((Dp, Hp, Wp), dtype=np.float32)
    in_name    = session.get_inputs()[0].name
    out_name   = session.get_outputs()[0].name

    for dz in _compute_steps((Dp, Hp, Wp), patch_size, tile_step)[0]:
        for dy in _compute_steps((Dp, Hp, Wp), patch_size, tile_step)[1]:
            for dx in _compute_steps((Dp, Hp, Wp), patch_size, tile_step)[2]:
                patch  = data[dz:dz+pd, dy:dy+ph, dx:dx+pw][np.newaxis, np.newaxis]
                logits = session.run([out_name], {in_name: patch})[0][0]
                accum      [:, dz:dz+pd, dy:dy+ph, dx:dx+pw] += logits * gauss
                weight_map [dz:dz+pd, dy:dy+ph, dx:dx+pw]    += gauss

    return (accum / weight_map)[:, d0:d0+D, h0:h0+H, w0:w0+W]  # (2, D, H, W)


# ── ONNX inference ────────────────────────────────────────────────────────────

def _read_plans_from_onnx(session):
    """Read plans embedded in ONNX metadata (set by export.py)."""
    meta = session.get_modelmeta().custom_metadata_map
    return json.loads(meta['plans'])


def infer_onnx(img, model_path, tile_step=0.5, threads=None):
    """Run nnUNet inference via ONNX Runtime (no PyTorch).

    Plans (target spacing, patch size) are read directly from the ONNX metadata
    embedded at export time — no external plans.json needed.

    Args:
        img:        nibabel NIfTI image, any orientation.
        model_path: path to .onnx model file (exported with nnunet_onnx.export).
        tile_step:  sliding window step as fraction of patch size (default 0.5).
        threads:    ONNX Runtime intra-op threads (default: auto).

    Returns:
        nibabel NIfTI1Image — binary segmentation in the same space/orientation as img.
    """
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    if threads:
        sess_options.intra_op_num_threads = threads
    session = ort.InferenceSession(model_path, sess_options=sess_options,
                                   providers=['CPUExecutionProvider'])

    plans      = _read_plans_from_onnx(session)   # {'target_spacing': ..., 'patch_size': ...}
    target_sp  = plans['target_spacing']
    patch_size = plans['patch_size']

    orig_ornt = io_orientation(img.affine)
    img_rpi   = reorient_to_rpi(img)

    data         = img_rpi.get_fdata().transpose((2, 1, 0)).astype(np.float32)
    orig_spacing = get_voxel_spacing_zyx(img_rpi)
    shape_before = data.shape

    data_cropped, bbox = crop_to_nonzero(data)
    data_norm          = zscore_normalize(data_cropped)
    new_shape          = compute_new_shape(data_cropped.shape, orig_spacing, target_sp)
    data_rs            = resample(data_norm, new_shape, orig_spacing, target_sp, order=3, order_z=0)

    avg_logits = _sliding_window(data_rs, session, patch_size, tile_step)

    logits_back = np.stack([
        resample(avg_logits[c], data_cropped.shape, target_sp, orig_spacing, order=1, order_z=0)
        for c in range(2)
    ])
    pred_zyx = pad_back(softmax_threshold(logits_back), bbox, shape_before)
    pred_xyz = pred_zyx.transpose((2, 1, 0)).astype(np.uint8)

    seg_rpi = nib.Nifti1Image(pred_xyz, img_rpi.affine)
    return reorient_back(seg_rpi, orig_ornt)


# ── PyTorch inference ─────────────────────────────────────────────────────────

def infer_pt(img, checkpoint_path, device='cpu', use_mirroring=False):
    """Run nnUNet inference via PyTorch (requires nnunetv2 + torch).

    Args:
        img:             nibabel NIfTI image, any orientation.
        checkpoint_path: path to fold_N/checkpoint_final.pth
                         The model folder (parent of fold_N/) must contain plans.json
                         and dataset.json — standard nnUNet training output structure.
        device:          'cpu', 'cuda', or 'mps' (default 'cpu').
        use_mirroring:   enable test-time augmentation (8× slower, default False).

    Returns:
        nibabel NIfTI1Image — binary segmentation in the same space/orientation as img.
    """
    import os
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    fold_dir     = os.path.dirname(checkpoint_path)
    model_folder = os.path.dirname(fold_dir)
    fold_num     = int(os.path.basename(fold_dir).replace('fold_', ''))
    ckpt_name    = os.path.basename(checkpoint_path)

    orig_ornt = io_orientation(img.affine)
    img_rpi   = reorient_to_rpi(img)

    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                        use_mirroring=use_mirroring,
                        perform_everything_on_device=(device != 'cpu'),
                        device=torch.device(device),
                        verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(model_folder, use_folds=[fold_num],
                                           checkpoint_name=ckpt_name)

    crop_tmp = tempfile.mktemp(suffix='_0000.nii.gz')
    nib.save(img_rpi, crop_tmp)
    tmpdir = tempfile.mkdtemp()
    p.predict_from_files([[crop_tmp]], tmpdir, save_probabilities=False, overwrite=True,
                         num_processes_preprocessing=1, num_processes_segmentation_export=1)

    pred_arr = np.asarray(nib.load(glob.glob(tmpdir + '/*.nii.gz')[0]).dataobj).astype(np.uint8)
    os.remove(crop_tmp)

    seg_rpi = nib.Nifti1Image(pred_arr, img_rpi.affine)
    return reorient_back(seg_rpi, orig_ornt)


# ── CLI entry points ──────────────────────────────────────────────────────────

def _cli_compare():
    """Entry point for: nnunet-compare"""
    import argparse
    parser = argparse.ArgumentParser(description='Compare two segmentation masks (Dice)')
    parser.add_argument('seg_a', help='First segmentation (.nii.gz)')
    parser.add_argument('seg_b', help='Second segmentation (.nii.gz)')
    args = parser.parse_args()

    a = np.asarray(nib.load(args.seg_a).dataobj).astype(bool)
    b = np.asarray(nib.load(args.seg_b).dataobj).astype(bool)

    inter = (a & b).sum()
    denom = a.sum() + b.sum()
    dice  = 2 * inter / denom if denom > 0 else 1.0
    diff  = int((a != b).sum())

    print(f'Dice        : {dice:.4f}')
    print(f'Diff voxels : {diff}')


def _cli_onnx():
    """Entry point for: nnunet-segment-onnx"""
    import argparse, time
    import onnxruntime as ort
    from nibabel.orientations import io_orientation
    from .preprocessing import (reorient_to_rpi, reorient_back, get_voxel_spacing_zyx,
                                 crop_to_nonzero, zscore_normalize, resample, compute_new_shape)
    from .postprocessing import pad_back, softmax_threshold

    parser = argparse.ArgumentParser(description='nnUNet ONNX inference (no PyTorch)')
    parser.add_argument('-i',          required=True, help='Input NIfTI image')
    parser.add_argument('-o',          required=True, help='Output segmentation mask')
    parser.add_argument('--model',     required=True, help='Path to model.onnx')
    parser.add_argument('--tile-step', type=float, default=0.5)
    parser.add_argument('--threads',   type=int,   default=None)
    args = parser.parse_args()

    SEP = '─' * 38

    # ── preprocessing ─────────────────────────────────────────────────────────
    t0  = time.perf_counter()
    img = nib.load(args.i)

    sess_options = ort.SessionOptions()
    if args.threads:
        sess_options.intra_op_num_threads = args.threads
    session    = ort.InferenceSession(args.model, sess_options=sess_options,
                                      providers=['CPUExecutionProvider'])
    plans      = _read_plans_from_onnx(session)
    target_sp  = plans['target_spacing']
    patch_size = plans['patch_size']

    orig_ornt    = io_orientation(img.affine)
    img_rpi      = reorient_to_rpi(img)
    data         = img_rpi.get_fdata().transpose((2, 1, 0)).astype(np.float32)
    orig_spacing = get_voxel_spacing_zyx(img_rpi)
    shape_before = data.shape
    data_cropped, bbox = crop_to_nonzero(data)
    data_norm          = zscore_normalize(data_cropped)
    new_shape          = compute_new_shape(data_cropped.shape, orig_spacing, target_sp)
    data_rs            = resample(data_norm, new_shape, orig_spacing, target_sp, order=3, order_z=0)
    t1 = time.perf_counter()

    # ── sliding window ─────────────────────────────────────────────────────────
    steps     = _compute_steps(list(data_rs.shape), patch_size, args.tile_step)
    n_patches = len(steps[0]) * len(steps[1]) * len(steps[2])
    avg_logits = _sliding_window(data_rs, session, patch_size, args.tile_step)
    t2 = time.perf_counter()

    # ── postprocessing ─────────────────────────────────────────────────────────
    logits_back = np.stack([
        resample(avg_logits[c], data_cropped.shape, target_sp, orig_spacing, order=1, order_z=0)
        for c in range(2)
    ])
    pred_zyx = pad_back(softmax_threshold(logits_back), bbox, shape_before)
    seg_rpi  = nib.Nifti1Image(pred_zyx.transpose((2, 1, 0)).astype(np.uint8), img_rpi.affine)
    seg      = reorient_back(seg_rpi, orig_ornt)
    nib.save(seg, args.o)
    t3 = time.perf_counter()

    print(SEP)
    print(f'  preprocessing  : {t1-t0:.1f}s')
    print(f'  inference      : {t2-t1:.1f}s  ({n_patches} patches)')
    print(f'  postprocessing : {t3-t2:.1f}s')
    print(SEP)
    print(f'  total          : {t3-t0:.1f}s  →  {args.o}')


def _cli_pt():
    """Entry point for: nnunet-segment-pt"""
    import argparse, os, io, contextlib, time, tempfile
    from nibabel.orientations import io_orientation
    from .preprocessing import reorient_to_rpi, reorient_back

    # Set env vars before nnunetv2 import to suppress its startup warnings
    os.environ.setdefault('nnUNet_raw', 'dummy')
    os.environ.setdefault('nnUNet_preprocessed', 'dummy')
    os.environ.setdefault('nnUNet_results', 'dummy')
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    parser = argparse.ArgumentParser(description='nnUNet PyTorch inference')
    parser.add_argument('-i',           required=True, help='Input NIfTI image')
    parser.add_argument('-o',           required=True, help='Output segmentation mask')
    parser.add_argument('--checkpoint', required=True, help='Path to fold_N/checkpoint_*.pth')
    parser.add_argument('--device',     default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--tta',        action='store_true', help='Enable mirroring TTA (8× slower)')
    args = parser.parse_args()

    SEP = '─' * 38

    fold_dir     = os.path.dirname(args.checkpoint)
    model_folder = os.path.dirname(fold_dir)
    fold_num     = int(os.path.basename(fold_dir).replace('fold_', ''))
    ckpt_name    = os.path.basename(args.checkpoint)

    # ── load model ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                            use_mirroring=args.tta,
                            perform_everything_on_device=(args.device != 'cpu'),
                            device=torch.device(args.device),
                            verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        p.initialize_from_trained_model_folder(model_folder, use_folds=[fold_num],
                                               checkpoint_name=ckpt_name)
    t1 = time.perf_counter()

    # ── inference (preproc + network + postproc) ───────────────────────────────
    img       = nib.load(args.i)
    orig_ornt = io_orientation(img.affine)
    img_rpi   = reorient_to_rpi(img)
    crop_tmp  = tempfile.mktemp(suffix='_0000.nii.gz')
    nib.save(img_rpi, crop_tmp)
    tmpdir = tempfile.mkdtemp()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        p.predict_from_files([[crop_tmp]], tmpdir, save_probabilities=False, overwrite=True,
                             num_processes_preprocessing=1, num_processes_segmentation_export=1)
    t2 = time.perf_counter()

    pred_arr = np.asarray(nib.load(glob.glob(tmpdir + '/*.nii.gz')[0]).dataobj).astype(np.uint8)
    os.remove(crop_tmp)
    seg = reorient_back(nib.Nifti1Image(pred_arr, img_rpi.affine), orig_ornt)
    nib.save(seg, args.o)
    t3 = time.perf_counter()

    print(SEP)
    print(f'  model loading  : {t1-t0:.1f}s')
    print(f'  inference      : {t2-t1:.1f}s  (preproc + network + postproc)')
    print(f'  save           : {t3-t2:.1f}s')
    print(SEP)
    print(f'  total          : {t3-t0:.1f}s  →  {args.o}')
