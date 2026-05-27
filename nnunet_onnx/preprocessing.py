"""
nnUNet preprocessing primitives (no PyTorch dependency).

Matches nnUNet's resample_data_or_seg_to_shape exactly:
  - Isotropic images  : skimage.transform.resize with order=3/0, mode='edge'
  - Anisotropic images: skimage.transform.resize in-plane + scipy.ndimage.map_coordinates
                        on the thick axis with order_z=0 (nearest-neighbour)

Usage:
    from nnunet_onnx.preprocessing import load_plans, preprocess
"""

import json

import numpy as np
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from scipy.ndimage import binary_fill_holes, map_coordinates
from skimage.transform import resize as sk_resize

_ANISO_THRESHOLD = 3


# ── Plans ─────────────────────────────────────────────────────────────────────

def load_plans(plans_path):
    """Load target_spacing and patch_size from plans.json."""
    plans = json.load(open(plans_path))
    cfg = plans['configurations']['3d_fullres']
    return {
        'target_spacing': cfg['spacing'],
        'patch_size':     cfg['patch_size'],
    }


# ── Reorientation ─────────────────────────────────────────────────────────────

def reorient_to_rpi(img):
    target  = axcodes2ornt(('R', 'P', 'I'))
    current = io_orientation(img.affine)
    return img.as_reoriented(ornt_transform(current, target))


def reorient_back(img_rpi, original_ornt):
    rpi_ornt = axcodes2ornt(('R', 'P', 'I'))
    return img_rpi.as_reoriented(ornt_transform(rpi_ornt, original_ornt))


# ── Spacing ───────────────────────────────────────────────────────────────────

def get_voxel_spacing_zyx(img):
    return [float(v) for v in img.header.get_zooms()[:3][::-1]]


def compute_new_shape(orig_shape, orig_spacing, target_spacing):
    return tuple(int(round(s * o / t))
                 for s, o, t in zip(orig_shape, orig_spacing, target_spacing))


# ── Normalisation ─────────────────────────────────────────────────────────────

def crop_to_nonzero(data):
    """Crop 3D array to its non-zero bounding box. Returns (cropped, bbox)."""
    mask = binary_fill_holes(data != 0)
    bbox = []
    for ax in range(3):
        proj    = np.any(mask, axis=tuple(i for i in range(3) if i != ax))
        indices = np.where(proj)[0]
        bbox.append([int(indices[0]), int(indices[-1]) + 1] if len(indices) > 0
                    else [0, data.shape[ax]])
    return data[tuple(slice(b[0], b[1]) for b in bbox)], bbox


def zscore_normalize(data):
    mean = data.mean()
    std  = data.std()
    return ((data - mean) / max(float(std), 1e-8)).astype(np.float32)


# ── Resampling ────────────────────────────────────────────────────────────────

def _get_lowres_axis(spacing):
    return np.where(max(spacing) / np.array(spacing) == 1)[0]


def _do_separate_z(spacing):
    return (max(spacing) / min(spacing)) > _ANISO_THRESHOLD


def _determine_sep_z_axis(current_spacing, new_spacing):
    if _do_separate_z(current_spacing):
        axis = _get_lowres_axis(current_spacing)
    elif _do_separate_z(new_spacing):
        axis = _get_lowres_axis(new_spacing)
    else:
        return False, None
    if len(axis) in (2, 3):
        return False, None
    return True, int(axis[0])


def resample(data, new_shape, current_spacing, new_spacing, order=3, order_z=0):
    """Resample 3D array to new_shape, matching nnUNet's resample_data_or_seg_to_shape.

    Args:
        data:            3D numpy array in ZYX order.
        new_shape:       Target shape (Z, Y, X).
        current_spacing: Voxel spacing of data in ZYX order.
        new_spacing:     Target voxel spacing in ZYX order.
        order:           Interpolation order for in-plane axes (default 3).
        order_z:         Interpolation order for the anisotropic axis (default 0).
    """
    old_shape = np.array(data.shape)
    new_shape  = np.array(new_shape)

    if np.all(old_shape == new_shape):
        return data.astype(np.float32)

    do_sep, axis = _determine_sep_z_axis(current_spacing, new_spacing)
    data_f = data.astype(float)

    if not do_sep:
        return sk_resize(data_f, new_shape, order=order,
                         mode='edge', anti_aliasing=False).astype(np.float32)

    if axis == 0:
        new_shape_2d = new_shape[1:]
    elif axis == 1:
        new_shape_2d = new_shape[[0, 2]]
    else:
        new_shape_2d = new_shape[:-1]

    tmp_shape = new_shape.copy()
    tmp_shape[axis] = old_shape[axis]
    reshaped = np.zeros(tmp_shape, dtype=float)
    for idx in range(old_shape[axis]):
        if axis == 0:
            reshaped[idx] = sk_resize(data_f[idx], new_shape_2d, order=order,
                                      mode='edge', anti_aliasing=False)
        elif axis == 1:
            reshaped[:, idx] = sk_resize(data_f[:, idx], new_shape_2d, order=order,
                                         mode='edge', anti_aliasing=False)
        else:
            reshaped[:, :, idx] = sk_resize(data_f[:, :, idx], new_shape_2d, order=order,
                                            mode='edge', anti_aliasing=False)

    if old_shape[axis] == new_shape[axis]:
        return reshaped.astype(np.float32)

    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
    orig_rows, orig_cols, orig_dim = reshaped.shape
    row_scale = float(orig_rows) / rows
    col_scale = float(orig_cols) / cols
    dim_scale = float(orig_dim)  / dim

    map_r, map_c, map_d = np.mgrid[:rows, :cols, :dim]
    map_r = row_scale * (map_r + 0.5) - 0.5
    map_c = col_scale * (map_c + 0.5) - 0.5
    map_d = dim_scale * (map_d + 0.5) - 0.5

    coord_map = np.array([map_r, map_c, map_d])
    return map_coordinates(reshaped, coord_map, order=order_z, mode='nearest').astype(np.float32)
