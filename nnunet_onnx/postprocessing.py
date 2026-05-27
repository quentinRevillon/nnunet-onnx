"""
nnUNet postprocessing primitives (numpy only).

Usage:
    from nnunet_onnx.postprocessing import pad_back, softmax_threshold
"""

import numpy as np


def pad_back(pred_cropped, bbox, full_shape):
    """Restore a cropped prediction to its original full_shape."""
    out    = np.zeros(full_shape, dtype=pred_cropped.dtype)
    slicer = tuple(slice(b[0], b[1]) for b in bbox)
    out[slicer] = pred_cropped
    return out


def softmax_threshold(logits, threshold=0.5):
    """Apply softmax over class axis (dim 0) and threshold class-1 probability.

    Matches nnUNet's convert_predicted_logits_to_segmentation:
      logits shape : (n_classes, Z, Y, X)
      returns      : binary uint8 array (Z, Y, X)
    """
    exp   = np.exp(logits - logits.max(axis=0, keepdims=True))
    prob1 = exp[1] / exp.sum(axis=0)
    return (prob1 > threshold).astype(np.uint8)
