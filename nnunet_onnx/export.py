"""
Export a trained nnUNet model (PlainConvUNet) to ONNX format.

Takes a single checkpoint_final.pth file — no model folder or plans.json needed.
The preprocessing parameters (spacing, patch_size) are read directly from the checkpoint
and embedded into the ONNX metadata so that inference requires only the .onnx file.

Requires the [export] extra: pip install nnunet-onnx[export]

Usage:
    python -m nnunet_onnx.export \
        --checkpoint /path/to/fold_0/checkpoint_final.pth \
        --output     model.onnx
"""

import argparse
import json
import os

import onnx
import torch
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def _load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path, map_location='cpu', weights_only=False)


def _build_network(ckpt):
    ia = ckpt['init_args']
    plans_manager    = PlansManager(ia['plans'])
    cfg              = plans_manager.get_configuration(ia['configuration'])
    label_manager    = plans_manager.get_label_manager(ia['dataset_json'])
    n_input          = len(ia['dataset_json']['channel_names'])
    n_output         = label_manager.num_segmentation_heads

    net = get_network_from_plans(
        cfg.network_arch_class_name,
        cfg.network_arch_init_kwargs,
        cfg.network_arch_init_kwargs_req_import,
        n_input, n_output,
        deep_supervision=False,
    )
    net.load_state_dict(ckpt['network_weights'])
    net.eval()
    return net


def _extract_plans_for_inference(ckpt):
    """Return the flat plans dict used by nnunet_onnx.preprocessing.load_plans."""
    ia = ckpt['init_args']
    plans_manager = PlansManager(ia['plans'])
    cfg           = plans_manager.get_configuration(ia['configuration'])
    return {
        'target_spacing': cfg.spacing,
        'patch_size':     cfg.patch_size,
    }


def export(checkpoint_path, output):
    """Export a nnUNet checkpoint to ONNX with plans embedded in metadata.

    Args:
        checkpoint_path: path to fold_N/checkpoint_final.pth
        output:          destination .onnx file path
    """
    print(f'Loading checkpoint: {checkpoint_path}')
    ckpt = _load_checkpoint(checkpoint_path)

    net   = _build_network(ckpt)
    plans = _extract_plans_for_inference(ckpt)

    ia         = ckpt['init_args']
    plans_mgr  = PlansManager(ia['plans'])
    cfg        = plans_mgr.get_configuration(ia['configuration'])
    n_channels = len(ia['dataset_json']['channel_names'])
    dummy      = torch.randn(1, n_channels, *cfg.patch_size)

    print(f'Network : {type(net).__name__}')
    print(f'Input   : (1, {n_channels}, {cfg.patch_size[0]}, {cfg.patch_size[1]}, {cfg.patch_size[2]})')
    with torch.no_grad():
        out = net(dummy)
    print(f'Output  : {tuple(out.shape)}')

    torch.onnx.export(
        net, dummy, output,
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    )

    # Embed plans in ONNX metadata → inference needs only the .onnx file
    model_proto = onnx.load(output)
    entry       = model_proto.metadata_props.add()
    entry.key   = 'plans'
    entry.value = json.dumps(plans)
    onnx.save(model_proto, output)

    size_mb = os.path.getsize(output) / 1024 / 1024
    print(f'Exported → {output}  ({size_mb:.1f} MB)  [plans embedded in metadata]')


def main():
    parser = argparse.ArgumentParser(description='Export nnUNet checkpoint to ONNX')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to fold_N/checkpoint_final.pth')
    parser.add_argument('--output', default='model.onnx',
                        help='Output ONNX file path (default: model.onnx)')
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
