#!/usr/bin/env python
"""Export a trained DMNestUnet PyTorch checkpoint to a fixed-shape ONNX graph.

Only the L1 exit is exported: DMUnet.py's forward(x, L) computes each nested
level sequentially and returns early, so fixing L=1 also gives the *cheapest*
sub-network (see DMUnet.py / CLAUDE.md for what L1/L2/L3 mean) -- the obvious
choice for an edge deployment target. The graph takes the network's native
4-channel [R, G1, G2, B] packed input (utils.rgb2RGGB's layout, i.e. the
*GBRG*-phase Bayer mosaic packed into planes at half the raw image's spatial
resolution -- see CLAUDE.md's "Architecture" section) and produces a 3-channel
RGB output at 2x that height/width (DMNestUnet's final upsample doubles
resolution back to the raw image's own size). No gamma is baked into the
graph; that domain bridging is done externally, the same way test.py does it
for the full (untiled) L1/L2/L3 script -- see tile_infer.py for the tiled
equivalent that runs this exported graph.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config as config_module

# DMUnet.py reads Config.DEVICE once, at import time, to place the
# (non-buffer, non-parameter) Gaussian-blur kernel tensor -- net.to(device)
# later would NOT move it, since it's a plain attribute, not a registered
# buffer. Exporting is always done on CPU, so patch this before importing
# DMUnet (same trick test.py uses for the same reason).
config_module.Config.DEVICE = torch.device('cpu')
from DMUnet import DMNestUnet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a trained DMNestUnet checkpoint (L1 exit only) to a fixed-shape ONNX graph")
    parser.add_argument('--model', type=Path, help='path to the pytorch checkpoint (state_dict), e.g. pretrained_model/params_e123.pth')
    parser.add_argument('--output', type=Path, default=None,
                        help='output .onnx path; defaults to <model>_<dtype>_<height>x<width>_L1.onnx next to the checkpoint')
    parser.add_argument('--height', type=int, default=256,
                        help='network input height, in packed [R,G1,G2,B] plane space (i.e. half the raw '
                             'Bayer image height); baked into the exported graph. Unlike a spatial-pyramid '
                             'UNet, DMNestUnet never downsamples (see CLAUDE.md), so there is no '
                             'divisibility requirement on this value.')
    parser.add_argument('--width', type=int, default=256,
                        help='network input width, in packed [R,G1,G2,B] plane space (i.e. half the raw '
                             'Bayer image width); baked into the exported graph.')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version')
    parser.add_argument(
        '--dtype', type=str, default='fp32', choices=['fp32', 'fp16'],
        help="data type for the exported graph's weights and input/output tensors. fp32 is the normal "
             "export. fp16 exports fp32 first, then casts the *exported graph* (weights + tensor dtypes) "
             "to float16 as a separate graph-rewrite step -- CPU PyTorch has no fast fp16 Conv2d kernel, "
             "so the model is never actually run in fp16 during export/tracing.",
    )
    return parser.parse_args()


class Level1Module(nn.Module):
    """Wraps DMNestUnet to fix L=1 so the exported graph has exactly one
    static output tensor (forward(x, L=1) always returns a plain tensor --
    see DMUnet.py -- unlike L>=2, which return a list under deep supervision)."""

    def __init__(self, net: DMNestUnet):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x, L=1)


def load_net(model_path: Path) -> DMNestUnet:
    net = DMNestUnet(in_channels=config_module.Config.INP, n_classes=config_module.Config.OUP)
    state_dict = torch.load(str(model_path), map_location='cpu')
    net.load_state_dict(state_dict)
    return net.eval()


def main():
    opt = parse_args()

    net = load_net(opt.model)
    model = Level1Module(net).eval()
    # DMNestUnet's native input: 4-channel [R,G1,G2,B] packed mosaic (GBRG phase)
    dummy = torch.randn(1, config_module.Config.INP, opt.height, opt.width, dtype=torch.float32)

    output_path = opt.output or opt.model.with_name(f'{opt.model.stem}_{opt.dtype}_{opt.height}x{opt.width}_L1.onnx')

    # Always trace/export in float32 (see --dtype help above for why); for --dtype fp16 we
    # export to a temp fp32 file first, then convert that graph to fp16 and remove the temp file.
    export_path = output_path if opt.dtype == 'fp32' else output_path.with_suffix('.fp32tmp.onnx')
    torch.onnx.export(
        model, (dummy,), str(export_path),
        input_names=['input'], output_names=['output'],
        opset_version=opt.opset,
        dynamo=False,  # this graph is static (no control flow); skip the dynamo exporter to avoid the extra onnxscript dependency
    )

    if opt.dtype == 'fp16':
        import onnx
        from onnxconverter_common import float16 as onnx_float16
        model_fp16 = onnx_float16.convert_float_to_float16(onnx.load(str(export_path)), keep_io_types=False)
        onnx.save(model_fp16, str(output_path))
        export_path.unlink()

    out_h, out_w = 2 * opt.height, 2 * opt.width
    print(f'Exported to {output_path} (dtype={opt.dtype}, '
          f'input=1x{config_module.Config.INP}x{opt.height}x{opt.width}, '
          f'output=1x{config_module.Config.OUP}x{out_h}x{out_w})')

    # verify the exported graph reproduces the PyTorch (float32) output, within the dtype's precision
    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
    onnx_dtype = np.float16 if opt.dtype == 'fp16' else np.float32
    onnx_out = sess.run(None, {'input': dummy.numpy().astype(onnx_dtype)})[0]
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    diff = np.abs(onnx_out.astype(np.float64) - torch_out.astype(np.float64)).max()
    print(f'ONNX ({opt.dtype}) vs PyTorch float32 max abs diff: {diff:.3e}')


if __name__ == '__main__':
    main()
