#!/usr/bin/env python
"""Tiled inference through a fixed-shape DMNestUnet-L1 ONNX export (see
export_onnx.py), over real (arbitrary-size) raw Bayer mosaic files.

Expected input, domain bridging (gamma) and Bayer-phase alignment are the
same as test.py (see its module docstring for the full rationale) -- this
script exists because export_onnx.py bakes a fixed --height/--width into the
graph, so a real (typically much larger) raw image has to be split into
overlapping tiles, each tile run through the fixed-shape graph, and the
non-overlapping "ownership" region of each tile's output stitched back
together. The tile_positions/ownership_bounds scheme below is the same one
used by ../../AIDenoise_research/PMRID/tile_infer.py, with one structural
difference: PMRID's network outputs at the *same* resolution as its packed
input, while DMNestUnet's L1 exit *upsamples 2x* (DMUnet.py's
final_upSample) -- every tile-position/ownership computation happens in the
network's packed [R,G1,G2,B]-plane input space, then gets scaled x2 when
applied to the (higher-resolution) output array.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from test import (
    GAMMA, PATTERN_NAMES, align_to_gbrg, undo_flip, mosaic_to_4ch,
    read_raw, find_raw_files, crop_margin,
)

NP_DTYPE = {'fp32': np.float32, 'fp16': np.float16}
ONNX_TYPE = {'fp32': 'tensor(float)', 'fp16': 'tensor(float16)'}


def tile_positions(length, tile, stride):
    """1D tile start positions covering [0, length). The last tile is shifted back
    to end exactly at `length` instead of running past the edge."""
    if length <= tile:
        return [0]
    positions = [0]
    while positions[-1] + tile < length:
        nxt = positions[-1] + stride
        if nxt + tile >= length:
            nxt = length - tile
        positions.append(nxt)
        if nxt == length - tile:
            break
    return positions


def ownership_bounds(positions, tile, length):
    """Split the full [0, length) range into one non-overlapping [start, end) slice
    per tile, cut at the midpoint of every pair of neighboring tiles' overlap."""
    bounds = [0]
    for p_prev, p_next in zip(positions, positions[1:]):
        bounds.append((p_prev + tile + p_next) // 2)
    bounds.append(length)
    return bounds


def tiled_infer_l1(sess, input_name, mosaic4_hwc, tile_h, tile_w, margin, np_dtype):
    """Run a full-size (H, W, 4) packed [R,G1,G2,B]-plane image (GBRG-aligned,
    gamma-encoded) through a fixed tile_h x tile_w x 4 -> (2*tile_h) x (2*tile_w)
    x 3 ONNX DMNestUnet-L1 graph, splitting it into overlapping tiles and
    stitching the non-overlapping "ownership" region of each tile back
    together. Ownership bounds and per-tile output offsets are computed in
    input (packed) space, then scaled by 2 wherever they're applied to the
    output array, since the L1 exit upsamples 2x."""
    H, W = mosaic4_hwc.shape[:2]

    # if the image is smaller than one tile in either dimension, reflect-pad up to
    # tile size; this content is never cropped away by a neighboring tile's margin
    # (there is none), so it does leak into the receptive field near the true edge
    pad_h, pad_w = max(0, tile_h - H), max(0, tile_w - W)
    if pad_h or pad_w:
        mosaic4_hwc = np.pad(mosaic4_hwc, [(0, pad_h), (0, pad_w), (0, 0)], mode='reflect')
    Hp, Wp = mosaic4_hwc.shape[:2]

    stride_h = max(tile_h - 2 * margin, 1)
    stride_w = max(tile_w - 2 * margin, 1)
    ys = tile_positions(Hp, tile_h, stride_h)
    xs = tile_positions(Wp, tile_w, stride_w)
    y_bounds = ownership_bounds(ys, tile_h, Hp)
    x_bounds = ownership_bounds(xs, tile_w, Wp)

    out = np.zeros((Hp * 2, Wp * 2, 3), np.float32)  # accumulate in float32 regardless of --dtype; bookkeeping only, not model compute
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            tile = mosaic4_hwc[y0:y0 + tile_h, x0:x0 + tile_w].transpose(2, 0, 1)[np.newaxis].astype(np_dtype)
            tile_out = sess.run(None, {input_name: tile})[0][0].transpose(1, 2, 0).astype(np.float32)  # (2*tile_h, 2*tile_w, 3)

            oy0, oy1 = y_bounds[i] * 2, y_bounds[i + 1] * 2
            ox0, ox1 = x_bounds[j] * 2, x_bounds[j + 1] * 2
            out[oy0:oy1, ox0:ox1] = tile_out[oy0 - y0 * 2:oy1 - y0 * 2, ox0 - x0 * 2:ox1 - x0 * 2]

    grid_info = f'{len(ys)}x{len(xs)} tiles (tile {tile_h}x{tile_w}, margin {margin}, stride {stride_h}x{stride_w})'
    return out[:H * 2, :W * 2], grid_info  # crop back off any reflect-padding (in output/2x space)


class TiledOnnxDemosaicker:
    """ONNX + tiling equivalent of test.py's per-file pipeline: same
    Bayer-phase alignment / gamma round-trip, but the network forward pass is
    replaced by tiled_infer_l1 over a fixed-shape onnxruntime session instead
    of one single-shot PyTorch forward over the whole raw image."""

    def __init__(self, sess: ort.InferenceSession, margin: int, in_bitwidth: int, out_bitwidth: int, dtype: str):
        self.sess = sess
        self.input_name = sess.get_inputs()[0].name
        self.tile_h, self.tile_w = sess.get_inputs()[0].shape[2:]
        self.margin = margin
        self.in_bitwidth = in_bitwidth
        self.out_bitwidth = out_bitwidth
        self.np_dtype = NP_DTYPE[dtype]

    def run(self, raw: np.ndarray, pattern_name: str):
        raw_aligned, flips = align_to_gbrg(raw, pattern_name)

        # reflect-pad the raw mosaic by `margin` packed-plane pixels (= 2*margin
        # raw pixels, always even so mosaic_to_4ch's color assignment stays
        # correct) so the outermost tile's true-image-edge side also gets real,
        # mirrored context instead of relying solely on the network's own
        # implicit zero-padding right at the true edge.
        pad = 2 * self.margin
        if pad:
            raw_aligned = np.pad(raw_aligned, [(pad, pad), (pad, pad)], mode='reflect')

        in_max = 2 ** self.in_bitwidth - 1
        linear = np.clip(raw_aligned / in_max, 0.0, 1.0)
        gamma_encoded = linear ** (1.0 / GAMMA)

        mosaic4_hwc = np.transpose(mosaic_to_4ch(gamma_encoded), (1, 2, 0)).astype(np.float32)  # (Hp, Wp, 4)

        pred, grid_info = tiled_infer_l1(
            self.sess, self.input_name, mosaic4_hwc,
            self.tile_h, self.tile_w, self.margin, self.np_dtype)
        # pred is still gamma-encoded (pseudo-sRGB domain), at raw-pixel resolution

        pred = crop_margin(pred, pad)  # pad (raw-pixel space) == output space here, 1:1
        pred = np.clip(pred, 0.0, 1.0)
        linear_out = pred ** GAMMA  # invGamma back to pseudo-linear domain

        out_hwc = undo_flip(linear_out, flips)  # restore original orientation
        out_max = 2 ** self.out_bitwidth - 1
        out_dtype = np.uint8 if self.out_bitwidth <= 8 else np.uint16
        quantized = np.round(np.clip(out_hwc, 0.0, 1.0) * out_max).astype(out_dtype)
        return quantized, grid_info


def main():
    parser = argparse.ArgumentParser(
        description="Tiled inference through a fixed-shape DMNestUnet-L1 ONNX export, over real (arbitrary-size) raw images")
    parser.add_argument('--onnx', type=Path, required=True, help='fixed-shape ONNX model exported by export_onnx.py')
    parser.add_argument('--input_dir', type=Path, required=True, help='root directory to search recursively for .raw files.')
    parser.add_argument('--output_dir', type=Path, required=True, help='root directory to mirror results into.')
    parser.add_argument('--height', type=int, required=True, help='raw image height (imgH), in raw Bayer pixel space.')
    parser.add_argument('--width', type=int, required=True, help='raw image width (imgW), in raw Bayer pixel space.')
    parser.add_argument('--in_bitwidth', type=int, default=12,
                        help='bit depth of the input raw samples (post BLC/Denoise/LSC/WBG).')
    parser.add_argument('--out_bitwidth', type=int, default=12,
                        help='bit depth for the output binary.')
    parser.add_argument('--bayer_pattern', type=int, required=True, choices=[0, 1, 2, 3],
                        help='0=BGGR, 1=GBRG, 2=GRBG, 3=RGGB.')
    parser.add_argument(
        '--margin', type=int, default=16,
        help='pixels (in packed [R,G1,G2,B]-plane space, i.e. half the raw Bayer resolution) '
             'discarded from each tile edge before stitching neighboring tiles together. Start '
             'around here and increase if you see tile-boundary seams in the output.',
    )
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp16'],
                        help="the ONNX graph's declared input/output tensor dtype -- must match what "
                             "export_onnx.py --dtype produced.")
    args = parser.parse_args()

    sess = ort.InferenceSession(str(args.onnx), providers=['CPUExecutionProvider'])
    tile_h, tile_w = sess.get_inputs()[0].shape[2:]
    if not isinstance(tile_h, int) or not isinstance(tile_w, int):
        raise ValueError(
            f'tile_infer.py needs a fixed-shape ONNX export (export_onnx.py bakes in '
            f'--height/--width); got dynamic shape {tile_h!r} x {tile_w!r}')
    onnx_input_type = sess.get_inputs()[0].type
    if onnx_input_type != ONNX_TYPE[args.dtype]:
        raise ValueError(
            f"--dtype {args.dtype} doesn't match the ONNX model's actual input type "
            f"{onnx_input_type} -- pass the --dtype it was exported with")

    demosaicker = TiledOnnxDemosaicker(sess, args.margin, args.in_bitwidth, args.out_bitwidth, args.dtype)
    pattern_name = PATTERN_NAMES[args.bayer_pattern]

    raw_files = find_raw_files(args.input_dir)
    if not raw_files:
        raise ValueError("No .raw files found under {}".format(args.input_dir))

    for raw_path in raw_files:
        raw = read_raw(raw_path, args.height, args.width, args.in_bitwidth)
        quantized, grid_info = demosaicker.run(raw, pattern_name)

        rel_path = os.path.relpath(raw_path, args.input_dir)
        out_bin_path = os.path.join(args.output_dir, os.path.splitext(rel_path)[0] + '.bin')
        os.makedirs(os.path.dirname(out_bin_path) or '.', exist_ok=True)
        quantized.tofile(out_bin_path)

        png_path = os.path.splitext(out_bin_path)[0] + '.png'
        import imageio
        imageio.imsave(png_path, (quantized.astype(np.float64) / (2 ** args.out_bitwidth - 1) * 255).astype(np.uint8))

        print(f'{raw_path} -> {out_bin_path}  [{grid_info}]  + {png_path}')


if __name__ == '__main__':
    main()
