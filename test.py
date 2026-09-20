#!/usr/bin/env python
"""Run the pretrained DMNestUnet model on real hardware raw mosaic dumps.

Recursively finds every `.raw` file under --input_dir and writes its result
under --output_dir, mirroring the same relative directory layout.

Expected input: flat, headerless binary files each holding a single-channel
Bayer mosaic that has already gone through BLC, raw denoise, LSC and WBG in
the hardware ISP (i.e. linear, black-level-corrected, white-balanced sensor
data -- this is the DMSC module's usual input, just not demosaicked yet).
All files must share the same --height, --width, --in_bitwidth and
--bayer_pattern.

DMNestUnet was trained on synthetically re-mosaicked sRGB (gamma-encoded)
images (see utils.rgb2RGGB / train.py), not on real linear sensor data, so
for each file this script bridges that domain gap the same way
demosaicnet_pytorch/scripts/test.py does for its own network:

  1. reflect-pads the raw image's true edge by --margin packed-plane pixels,
     so the network sees a plausible continuation of the scene there instead
     of relying on its own implicit zero-padding (PyTorch Conv2d's default
     border handling) right at the frame edge -- see --margin's help,
  2. normalizes the raw samples to [0, 1] using --in_bitwidth,
  3. Gamma-encodes them (fixed gamma = 2.2) to match the network's domain,
  4. packs the single-channel mosaic into the [R, G1, G2, B] / half-resolution
     layout DMNestUnet expects (utils.rgb2RGGB's convention),
  5. runs DMNestUnet at the requested pruning depth (--level, i.e. L1/L2/L3),
  6. crops the --margin padding back off, then inverts the Gamma to bring the
     result back to a pseudo-linear domain,
  7. quantizes to --out_bitwidth and writes an HWC binary (plus a PNG next to
     it for a quick visual check).

No CCM is applied on either side: CCM needs per-pixel RGB, which doesn't
exist yet in a Bayer mosaic, so it cannot run before the network -- and the
network doesn't produce/expect one either. The output is meant to be fed to
the hardware pipeline's own downstream CCM/Gamma/YUV stages.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import imageio

import config as config_module
from utils import imgSIZEnormalize  # noqa: F401  (kept for parity with eval.py's normalization helpers)


# Shared with tile_infer.py (imported from there) -- kept here since this is
# where the raw-domain <-> network-domain bridging logic was first written.
GAMMA = 2.2

PATTERN_NAMES = {0: "BGGR", 1: "GBRG", 2: "GRBG", 3: "RGGB"}

# utils.rgb2RGGB samples R at (row odd, col even), G1 at (row even, col even),
# G2 at (row odd, col odd), B at (row even, col odd) -- i.e. the 2x2 quad
# [[G, B], [R, G]], which is the GBRG convention. (flip_vertical, flip_horizontal)
# below turns each real-world pattern into GBRG so DMNestUnet always sees the
# layout it was trained on. The Bayer pattern is 2-periodic, so flipping an
# even-length axis swaps that axis's parity everywhere -- no crop, no padding.
# Flipping the network's output the same way undoes it exactly.
FLIP_TO_GBRG = {
    "GBRG": (False, False),
    "RGGB": (True, False),
    "BGGR": (False, True),
    "GRBG": (True, True),
}


def find_raw_files(input_dir):
    raw_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".raw"):
                raw_files.append(os.path.join(root, f))
    raw_files.sort()
    return raw_files


def read_raw(path, height, width, bitwidth):
    dtype = np.uint8 if bitwidth <= 8 else np.uint16
    raw = np.fromfile(path, dtype=dtype)
    if raw.size != height * width:
        raise ValueError(
            "Expected {}x{}={} samples, got {} from {}".format(
                height, width, height * width, raw.size, path))
    return raw.reshape(height, width).astype(np.float32)


def align_to_gbrg(raw, pattern_name):
    h, w = raw.shape
    if h % 2 or w % 2:
        # rgb2RGGB-style packing below needs even height/width, and the flip
        # trick only preserves periodicity on even axes.
        raise ValueError(
            "Expected even height/width for a 2x2 Bayer pattern, got "
            "{}x{}".format(h, w))
    flip_v, flip_h = FLIP_TO_GBRG[pattern_name]
    if flip_v:
        raw = raw[::-1, :]
    if flip_h:
        raw = raw[:, ::-1]
    return np.ascontiguousarray(raw), (flip_v, flip_h)


def undo_flip(img_hwc, flips):
    flip_v, flip_h = flips
    if flip_h:
        img_hwc = img_hwc[:, ::-1, :]
    if flip_v:
        img_hwc = img_hwc[::-1, :, :]
    return np.ascontiguousarray(img_hwc)


def mosaic_to_4ch(raw_gbrg):
    """Pack a GBRG-aligned single-channel mosaic the way utils.rgb2RGGB does,
    but reading real sampled values directly instead of subsampling a full
    RGB image -- same offsets, so the result matches the network's training
    distribution pixel-for-pixel."""
    r = raw_gbrg[1::2, 0::2]
    g1 = raw_gbrg[0::2, 0::2]
    g2 = raw_gbrg[1::2, 1::2]
    b = raw_gbrg[0::2, 1::2]
    return np.stack([r, g1, g2, b], axis=0)  # [4, H/2, W/2]


def crop_margin(arr, c):
    """arr[c:-c, c:-c] that doesn't break when c == 0 (arr[0:-0] would be empty)."""
    return arr if c == 0 else arr[c:-c, c:-c]


def _load_model(model_path, device, level):
    # Config.DEVICE is read once, at import time, by DMUnet.py to place the
    # (non-buffer, non-parameter) Gaussian-blur kernel tensor -- net.to(device)
    # below will NOT move it, since it's a plain attribute, not a registered
    # buffer. Patching Config.DEVICE before importing DMUnet is the only way
    # to make sure that kernel ends up on the right device.
    config_module.Config.DEVICE = device
    from DMUnet import DMNestUnet

    net = DMNestUnet(in_channels=config_module.Config.INP,
                      n_classes=config_module.Config.OUP)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device)
    net.eval()
    return net


def _process_one(net, device, raw_path, out_bin_path, args):
    raw = read_raw(raw_path, args.height, args.width, args.in_bitwidth)

    pattern_name = PATTERN_NAMES[args.bayer_pattern]
    raw, flips = align_to_gbrg(raw, pattern_name)

    # Reflect-pad the true image edge by `margin` packed-plane pixels (= 2*margin
    # raw pixels, always even so mosaic_to_4ch's color assignment stays correct)
    # before inference, then crop the same amount back off the output. Without
    # this, every conv layer's implicit zero-padding (PyTorch's Conv2d default)
    # treats the area just outside the frame as pure black, which measurably
    # degrades roughly the outermost receptive-field-width ring of the output
    # (empirically ~30 raw pixels for this network) -- reflecting real, mirrored
    # content there instead gives the network a plausible continuation of the
    # scene, the same fix tile_infer.py applies at each tile's outermost edge.
    pad = 2 * args.margin
    if pad:
        raw = np.pad(raw, [(pad, pad), (pad, pad)], mode='reflect')

    in_max = 2 ** args.in_bitwidth - 1
    linear = np.clip(raw / in_max, 0.0, 1.0)
    gamma_encoded = linear ** (1.0 / GAMMA)

    mosaic4 = mosaic_to_4ch(gamma_encoded)  # [4, H/2, W/2]
    inp = torch.from_numpy(mosaic4).unsqueeze(0).float().to(device)

    with torch.no_grad():
        outputs = net(inp, L=args.level)
    # forward() returns a plain tensor for L==1 but a list of per-level
    # outputs otherwise (deep supervision) -- always take the requested level.
    output = outputs[-1] if isinstance(outputs, list) else outputs
    output = output.clamp(0.0, 1.0).squeeze(0).cpu().numpy()  # [3, H, W]

    out_hwc = np.transpose(output, [1, 2, 0])  # [H, W, 3]
    out_hwc = crop_margin(out_hwc, pad)  # output is 1:1 with (padded) raw-pixel space, so pad == crop
    linear_out = out_hwc ** GAMMA  # invGamma back to pseudo-linear domain
    out_hwc = undo_flip(linear_out, flips)  # restore original orientation

    out_max = 2 ** args.out_bitwidth - 1
    out_dtype = np.uint8 if args.out_bitwidth <= 8 else np.uint16
    quantized = np.round(np.clip(out_hwc, 0.0, 1.0) * out_max).astype(out_dtype)

    os.makedirs(os.path.dirname(out_bin_path), exist_ok=True)
    quantized.tofile(out_bin_path)

    png_path = os.path.splitext(out_bin_path)[0] + ".png"
    png = np.round(np.clip(out_hwc, 0.0, 1.0) * 255.0).astype(np.uint8)
    imageio.imsave(png_path, png)

    return png_path


def main(args):
    raw_files = find_raw_files(args.input_dir)
    if not raw_files:
        raise ValueError("No .raw files found under {}".format(args.input_dir))

    device = torch.device(args.device)
    net = _load_model(args.model, device, args.level)

    out_dtype_name = "uint8" if args.out_bitwidth <= 8 else "uint16"
    for raw_path in raw_files:
        rel_path = os.path.relpath(raw_path, args.input_dir)
        out_bin_path = os.path.join(
            args.output_dir, os.path.splitext(rel_path)[0] + ".bin")

        png_path = _process_one(net, device, raw_path, out_bin_path, args)

        print("{} -> {} ({}x{}x3, {}) + {}".format(
            raw_path, out_bin_path, args.height, args.width,
            out_dtype_name, png_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=Path, required=True, help="root directory to search recursively for .raw files.")
    parser.add_argument("--output_dir", type=Path, required=True, help="root directory to mirror results into.")
    parser.add_argument("--height", type=int, required=True, help="image height (imgH).")
    parser.add_argument("--width", type=int, required=True, help="image width (imgW).")
    parser.add_argument("--in_bitwidth", type=int, default=12,
                        help="bit depth of the input raw samples (post BLC/Denoise/LSC/WBG).")
    parser.add_argument("--out_bitwidth", type=int, default=12,
                        help="bit depth for the output binary.")
    parser.add_argument("--bayer_pattern", type=int, required=True, choices=[0, 1, 2, 3],
                        help="0=BGGR, 1=GBRG, 2=GRBG, 3=RGGB.")
    parser.add_argument("--model", default=config_module.Config.TEST_PARAM_ROOT,
                        help="path to a DMNestUnet state_dict .pth file.")
    parser.add_argument("--level", type=int, default=config_module.Config.DMUnetL, choices=[1, 2, 3, 4],
                        help="pruning depth L to run inference at (see DMUnet.py forward()).")
    parser.add_argument("--margin", type=int, default=16,
                        help="reflect-pad the raw image's true edge by this many packed-plane pixels "
                             "before inference (0 disables it, reverting to the network's own implicit "
                             "zero-padding at the frame edge). Same technique and default as "
                             "tile_infer.py's --margin; ~16 was enough to fully absorb this network's "
                             "receptive field in testing.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
