# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research code accompanying the paper "A Compact High-Quality Image Demosaicking Neural Network for Edge Computing Devices" (MDPI). It implements DMNestUnet ("DMUnet++"), a UNet++-style nested network with RRDB+SE blocks that reconstructs a full RGB image from a simulated Bayer-pattern mosaic input.

## Commands

There is no build system, package manifest, or test suite — this is a small, flat set of Python scripts run directly.

- Train: `python train.py` — saves checkpoints/loss under `./TrainingResult` (`params/` and `loss/` subfolders, created automatically).
- Evaluate: `python eval.py` — runs `test_datasets(...)` by default, writing per-level output images and a `*_cpsnr_crop_{crop}_ignore_{ignore}.xlsx` PSNR report under `./TestingResult/<Test_DataSet>/`. Swap in `test_for_time(...)` in the `__main__` block to benchmark inference latency instead.
- All tunables (dataset selection, paths, hyperparameters, device, crop/ignore defaults) live in `config.py` and must be edited there — there is no CLI argument parsing.
- Inspect a model's architecture/parameter count: run `DMUnet.py` or `models.py` directly (each has a `__main__` that builds a network and calls `torchsummary.summary` — requires a CUDA device as written; `torchsummary` is only imported inside those `__main__` blocks, so it isn't a hard dependency for importing the modules elsewhere).
- Run inference on real hardware raw dumps (not the synthetic sRGB-remosaic path above): `python test.py <input_dir> <output_dir> --height H --width W --bayer_pattern {0..3} [--in_bitwidth 12] [--out_bitwidth 8] [--model pretrained_model/params_e123.pth] [--level 1-4] [--margin 16] [--device cpu|cuda]`. See "Real raw-sensor inference" below.
- Export the L1 exit to a fixed-shape ONNX graph: `python export_onnx.py <checkpoint.pth> [--height 256] [--width 256] [--dtype fp32|fp16] [--opset 13] [--output path.onnx]`.
- Run that ONNX graph, tiled, over an arbitrary-size real raw dump: `python tile_infer.py --onnx <exported.onnx> <input_dir> <output_dir> --height H --width W --bayer_pattern {0..3} [--margin 16] [--dtype fp32|fp16] [--in_bitwidth 12] [--out_bitwidth 8]`. See "ONNX export and tiled inference" below.

## Data layout

Training data is not provided; evaluation demo sets (Kodak24, McMaster) are under `DataSets/Evaluating/`. Both loaders use `torchvision.datasets.ImageFolder`, so every dataset directory needs a nested class subfolder even for evaluation-only use:

```
DataSets/
├── Training/<YOUR_SET>/train/*.jpg
└── Evaluating/<YOUR_SET>/test/*.{png,tif,...}
```

`config.Train_DataSet` / `config.Test_DataSet` select an entry from `config.DataSets_Name` and are joined with `TRAIN_DATA_ROOT` / `TEST_DATA_ROOT` to build the actual path passed to the loaders.

## Architecture

**Bayer simulation (`utils.rgb2RGGB`)**: turns a full RGB tensor into a 4-channel `[R, G1, G2, B]` mosaic by subsampling each plane on a *fixed* offset — this is the network's input, not real sensor data. The four offsets (`R` at `(row%2,col%2)=(1,0)`, `G1` at `(0,0)`, `G2` at `(1,1)`, `B` at `(0,1)`) form the 2×2 quad `[[G,B],[R,G]]`, i.e. **the network is trained and hard-wired to exactly one Bayer phase: GBRG** — not RGGB/BGGR/GRBG, and not pattern-invariant. This isn't stated anywhere in the code/paper; it only falls out of reading `rgb2RGGB`'s slice indices. Two things reinforce it: the first conv in every `RRDBNet_SENet` stage is a plain (non-permutation-invariant) `Conv2d` over the 4 input channels, and `load_data.py` has `RandomHorizontalFlip`/`RandomRotation` commented out, so training never showed the network any other phase. Feeding the wrong phase without correcting for it is not a minor accuracy hit — on a McMaster test image, PSNR measured 30.15 dB with the correct phase vs. 11.59 dB feeding an uncorrected RGGB mosaic (verified with the pretrained checkpoint). `test.py` (see below) handles this via pattern-to-GBRG alignment before inference.

**DMNestUnet (`DMUnet.py`)**: a UNet++ nested/dense-skip topology with depth up to 5 encoder stages (`x_00` … `x_40`). Each stage's "downsampling" is a fixed depthwise Gaussian blur (`utils.get_GaussKernel_2`, applied via `nn.functional.conv2d(..., stride=1, groups=4)`) — **stride is 1, so this is pure blurring, not spatial downsampling**: every stage from `x_00` to `x_40` operates at the *same* internal spatial resolution, only the channel width grows (8→16→32→64→128). This is the key cost driver: because there's no resolution pyramid to offset the growing channel count, compute roughly **quadruples with each additional nested level** rather than the ~4x-cheaper scaling you'd get with real downsampling. Measured (via `thop`) for a simulated 1920×1080 input (internal 960×540 after Bayer packing) at 30 fps: L1 ≈ 0.37 TOPS, L2 ≈ 1.44 TOPS, L3 ≈ 5.64 TOPS, L4 ≈ 21.5 TOPS (1 MAC = 2 ops convention). Each `x_i0` encoder stage is an `RRDBNet_SENet` block (`models.py`); each nested skip-connection node (`x_ij`, j>0) is a `Reconstruction` block that concatenates same-row and upsampled deeper-row features. `forward(x, L)` takes a pruning factor `L` (1–4, default `config.DMUnetL`) that controls how many nested output stages actually run — this is the mechanism for trading accuracy against compute/latency for edge deployment, and because `forward` computes levels sequentially with early `return`, running at level `L` always also computes (but discards, when not under deep supervision) every level `< L`. When `deep_supervision` is on (default), `forward` returns a **list** of outputs for `L >= 2`, but a **plain tensor** for `L == 1` — callers must handle both shapes (see `test.py`'s `outputs[-1] if isinstance(outputs, list) else outputs`). `self.final_upSample` (the `ConvTranspose2d` that undoes the 4-channel packing back to 3-channel RGB at 2x the packed input's height/width) is a single shared instance called once per level (`self.final_upSample(x_0k_output)` for every `k`) — **every level's output is 2x the packed input's resolution, not just L1's.**

**RRDBNet_SENet (`models.py`)**: Residual-in-Residual Dense Block stack (3× `ResidualDenseBlock_3C_dp`, each built from depthwise-separable convs, `Convdw`) followed by a squeeze-and-excite gate (`RRDBblock_SENet`). This is the core learned feature extractor at every encoder depth.

**Loss (`config.py` + `pytorch_ssim/`)**: training uses a two-phase criterion switch — `CRITERION1` (windowed SSIM, self-contained implementation in `pytorch_ssim/`) for the first `CHANGE_CRI_EPOCH` epochs, then `CRITERION2` (MSE) afterward. With deep supervision, the loss is the mean of the criterion applied to every returned output level, averaged over `DMUnetL`.

**Config (`config.py`)**: single source of truth for device, model depth/channels, dataset selection, optimizer/LR settings, and all input/output paths. Nearly every other module imports `from config import Config` and reads class attributes directly (no instantiation-time overrides) — the pattern in this codebase is to edit `config.py` in place rather than pass parameters through.

## Real raw-sensor inference (`test.py`)

`train.py`/`eval.py` only ever exercise the synthetic path: clean sRGB ground truth → `rgb2RGGB` → network. `test.py` is a separate entry point for running the pretrained network on **actual hardware raw dumps** — single-channel, headerless `.raw` binaries that have already been through a traditional ISP's BLC, raw denoise, LSC and WBG stages (i.e. linear, black-level-corrected, white-balanced sensor data, sitting exactly where the DMSC module would normally go), one directory tree in, mirrored `.bin`+`.png` results out. It bridges two domain gaps that don't exist in the synthetic path:

- **Bayer phase**: input can be any of BGGR/GBRG/GRBG/RGGB (`--bayer_pattern`); the script flips the raw array (even-sized axes only — flipping is exact because the pattern is 2-periodic) to align it to the network's native GBRG phase before packing it into `[R,G1,G2,B]`, then flips the output back. Modeled on the equivalent (GRBG-targeted) alignment step in `../demosaicnet_pytorch/scripts/test.py`, which faces the same single-fixed-pattern constraint on its own network.
- **Gamma domain**: the network was trained on gamma-encoded (sRGB) images, but real raw sensor data is linear. The script normalizes by `--in_bitwidth`, gamma-encodes with a fixed gamma of 2.2 before inference, then inverse-gammas the output back to a pseudo-linear domain before quantizing to `--out_bitwidth` — no CCM is applied on either side (a Bayer mosaic has no per-pixel RGB for CCM to operate on); the output is meant to feed the hardware pipeline's own downstream CCM/Gamma/YUV stages.
- **Frame-edge padding (`--margin`, default 16)**: every conv in this network uses PyTorch's default `padding_mode='zeros'`, so without any help, the outermost ring of the output is computed from implicit black padding at the true image edge, not real scene content. `test.py` reflect-pads the raw image's true edge by `2*margin` raw pixels (always even, to keep the 2x2 CFA phase intact) before inference and crops the same amount back off after — measured empirically (see "ONNX export and tiled inference" below) to fully absorb this network's receptive field at `margin=16`. `--margin 0` reproduces the old (pre-padding) behavior exactly.

One implementation detail worth knowing if you touch device handling: `DMUnet.py`'s Gaussian-blur kernel (`self.kernel`) is a plain tensor attribute, not a registered buffer/parameter, so `net.to(device)` won't move it — `test.py` works around this by patching `config.Config.DEVICE` *before* importing `DMUnet`, so the kernel is built on the right device from the start.

## ONNX export and tiled inference (`export_onnx.py`, `tile_infer.py`)

`export_onnx.py` exports **only the L1 exit** (the cheapest sub-network, see above) of a checkpoint to a fixed-shape ONNX graph — `forward(x, L=1)` always returns a plain tensor, so wrapping it in a one-line `Level1Module` gives a graph with exactly one static output. `--height`/`--width` (default 256, no divisibility constraint — unlike a spatial-pyramid UNet, this network never downsamples) are in packed `[R,G1,G2,B]`-plane space and get baked into the graph; `--dtype fp16` does a post-export graph-level cast (`onnxconverter_common`), same caveat as any CPU-side fp16 export: never actually traced/run in fp16, just cast afterward.

`tile_infer.py` runs that fixed-shape graph, tiled, over an arbitrary-size real raw dump — same Bayer-phase alignment and gamma round-trip as `test.py` (the helper functions `GAMMA`, `PATTERN_NAMES`, `align_to_gbrg`, `undo_flip`, `mosaic_to_4ch`, `read_raw`, `find_raw_files`, `crop_margin` live in `test.py` and are imported from there — single source of truth, don't duplicate them). Its tiling scheme is adapted from `../../AIDenoise_research/PMRID/tile_infer.py`'s `tile_positions`/`ownership_bounds` overlap-and-stitch pattern, with one structural change: PMRID's network outputs at the same resolution as its packed input, but DMNestUnet's L1 exit **upsamples 2x** (see above), so every ownership-bound/tile-offset computed in input (packed) space gets scaled ×2 before being applied to the (higher-resolution) output array — see `tiled_infer_l1`.

**Verified findings from building this** (worth knowing before changing `--margin` or debugging a "tiled output looks off" report):
- The tile-to-tile stitching math itself is correct: a tiled run reproduces a single-shot forward pass over the same (padded) input to ~2.5e-6 (float32 rounding only), confirmed by direct comparison, independent of tile grid size.
- This network's effective receptive field is empirically **~30 raw pixels** (~15 packed pixels): comparing padded-and-cropped output against an unpadded reference, the diff is large near the true image edge but drops to float32-rounding-only (max diff of 1 count in 16-bit output) once you exclude a 32-raw-pixel border — this is why `margin=16` (packed-space) is enough as a default for both scripts.
- A naive first attempt to explain a large tiled-vs-`test.py` diff as "the SE gate's global average pool makes output depend on total image size" was **disproved** by a controlled experiment (pure crop, no padding, deep-interior pixels matched to ~1e-7) — the real cause was `test.py` not padding the true image edge at all (relying on implicit zero-padding, above) while `tile_infer.py` did. Once `test.py` gained the same `--margin` reflect-padding, the two scripts agree almost exactly everywhere, edges included. If a future non-obvious inference-quality diff shows up again, don't assume it's the SE gate without a controlled test like this one — it wasn't the cause here.

## Known rough edges (be aware of when editing nearby code)

- `train.py` reads `config.CHANGE_CRI_EPOCH[0]`, but `Config.CHANGE_CRI_EPOCH` is defined as a plain int (`10`) — indexing it will fail. `learning_rate_schedulers.StepDecay` is also called in `train.py` as `StepDecay(optimizer, epoch)` (i.e. as if it were the `__call__` function), but `StepDecay` is a class whose `__init__` takes `(initAlpha, factor, dropEvery)` — it is never instantiated before being invoked. Both call sites need reconciling with their definitions before a training run will actually work end-to-end.
- `eval.py`'s `test_for_datasets` helper references `CONFIG.UnetL`, which doesn't exist on `Config` (the attribute is `DMUnetL`) — that function path is currently broken; the `__main__` entry point uses `test_datasets`/`test_for_time` instead, which are unaffected.
