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
- Run inference on real hardware raw dumps (not the synthetic sRGB-remosaic path above): `python test.py <input_dir> <output_dir> --height H --width W --bayer_pattern {0..3} [--in_bitwidth 12] [--out_bitwidth 8] [--model pretrained_model/params_e123.pth] [--level 1-4] [--device cpu|cuda]`. See "Real raw-sensor inference" below.

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

**DMNestUnet (`DMUnet.py`)**: a UNet++ nested/dense-skip topology with depth up to 5 encoder stages (`x_00` … `x_40`). Each stage's "downsampling" is a fixed depthwise Gaussian blur (`utils.get_GaussKernel_2`, applied via `nn.functional.conv2d(..., stride=1, groups=4)`) — **stride is 1, so this is pure blurring, not spatial downsampling**: every stage from `x_00` to `x_40` operates at the *same* internal spatial resolution, only the channel width grows (8→16→32→64→128). This is the key cost driver: because there's no resolution pyramid to offset the growing channel count, compute roughly **quadruples with each additional nested level** rather than the ~4x-cheaper scaling you'd get with real downsampling. Measured (via `thop`) for a simulated 1920×1080 input (internal 960×540 after Bayer packing) at 30 fps: L1 ≈ 0.37 TOPS, L2 ≈ 1.44 TOPS, L3 ≈ 5.64 TOPS, L4 ≈ 21.5 TOPS (1 MAC = 2 ops convention). Each `x_i0` encoder stage is an `RRDBNet_SENet` block (`models.py`); each nested skip-connection node (`x_ij`, j>0) is a `Reconstruction` block that concatenates same-row and upsampled deeper-row features. `forward(x, L)` takes a pruning factor `L` (1–4, default `config.DMUnetL`) that controls how many nested output stages actually run — this is the mechanism for trading accuracy against compute/latency for edge deployment, and because `forward` computes levels sequentially with early `return`, running at level `L` always also computes (but discards, when not under deep supervision) every level `< L`. When `deep_supervision` is on (default), `forward` returns a **list** of outputs for `L >= 2`, but a **plain tensor** for `L == 1` — callers must handle both shapes (see `test.py`'s `outputs[-1] if isinstance(outputs, list) else outputs`).

**RRDBNet_SENet (`models.py`)**: Residual-in-Residual Dense Block stack (3× `ResidualDenseBlock_3C_dp`, each built from depthwise-separable convs, `Convdw`) followed by a squeeze-and-excite gate (`RRDBblock_SENet`). This is the core learned feature extractor at every encoder depth.

**Loss (`config.py` + `pytorch_ssim/`)**: training uses a two-phase criterion switch — `CRITERION1` (windowed SSIM, self-contained implementation in `pytorch_ssim/`) for the first `CHANGE_CRI_EPOCH` epochs, then `CRITERION2` (MSE) afterward. With deep supervision, the loss is the mean of the criterion applied to every returned output level, averaged over `DMUnetL`.

**Config (`config.py`)**: single source of truth for device, model depth/channels, dataset selection, optimizer/LR settings, and all input/output paths. Nearly every other module imports `from config import Config` and reads class attributes directly (no instantiation-time overrides) — the pattern in this codebase is to edit `config.py` in place rather than pass parameters through.

## Real raw-sensor inference (`test.py`)

`train.py`/`eval.py` only ever exercise the synthetic path: clean sRGB ground truth → `rgb2RGGB` → network. `test.py` is a separate entry point for running the pretrained network on **actual hardware raw dumps** — single-channel, headerless `.raw` binaries that have already been through a traditional ISP's BLC, raw denoise, LSC and WBG stages (i.e. linear, black-level-corrected, white-balanced sensor data, sitting exactly where the DMSC module would normally go), one directory tree in, mirrored `.bin`+`.png` results out. It bridges two domain gaps that don't exist in the synthetic path:

- **Bayer phase**: input can be any of BGGR/GBRG/GRBG/RGGB (`--bayer_pattern`); the script flips the raw array (even-sized axes only — flipping is exact because the pattern is 2-periodic) to align it to the network's native GBRG phase before packing it into `[R,G1,G2,B]`, then flips the output back. Modeled on the equivalent (GRBG-targeted) alignment step in `../demosaicnet_pytorch/scripts/test.py`, which faces the same single-fixed-pattern constraint on its own network.
- **Gamma domain**: the network was trained on gamma-encoded (sRGB) images, but real raw sensor data is linear. The script normalizes by `--in_bitwidth`, gamma-encodes with a fixed gamma of 2.2 before inference, then inverse-gammas the output back to a pseudo-linear domain before quantizing to `--out_bitwidth` — no CCM is applied on either side (a Bayer mosaic has no per-pixel RGB for CCM to operate on); the output is meant to feed the hardware pipeline's own downstream CCM/Gamma/YUV stages.

One implementation detail worth knowing if you touch device handling: `DMUnet.py`'s Gaussian-blur kernel (`self.kernel`) is a plain tensor attribute, not a registered buffer/parameter, so `net.to(device)` won't move it — `test.py` works around this by patching `config.Config.DEVICE` *before* importing `DMUnet`, so the kernel is built on the right device from the start.

## Known rough edges (be aware of when editing nearby code)

- `train.py` reads `config.CHANGE_CRI_EPOCH[0]`, but `Config.CHANGE_CRI_EPOCH` is defined as a plain int (`10`) — indexing it will fail. `learning_rate_schedulers.StepDecay` is also called in `train.py` as `StepDecay(optimizer, epoch)` (i.e. as if it were the `__call__` function), but `StepDecay` is a class whose `__init__` takes `(initAlpha, factor, dropEvery)` — it is never instantiated before being invoked. Both call sites need reconciling with their definitions before a training run will actually work end-to-end.
- `eval.py`'s `test_for_datasets` helper references `CONFIG.UnetL`, which doesn't exist on `Config` (the attribute is `DMUnetL`) — that function path is currently broken; the `__main__` entry point uses `test_datasets`/`test_for_time` instead, which are unaffected.
