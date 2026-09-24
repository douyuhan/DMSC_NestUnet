# DMNestUnet → ISP Cmodel Integration Notes

Reference info to carry into the hardware ISP Cmodel project session when wiring up the exported DMNestUnet ONNX graph as a replacement for the pipeline's existing traditional DMSC (demosaic) module. Values/line references are current as of this repo's `export_onnx.py`/`tile_infer.py`/`test.py`.

**`tile_infer.py` is the reference implementation to port** — it's the only place in this repo that already does "arbitrary-size real raw image → tiles → per-tile ONNX inference → stitched full image", which is exactly the shape of the problem in the Cmodel. Its tiling/stitching math (`tile_positions`/`ownership_bounds`, adapted for a 2x-upsampling network — see §6), the Bayer-coherent reflect-padding scheme (§5), and the gamma domain bridging (§3) are all things a C implementation needs to reproduce bit-for-bit (or close to it) to match this Python pipeline's output.

This repo's own AI-demosaic port follows the same overall shape as a prior AI-denoise Cmodel integration (`../AIDenoise_research/PMRID/AI_CMODEL_INTEGRATION.md`), but the two networks differ in ways that change several of these sections in substance, not just detail — most importantly: **this network's output is 3-channel RGB at 2x the input's resolution, not a same-resolution residual added back to the input.** Don't assume anything from that other doc carries over without checking against §2 below.

## 1. Exported ONNX artifact(s)

Whichever `.onnx` file you copy over, you need to know these things about it (not otherwise recoverable from the file alone without checking):

| Property | Where it's decided | How to check on the file |
|---|---|---|
| `--height`/`--width` (tile size, packed-plane space) | `export_onnx.py --height/--width` (default 256x256) | ONNX model's `input[0].shape[2:]` |
| `--dtype` (`fp32`/`fp16`) | `export_onnx.py --dtype` | `input[0].type` — `tensor(float)` vs `tensor(float16)` |
| Pruning level exported | **Always L1** — `export_onnx.py` only ever wraps `forward(x, L=1)` (`Level1Module`); there is no flag to export L2/L3/L4 | n/a — if you need a deeper level, `export_onnx.py` needs a small edit (change the hardcoded `L=1`) *and* you must handle that `forward` returns a **list** of per-level outputs for L>=2 under deep supervision, not a single tensor — the exported graph would then need to pick one output index before/at export, since ONNX graphs don't return Python lists |

Regenerate exports locally with `python export_onnx.py <checkpoint.pth> --height H --width W --dtype fp32|fp16` (see this repo's `ReadMe.md`). There is no baked-normalization variant to track (unlike some other AI-ISP exports in sibling repos) — gamma bridging is always external here, see §3.

## 2. Network I/O contract

- **Input** tensor `input`: shape `(1, 4, tile_h, tile_w)`, dtype fp32 or fp16. This is **not** raw Bayer and **not** 3-channel RGB — it's 4-channel packed `[R, G1, G2, B]` planes, one plane per 2x2 Bayer block position (see §4 for the exact packing/phase). Values are in **gamma-encoded `[0,1]`** domain (see §3) — do not feed linear sensor data directly.
- **Output** tensor `output`: shape `(1, 3, 2*tile_h, 2*tile_w)` — **note the 2x**. `DMUnet.py`'s final layer (`final_upSample`, a `ConvTranspose2d`) undoes the 4-channel packing back to full Bayer-image resolution, producing 3-channel RGB directly. This is fundamentally different from a same-resolution denoiser: **there is no "add the output back to the input" step** — the network's output tensor *is* the final demosaicked image (still in gamma-encoded domain, needs the inverse-gamma step in §3 applied before use).
- Batch is fixed at 1; H/W are static (baked in, no dynamic axes) — the graph only ever accepts exactly the tile size it was exported with. This is why tiling (§6) is needed for any real (larger) image.
- Tile size is in **packed-plane space** = half the resolution of the original raw Bayer image in each dimension. A 256x256 ONNX tile corresponds to a 512x512 region of the raw Bayer image as input, and produces a 512x512 RGB region as output.

## 3. Gamma domain bridging (always external — no baked variant exists for this network)

The network was trained on synthetically re-mosaicked **sRGB** (gamma-encoded) images (`utils.rgb2RGGB`/`train.py`), never on real linear sensor data. Real raw sensor data (post BLC/raw-denoise/LSC/WBG, i.e. exactly where the DMSC module sits in the pipeline) is linear. The Cmodel must apply this fixed transform itself, before feeding a tile in and after reading a tile out — there is no ISO/scene-dependent parameter here (unlike some other AI-ISP modules), just a fixed gamma of 2.2:

```
GAMMA = 2.2

# forward (before feeding the network), per raw sample:
linear    = clip(raw_sample / (2^in_bitwidth - 1), 0, 1)
network_input = linear ** (1.0 / GAMMA)

# inverse (after reading the network's output), per output sample:
network_output = clip(network_output, 0, 1)
pseudo_linear  = network_output ** GAMMA
out_sample     = round_half_away_from_zero(pseudo_linear * (2^out_bitwidth - 1))   # see §7
```

No CCM is applied on either side of the network: CCM needs per-pixel RGB, which doesn't exist yet in a Bayer mosaic (so it can't run before), and the network's output is meant to feed the hardware pipeline's own downstream CCM/Gamma/YUV stages (so it doesn't run after, either). Reference: `test.py`'s `_process_one`, `GAMMA` constant.

## 4. Bayer pattern → canonical GBRG, and packing order

**The network only ever sees the canonical GBRG phase — not RGGB.** This is not stated anywhere in the original paper/training code; it falls out of reading `utils.rgb2RGGB`'s fixed sampling offsets (see CLAUDE.md's Architecture section for the full derivation), and was confirmed empirically with the real pretrained checkpoint: feeding an uncorrected RGGB mosaic (vs. the same scene correctly phase-aligned) measured **30.15 dB vs. 11.59 dB PSNR** on a test image — i.e. getting this wrong is not a minor quality hit, it's catastrophic.

Real sensor data can be in any of the 4 Bayer orders; convert before inference and convert back after (this conversion is its own inverse — same operation both ways, exact because the pattern is 2-periodic and the flip axes are even-length):

| `bayer_pattern` | flip rows (vertical)? | flip cols (horizontal)? |
|---|---|---|
| `GBRG` (native — no-op) | no | no |
| `RGGB` | yes | no |
| `BGGR` | no | yes |
| `GRBG` | yes | yes |

Reference: `test.py`'s `FLIP_TO_GBRG` / `align_to_gbrg`. **Padding by an even number of pixels only** (§5) — this table's flips and the padding scheme both rely on preserving the 2x2 phase, which only holds for even-length axes/offsets.

**Packing** (`test.py`'s `mosaic_to_4ch`): given a canonical-GBRG-aligned 2D Bayer array of shape `(H, W)`, the 2x2 quad is `[[G1, B], [R, G2]]`. Each channel is sampled at a fixed `(row%2, col%2)` offset within that quad:

| Channel | Color | `(row%2, col%2)` |
|---|---|---|
| 0 | R  | `(1, 0)` |
| 1 | G1 | `(0, 0)` |
| 2 | G2 | `(1, 1)` |
| 3 | B  | `(0, 1)` |

Result shape `(4, H/2, W/2)` (channels-first), batched to `(1, 4, H/2, W/2)` for the network. There is no "rggb2bayer"-style unpacking needed on the output side — the output is already full-resolution 3-channel RGB, not a packed plane representation (see §2).

## 5. Padding for real (non-tile-sized) images

Every conv in this network uses PyTorch's default `padding_mode='zeros'`, so without help, the outermost ring of the output near the *true* image border is computed against fake black padding instead of real scene content. The fix (verified in this repo — see §8 for the exact numbers): **reflect-pad the raw Bayer mosaic itself** (before packing into the 4-channel planes), not the packed planes after, and always by an **even** number of raw pixels — both needed to keep the Bayer color at each mosaic position consistent across the seam, same reasoning as the flip table in §4.

Use numpy's default `mode='reflect'` (does **not** duplicate the edge pixel) — not an edge-duplicating reflect (e.g. `cv2.BORDER_REFLECT`, numpy's `'symmetric'`), which flips the color phase at the seam. If the Cmodel's padding primitive duplicates the edge pixel by default (many do), use its no-duplicate variant instead (e.g. `cv2.BORDER_REFLECT_101`), or implement the mirror-index math directly.

**Default margin = 16** (packed-plane pixels, i.e. `2*margin = 32` raw pixels padded on each side). This repo empirically measured this network's effective receptive field at **~30 raw pixels**: comparing a reflect-padded-then-cropped single-shot inference against an unpadded reference, the diff is large near the true image edge but drops to float32-rounding-only (max diff of 1 count in a 16-bit output) once a 32-raw-pixel border is excluded from the comparison — `margin=16` is comfortably past that point. Both `test.py --margin` (single-shot, whole image) and `tile_infer.py --margin` (tiled) use this same value and scheme; `--margin 0` reverts to the old (implicit zero-padding) behavior in both.

## 6. Tiling & stitching — the part that differs most from a same-resolution network

Reference: `tile_infer.py`'s `tile_positions`, `ownership_bounds`, `tiled_infer_l1`.

- **`tile_positions(length, tile, stride)`**: 1D tile start positions covering `[0, length)`. Starts at 0, advances by `stride` each time, except the last tile is shifted back to end exactly at `length` (no overshoot past the true edge).
- **`stride = tile - 2*margin`** (clamped to `>= 1`).
- **`ownership_bounds(positions, tile, length)`**: splits `[0, length)` into one non-overlapping slice per tile — each pair of neighboring tiles' overlap region is cut at its midpoint. Every output pixel is written by exactly one tile (whichever "owns" that region), not blended/averaged.
- Applied independently in H and W (in **packed-plane, i.e. network-input, space**) to get a 2D grid of tiles; each `(tile_h, tile_w, 4)` tile is run through the graph.

**The one thing that's genuinely different from a same-resolution network (e.g. a denoiser)**: since this network's output is 2x the input tile's resolution (§2), every ownership bound and per-tile offset computed above (in input/packed space) must be **scaled by 2** before being applied to the (higher-resolution) output array:

```
# y0, x0: this tile's start position, in packed-plane (input) space
# y_bounds[i], y_bounds[i+1]: this tile's ownership slice, in packed-plane space
tile_out = network(tile)                      # (2*tile_h, 2*tile_w, 3)

oy0, oy1 = y_bounds[i] * 2, y_bounds[i + 1] * 2      # ownership bounds, scaled into output space
ox0, ox1 = x_bounds[j] * 2, x_bounds[j + 1] * 2

output[oy0:oy1, ox0:ox1] = tile_out[oy0 - y0*2 : oy1 - y0*2, ox0 - x0*2 : ox1 - x0*2]
```

Get this ×2 wrong (e.g. porting a same-resolution-network's stitching code unchanged) and every tile boundary will visibly misalign in the output — this is the single easiest mistake to make when adapting a same-resolution reference tiling implementation to this network.

After the full padded image is stitched, crop the `2*margin` raw-pixel padding back off the **output** — note output space is 1:1 with (padded) raw-pixel space (not packed-plane space), so the crop amount is the same raw-pixel value used for the input pad, not doubled again.

**Verified in this repo**: the stitching math above reproduces a single-shot forward pass over the same (padded) input to ~2.5e-6 max abs diff on `[0,1]` scale (float32 rounding only), confirmed independent of tile grid size/margin — the stitching logic itself is not a source of error. See §8 for a specific false lead to avoid re-chasing if a similar-looking diff shows up during the Cmodel port.

## 7. Output rounding

Not yet independently confirmed as a live bug in this repo's own scripts (unlike the sibling AI-denoise integration, which did hit and fix this), but the same risk applies here since `test.py`/`tile_infer.py` both use plain `np.round` for final quantization to `--out_bitwidth`, which is round-half-to-even ("banker's rounding"), not round-half-away-from-zero ("真正的四舍五入"). **Recommend the Cmodel use round-half-away-from-zero explicitly** (`sign(x) * floor(abs(x) + 0.5)`) rather than whatever a given C rounding intrinsic defaults to, since pixel values can land on exact `.5` boundaries often enough (e.g. clean fractions of `1/255` or `1/65535`) for the rounding mode to measurably bias output — this was a real, previously-fixed bug in the sibling AI-denoise Cmodel port for exactly this reason.

## 8. Known findings / non-issues (don't re-diagnose these if they show up again)

- **A large diff between a naive tiled implementation and a single-shot reference is NOT caused by the SE (squeeze-and-excite) gate's global average pooling.** This was the first (wrong) hypothesis chased while building `tile_infer.py` in this repo, since `RRDBblock_SENet`'s `F.avg_pool2d(out3, (H, W))` computes a genuinely global statistic per forward call, and it's a reasonable thing to suspect first. It was disproved by a controlled experiment: a pure sub-crop (smaller true image, no padding at all) reproduced deep-interior pixels to ~1e-7 — if the SE gate's global stats mattered, changing the total image size alone should have shown a large diff even far from any border, and it didn't. The actual cause of the large diff observed at the time was simply that one reference script (`test.py`, before it gained `--margin`) wasn't reflect-padding the true image edge at all (relying on implicit zero-padding, §5) while the other (`tile_infer.py`) was — i.e. an expected, bounded, edge-only effect (see §5/§6 for the actual numbers), not a stitching bug and not a global-statistics issue.
- **The pretrained checkpoint's L4-only layers (`x_40`, `x_31`, `x_22`, `x_13`, `x_04`, `up_40_to_31`, etc.) were never trained.** Training used `config.DMUnetL=3`, and `forward`'s early `return` at `L=3` means those layers never received a gradient update — they exist in the state dict (random init) but produce meaningless output. Only L1/L2/L3 are usable from this checkpoint. (§1 also only supports exporting L1 as-is; exporting L2/L3 needs a small `export_onnx.py` edit, see §1's table.)
- **fp16 export has a real, measured precision cost**: fp32 ONNX export matches the PyTorch model to ~1.3e-5 max abs diff; fp16 export (via `onnxconverter_common`'s post-export graph cast) measured ~2.4e-2 max abs diff against the same fp32 PyTorch reference. Budget for this if choosing fp16 for the Cmodel deployment target — it is not a negligible rounding-only difference the way the fp32 export is.
- Compute cost reference (from this repo's `ReadMe.md`, measured via `thop` against the real checkpoint, 1920x1080 @ 30fps, 1 MAC = 2 ops): **L1 ≈ 0.37 TOPS**, L2 ≈ 1.44 TOPS, L3 ≈ 5.64 TOPS. `export_onnx.py` only exports L1 today specifically because it's the realistic edge-deployment target — confirm this still matches the actual Cmodel target accelerator's budget before assuming L1 is the right choice to integrate.

## 9. Code pointers in this repo (for exact reference while porting)

- Gamma math: `test.py`'s `GAMMA` constant, `_process_one` (forward gamma-encode, inverse gamma after inference)
- Bayer pattern flips / packing: `test.py`'s `FLIP_TO_GBRG`, `align_to_gbrg`, `mosaic_to_4ch`, `undo_flip`
- Reflect-padding scheme: `test.py`'s `_process_one` (single-shot, whole-image) and `tile_infer.py`'s `TiledOnnxDemosaicker.run` (tiled) — both call the shared `crop_margin` helper (defined in `test.py`, imported by `tile_infer.py`)
- Tiling/stitching (incl. the ×2 output-scaling adaptation): `tile_infer.py`'s `tile_positions`, `ownership_bounds`, `tiled_infer_l1`
- Network architecture / pruning-level semantics: `DMUnet.py`'s `DMNestUnet.forward`, this repo's `CLAUDE.md` (Architecture section)
- Export options/defaults: `export_onnx.py` (`parse_args` for all flags, `Level1Module` for the L=1 wrapping)
