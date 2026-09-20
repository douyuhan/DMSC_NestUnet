### A short demo code for research on Bayer pattern image demosaicking 
### (paper "A Compact High-Quality Image Demosaicking Neural Net-work for Edge Computing Devices" being previewed by MDPI)


- Usage:
    - Run `python train.py` to train a new model. The default saving file is `./TrainingResult`
    - Run `python eval.py` to test running time and psnr value of chosen dataset. The PSNR value of each level will saved in ` ./TestingResult/YOUR TESTING SETS NAME/`
    - For testing, you may change the `crop` and `ignore` value to set the padding value and the ignored pixel number of each edge when calculating PSNR value (Default is both 0).
    - Settings should be changed in config.py
    - Run `python test.py` to run inference on **real hardware raw dumps** instead of the synthetic sRGB-remosaic path above (see below).
    - Run `python export_onnx.py` to export the L1 exit to ONNX, and `python tile_infer.py` to run that ONNX graph (tiled) over arbitrary-size real raw dumps (see below).

- Note on the Bayer pattern the network expects:
    - The network is trained/tested only against the **GBRG** phase (`utils.rgb2RGGB`'s fixed sampling offsets bake this in); it is not pattern-invariant. Feeding it a mismatched phase without correcting for it degrades PSNR drastically (measured ~18 dB drop on a sample image).
    - `test.py` handles arbitrary input Bayer patterns for you via a `--bayer_pattern` flag (it flips the raw array to align it to GBRG before inference, and flips the result back).

- `test.py`: real raw-sensor inference
    - Unlike `train.py`/`eval.py` (which always synthesize their mosaic input from a clean sRGB ground-truth image via `rgb2RGGB`), `test.py` is meant for actual hardware raw dumps: flat, headerless, single-channel binary files that have already gone through a traditional ISP's BLC, raw denoise, LSC and WBG stages — i.e. exactly the input a DMSC (demosaicking) module would normally receive.
    - Because the network was trained on gamma-encoded (sRGB) images rather than linear sensor data, `test.py` gamma-encodes the normalized raw input with a fixed gamma of 2.2 before inference, and inverse-gammas the network's output back to a pseudo-linear domain afterward, so the result can be handed to the rest of a traditional ISP pipeline (CCM, gamma, YUV, ...).
    - `--margin` (default 16): every conv in the network implicitly zero-pads at whatever its input's edge happens to be, so without help, the outermost ring of the output near the *true* image border is computed against fake black padding instead of real scene content. `test.py` reflect-pads the raw image's true edge by `2*margin` raw pixels before inference and crops it back off afterward to fix this (`--margin 0` disables it, reverting to the old behavior).
    - Note: all pruning levels (L1/L2/L3/L4) upsample 2x — the network's output is always 2x the packed-input resolution, not just L1's.
    - Example:
      ```
      python test.py <input_dir> <output_dir> \
          --height 1080 --width 1920 \
          --in_bitwidth 12 --out_bitwidth 8 \
          --bayer_pattern 3 \        # 0=BGGR, 1=GBRG, 2=GRBG, 3=RGGB
          --model pretrained_model/params_e123.pth \
          --level 3 \                # pruning depth L1/L2/L3/L4
          --margin 16 \
          --device cuda
      ```
      This recursively finds every `.raw` file under `input_dir` and writes a matching `.bin` (raw HWC output at `--out_bitwidth`) plus a `.png` (for a quick visual check) under `output_dir`, mirroring the input's directory layout.

- `export_onnx.py` + `tile_infer.py`: ONNX export and tiled inference
    - `export_onnx.py` exports a checkpoint's **L1 exit only** (the cheapest pruning level) to a fixed-shape ONNX graph. `--height`/`--width` (default 256, in packed `[R,G1,G2,B]`-plane space, i.e. half the raw image's resolution) get baked into the graph; `--dtype fp32` or `fp16`.
      ```
      python export_onnx.py pretrained_model/params_e123.pth \
          --height 256 --width 256 --dtype fp32
      ```
    - `tile_infer.py` runs that fixed-shape graph over an arbitrary-size real raw dump by splitting it into overlapping tiles and stitching the results back together (same Bayer-phase alignment, gamma round-trip and `--margin` edge padding as `test.py`).
      ```
      python tile_infer.py --onnx pretrained_model/params_e123_fp32_256x256_L1.onnx \
          <input_dir> <output_dir> \
          --height 1080 --width 1920 --bayer_pattern 3 \
          --in_bitwidth 12 --out_bitwidth 8 \
          --margin 16 --dtype fp32
      ```
    - `margin=16` was verified (with the real pretrained checkpoint) to be enough to fully absorb this network's receptive field (empirically ~30 raw pixels) — both at internal tile-to-tile seams and at the image's true outer edge, matching a single-shot `test.py` run to float32 rounding precision.


- Dataset:
    - We have two prepared demo datesets Kodak24 and McMaster
    - For Training, you should prepare on your own
    - For evaluating, you can use our prepared one and also use your own one(it should be arranged in the manner as below)
    - If using your own datasets, donnot forget to change settings (filepath, etc.) in config.py
    
- The folder structure of the data folder should be:

``` DataSets File Arrangement:

DMUnet++  
|
└── Datasets  
    ├── Training  
    |   └── YOUR TRAINING SETS  
    |       └── train 
    |           └── img00001.jpg (example) 
    |           └── img00002.jpg (example) 
    |           └── img00003.jpg (example) 
    |           └── ...
    └── Evaluating  
        ├── Kodak24
        |   └── test 
        |       └── kodim01.png 
        |       └── kodim02.png 
        |       └── kodim03.png
        |       └── ...
        ├── McMaster
        |   └── test 
        |       └── 1.tif 
        |       └── 2.tif 
        |       └── 3.tif
        |       └── ...
        |
        └── YOUR TESTING SETS
            └── test 
                └── img01.png 
                └── img02.png 
                └── img03.png
                └── ...

```


