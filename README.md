# Napkin Sketch — Organic Vector Ink & Symbolic Sketch ML

Browser-first, non-diffusion pipeline for turning clean computerized diagrams into organic napkin sketches (`Computerized Drawing -> Napkin Sketch <-> Text`).

## Quick Start (Run Locally — Zero Dependencies)

No Python ML libraries or GPU needed to run the browser prototypes—just serve the folder locally:

```bash
python3 -m http.server 8765
```

Then open in your browser:
* **v4 Interactive Napkin (Method A vs B vs C)** *(Interactive drawing + Per-Char Glyph-VAE `~1 ms` + WASM SIMD128 INT8 cached cursive in a Web Worker + live A/B/C latency benchmark)*:  
  `http://localhost:8765/web/v4_draw.html`
* **v3 Interactive Napkin** *(Preserved v3 baseline: Smart Sketch auto-beautify + Freehand ink + `<1 ms` OpenSketch-ink handwriting)*:  
  `http://localhost:8765/web/v3_draw.html`
* **v2 Napkin Engine** *(OpenSketch `(dx, dy, pressure)` stroke residuals + pure-JS IAM-OnDB handwriting)*:  
  `http://localhost:8765/web/v2_napkin.html`
* **v1 Live Inference Studio** *(Conditional Sketch-GRU MDN + 68-feature Semantic Gesture MLP)*:  
  `http://localhost:8765/web/`
* **UI/UX Workflow Variations** *(Split-Pane Studio, Unified Living Napkin, Margin Sketchpad)*:  
  `http://localhost:8765/web/ux_explorations.html`

## Handwriting Latency Benchmarks (`"Sketch on Napkin"`)

| Engine | Architecture | Weights Size | Measured Latency (2-vCPU VM) | Speedup vs v2 |
| :--- | :--- | :--- | :--- | :--- |
| **v2 Baseline (Pure-JS)** | Uncached Graves LSTM (775 prime + 529 gen steps) | 14.5 MB FP32 | `~14,169 ms` (`10.87 ms/step`) | 1.0× |
| **Method C (WASM SIMD128)** | Cached-state Graves LSTM + INT8 `i32x4.dot_i16x8` in Web Worker | 4.0 MB INT8 + 13 KB `.wasm` | **`218 ms`** (`0.41 ms/step`, 0 ms UI block) | **64.9×** |
| **Method B (Glyph-VAE)** | Non-autoregressive Per-Char Conditional VAE (47k params, 6 styles) | 299 KB JSON | **`1.26 ms`** (`0.09 ms/char`) | **~11,200×** |
| **Method A (v3 Baseline)** | Curated single-stroke glyphs + OpenSketch Residual VAE | 433 KB JSON | **`<1 ms`** | **>14,000×** |

## Included Trained Models (`exports/`)

| Model | File | Size | Dataset & Training |
| :--- | :--- | :--- | :--- |
| **PerCharGlyphVAE** *(v4 — Method B Writing)* | `exports/glyph_vae.json` | 299 KB | 47k-param non-autoregressive VAE (`val xy_rmse=0.0167`, `pen_acc=100%`, `pressure_MAE=0.0379`) grounded in OpenSketch Wacom kinematics across 6 writer styles & 73 chars. |
| **Fast Graves WASM SIMD128** *(v4 — Method C Writing)* | `exports/handwriting/graves_simd.wasm` + `graves_fast_int8.bin` | 13 KB + 4.0 MB | Pre-cached primed states for all 13 IAM-OnDB writers + per-row INT8 quantized weights (`RMSE=0.0013`) compiled with `clang --target=wasm32 -O3 -msimd128`. |
| **DesignerStrokeResidualVAE** *(Drawing)* | `exports/opensketch_residual_vae.json` | 433 KB | 24,541 real Wacom strokes from 15 professional designers (**OpenSketch**, CC0). Learns `(rx, ry, pressure)` residuals over 32 arc-length steps. |
| **Graves (2013) Handwriting** *(v2 FP32)* | `exports/handwriting/hand_synth.bin` + `.json` | 14.5 MB | Original uncached FP32 IAM-OnDB LSTM+Attention+MDN weights. |
| **ConditionalStrokeGRU-MDN** *(v1)* | `exports/stroke_gru_mdn.json` | 793 KB | 39k-param GRU + 8-comp bivariate Gaussian MDN trained on cleaned Quick, Draw! vector primitives (`-1.3746` test NLL). |
| **GestureMLP (68→128→64→5)** *(v1)* | `exports/gesture_mlp.json` | 348 KB | 17k-param 32-pt online stroke classifier (`96.96%` test acc) for `emphasize`, `extract`, `summarize`, `compare`, `itemize`. |

## Retraining & Building WASM on CPU

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pillow torch --index-url https://download.pytorch.org/whl/cpu

# 1. OpenSketch (dx, dy, pressure) Designer Residual VAE (~20s on 2 CPUs)
mkdir -p data/opensketch_json
curl -L -o data/opensketch_v1.zip "https://repo-sam.inria.fr/d3/OpenSketch/Data/sketches/sketches_json_first_viewpoint.zip"
unzip -q data/opensketch_v1.zip '*.json' -d data/opensketch_json
python src/train_opensketch_residuals.py

# 2. v4 Method B: Per-Character Glyph-VAE (~29s on 2 CPUs)
python src/train_glyph_vae.py

# 3. v4 Method C: Export pre-cached INT8 Graves weights & compile WASM SIMD128 kernel
python src/export_fast_graves.py
clang --target=wasm32 -O3 -msimd128 -nostdlib -fno-builtin \
  -Wl,--no-entry -Wl,--export=init_offsets -Wl,--export=synthesize \
  -Wl,--export=get_weights_ptr -Wl,--export=get_chars_ptr -Wl,--export=get_out_ptr \
  -Wl,--initial-memory=10485760 -Wl,--max-memory=16777216 \
  -o exports/handwriting/graves_simd.wasm src/graves_wasm.c

# 4. Run Node.js latency benchmarks
node src/bench_glyph_vae.mjs
node src/bench_wasm_graves.mjs
```

## Notes on Data Licenses
* **OpenSketch** (Inria): CC0
* **Quick, Draw!**: CC BY 4.0
* **IAM-OnDB pretrained handwriting weights** (`Rybib/rytability-magic-handwriting` / `sjvasquez/handwriting-synthesis`): research/non-commercial terms of IAM-OnDB apply.
