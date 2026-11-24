# Napkin Sketch — Organic Vector Ink & Symbolic Sketch ML

Browser-first, non-diffusion pipeline for turning clean computerized diagrams into organic napkin sketches (`Computerized Drawing -> Napkin Sketch <-> Text`).

## Quick Start (Run Locally — Zero Dependencies)

No Python ML libraries or GPU needed to run the browser prototypes—just serve the folder locally:

```bash
python3 -m http.server 8765
```

Then open in your browser:
* **v3 Interactive Napkin** *(Smart Sketch auto-beautify + Freehand ink + `<1 ms` OpenSketch-ink handwriting)*:  
  `http://localhost:8765/web/v3_draw.html`
* **v2 Napkin Engine** *(OpenSketch `(dx, dy, pressure)` stroke residuals + IAM-OnDB neural handwriting)*:  
  `http://localhost:8765/web/v2_napkin.html`
* **v1 Live Inference Studio** *(Conditional Sketch-GRU MDN + 68-feature Semantic Gesture MLP)*:  
  `http://localhost:8765/web/`
* **UI/UX Workflow Variations** *(Split-Pane Studio, Unified Living Napkin, Margin Sketchpad)*:  
  `http://localhost:8765/web/ux_explorations.html`

## Included Trained Models (`exports/`)

| Model | File | Size | Dataset & Training |
| :--- | :--- | :--- | :--- |
| **DesignerStrokeResidualVAE** *(Idea 1 v2 — Drawing)* | `exports/opensketch_residual_vae.json` | 433 KB | 24,541 real Wacom strokes from 15 professional designers (**OpenSketch**, CC0). Learns `(rx, ry, pressure)` residuals over 32 arc-length steps. |
| **Graves (2013) Handwriting** *(Idea 1 v2 — Writing)* | `exports/handwriting/hand_synth.bin` + `.json` | 14.5 MB | IAM-OnDB LSTM+Attention+MDN with 13 built-in human writer styles + velocity-derived pen pressure tapering. |
| **ConditionalStrokeGRU-MDN** *(Idea 1 v1)* | `exports/stroke_gru_mdn.json` | 793 KB | 39k-param GRU + 8-comp bivariate Gaussian MDN trained on cleaned Quick, Draw! vector primitives (`-1.3746` test NLL). |
| **GestureMLP (68→128→64→5)** *(Idea 2)* | `exports/gesture_mlp.json` | 348 KB | 17k-param 32-pt online stroke classifier (`96.96%` test acc) for `emphasize`, `extract`, `summarize`, `compare`, `itemize`. |

## Retraining on CPU

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy torch --index-url https://download.pytorch.org/whl/cpu

# 1. OpenSketch (dx, dy, pressure) Designer Residual VAE (~20s on 2 CPUs)
mkdir -p data/opensketch_json
curl -L -o data/opensketch_v1.zip "https://repo-sam.inria.fr/d3/OpenSketch/Data/sketches/sketches_json_first_viewpoint.zip"
unzip -q data/opensketch_v1.zip '*.json' -d data/opensketch_json
python src/train_opensketch_residuals.py

# 2. Quick, Draw! Sketch-GRU MDN & GestureMLP (~70s total on 2 CPUs)
python src/fetch_data.py
python src/train_gesture_classifier.py
python src/train_stroke_generator.py
```

## Notes on Data Licenses
* **OpenSketch** (Inria): CC0
* **Quick, Draw!**: CC BY 4.0
* **IAM-OnDB pretrained handwriting weights** (`Rybib/rytability-magic-handwriting` / `sjvasquez/handwriting-synthesis`): research/non-commercial terms of IAM-OnDB apply.
