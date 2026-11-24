# Napkin Sketch

A browser-first, zero-dependency digital paper napkin for organic sketching and real-time neural handwriting. Draw rough shapes that snap into designer-grade vector ink, type directly onto the napkin in streaming cursive, hear velocity-matched pen-on-paper foley, and crumple finished sheets away with 3D physics.

![Napkin Sketch Demo](docs/v7.gif)

## Quick Start

No Python ML libraries, build steps, or GPU required—serve the repository locally and open the app:

```bash
python3 -m http.server 8765
```

Then open **`http://localhost:8765/web/v7_napkin.html`** in your browser.

![Napkin Sketch Interface](docs/v7.webp)

## Features

- **Smart Sketch (`S`) & Explicit Shapes (`B`, `O`, `D`, `A`):** Freehand single- or multi-stroke drawings (600 ms debounce) auto-beautify into organic boxes, ovals, diamonds, and arrows using a 21k-param OpenSketch `(dx, dy, pressure)` Residual VAE with subtle construction under-sketches.
- **Direct Handwrite (`H`):** Freehand writing styled with velocity-pressure tapering, smoothing, and slant matching the active writing style.
- **Live On-Napkin Typing (`T`):** Single-click anywhere on the napkin or inside any shape to type directly on the paper—no secondary input box. Hand-written `4.9 KB` WebAssembly SIMD128 (`graves_v5.wasm`) streams cursive words in parallel Web Workers (`~30–55 ms`) with per-word caching across 4 writing styles (*Designer Cursive*, *Crisp Print*, *Clean Upright*, *Fast Journal*).
- **Select, Move, Resize & Delete (`V`):** Drag any shape, arrow, handwritten burst, or text block, resize via 4 corner handles (regenerating crisp OpenSketch strokes on release), double-click to edit labels, or press `Backspace`/`Delete` to remove.
- **Procedurally Unique Paper Napkins:** Every fresh sheet has its own warm kraft/ecru/linen tone, seamless embossed texture (`dimple`, `linen`, `diamond`, `pinpoint`), optional embossed café border, tilted quarter-folds, corner handling creases, natural pulp flecks, and occasional faint coffee-cup ring.
- **Sound Board & Velocity-Matched Drawing SFX:** Collapsible `Sound` panel with 8 procedural Web Audio drawing surfaces (*Ballpoint Pen*, *2B Graphite Pencil*, *Felt-Tip Marker*, *Fountain Nib*, *Chalk / Charcoal*, *Soft Ink Brush*, *0.5mm Mechanical*, *ASMR Kraft Napkin*) whose pitch, grain rate, and volume track real-time drawing speed, plus an interactive Speed Test Pad and optional typing pen SFX.
- **3D Crumple & Toss Clear:** Clicking the trash icon crumples the napkin (populated or blank) into a 3D faceted paper ball choreographed to `web/crumble.mp3` and tosses it off-screen, revealing a brand-new unique napkin underneath.
- **Export (`Save`):** Download either a **2× HiDPI PNG** or an **editable Vector SVG**—both preserving the full textured napkin background and `multiply`-blended ink.

## Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `S` / `H` / `T` | Sketch / Handwrite / Text |
| `B` / `O` / `D` / `A` | Box / Oval / Diamond / Arrow |
| `V` (or `M`) | Select, Move & Resize |
| `Backspace` / `Delete` | Delete selected item |
| `Ctrl+Z` / `Cmd+Z` | Undo |
| `Enter` / `Escape` | Commit / Cancel active text |

## Architecture & Models

| Component | Files | Details |
| :--- | :--- | :--- |
| **App & UI** | `web/v7_napkin.html`, `web/hw_worker_v5.js`, `web/crumble.mp3` | Single-file HTML/Canvas/Web Audio app with parallel Web Worker pool and Cache API asset persistence. |
| **Designer Stroke VAE** | `exports/opensketch_residual_vae.json` (`433 KB`) | Trained on 24,541 real Wacom strokes from 15 industrial designers (**OpenSketch**, CC0) for `(rx, ry, pressure)` residuals. |
| **Streaming Cursive WASM** | `exports/handwriting/graves_v5.wasm` (`4.9 KB`), `graves_v5.bin` (`4.05 MB`), `graves_v5_meta.json` | Hand-written WAT SIMD128 (`src/graves_v5.wat`) with sparse character-column skipping, `f32x4` LSTM gates, INT8 quantized weights, 4-char primed attention states, and delayed crossbar/dot completion. |

## Rebuilding WASM (Optional)

```bash
wat2wasm src/graves_v5.wat -o exports/handwriting/graves_v5.wasm
```

## Documentation

- [Iteration Report (v1 → v7)](docs/report.md)

## Data Licenses

- **OpenSketch** (Inria): CC0
- **IAM-OnDB pretrained handwriting weights** (`Rybib/rytability-magic-handwriting` / `sjvasquez/handwriting-synthesis`): research/non-commercial terms of IAM-OnDB apply.
