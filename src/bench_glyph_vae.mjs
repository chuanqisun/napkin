/**
 * Benchmark Method B (Per-Character Non-Autoregressive Glyph-VAE) in Node.js JS.
 */
import fs from 'fs';
import { performance } from 'perf_hooks';

const M = JSON.parse(fs.readFileSync('/home/antigravity/napkin-ml/exports/glyph_vae.json', 'utf8'));
const charMap = new Map(M.chars.map((c, i) => [c, i]));

function silu(x) { return x / (1 + Math.exp(-x)); }
function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x)))); }

function decodeChar(ch, z) {
  const cid = charMap.get(ch);
  if (cid === undefined) return null;
  const emb = M.char_emb[cid];
  let h = new Float32Array(emb.length + z.length);
  for (let i = 0; i < emb.length; i++) h[i] = emb[i];
  for (let i = 0; i < z.length; i++) h[emb.length + i] = z[i];

  for (let li = 0; li < M.layers.length; li++) {
    const { w, b } = M.layers[li];
    const out = new Float32Array(b.length);
    const isLast = li === M.layers.length - 1;
    for (let i = 0; i < b.length; i++) {
      let s = b[i];
      const row = w[i];
      for (let j = 0; j < row.length; j++) s += row[j] * h[j];
      out[i] = isLast ? s : silu(s);
    }
    h = out;
  }
  const K = M.k_pts;
  const pts = new Array(K);
  for (let k = 0; k < K; k++) {
    pts[k] = [h[k * 4], h[k * 4 + 1], sigmoid(h[k * 4 + 2]), sigmoid(h[k * 4 + 3])];
  }
  return { pts, adv: h[K * 4] };
}

function synthPhrase(text, styleIdx = 0) {
  const z = M.styles[styleIdx].z;
  const t0 = performance.now();
  let chars = 0;
  for (const ch of text) {
    if (ch === ' ') continue;
    decodeChar(ch, z);
    chars++;
  }
  return { chars, ms: performance.now() - t0 };
}

// Warmup V8 JIT
for (let i = 0; i < 20; i++) synthPhrase('Warmup Phrase', 0);

const phrases = ['Raw Idea', 'Parse Strokes', 'Sketch on Napkin', 'Export Vector SVG'];
console.log(`=== Method B: Per-Char Glyph-VAE (${(fs.statSync('/home/antigravity/napkin-ml/exports/glyph_vae.json').size / 1024).toFixed(1)} KB, ${M.metrics.decoder_params} params) ===`);
for (const p of phrases) {
  let sum = 0;
  const N = 25;
  let chs = 0;
  for (let i = 0; i < N; i++) {
    const r = synthPhrase(p, 1);
    sum += r.ms;
    chs = r.chars;
  }
  const avg = sum / N;
  console.log(`  "${p.padEnd(18)}" (${String(chs).padStart(2)} chars) -> ${avg.toFixed(2).padStart(5)} ms (${(avg / chs).toFixed(3)} ms/char)`);
}
