/**
 * Benchmark & visually verify v5 Hand-Written WAT SIMD128 Graves engine (compiled via WABT wat2wasm)
 * vs v4 clang WASM SIMD128!
 */
import fs from 'fs';
import { performance } from 'perf_hooks';

const ALPHABET = [
  '\x00', ' ', '!', '"', '#', "'", '(', ')', ',', '-', '.',
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
  '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
  'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y',
  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
  'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
  'y', 'z'
];
const CHAR_MAP = new Map(ALPHABET.map((c, i) => [c, i]));

function encodeIndices(str) {
  const cleaned = str.replace(/Q/g, 'q').replace(/X/g, 'x').replace(/Z/g, 'z');
  const out = new Int32Array(cleaned.length + 1);
  for (let i = 0; i < cleaned.length; i++) out[i] = CHAR_MAP.get(cleaned[i]) ?? 1;
  out[cleaned.length] = 0;
  return out;
}

const wasmBytes = fs.readFileSync('/home/antigravity/napkin-ml/exports/handwriting/graves_v5.wasm');
const meta = JSON.parse(fs.readFileSync('/home/antigravity/napkin-ml/exports/handwriting/graves_v5_meta.json', 'utf8'));
const binBuf = fs.readFileSync('/home/antigravity/napkin-ml/exports/handwriting/graves_v5.bin');

const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const ex = instance.exports;
const memU8 = new Uint8Array(ex.memory.buffer);
const memI32 = new Int32Array(ex.memory.buffer);
const memF32 = new Float32Array(ex.memory.buffer);

const wPtr = ex.get_weights_ptr();
const cPtr = ex.get_chars_ptr();
const oPtr = ex.get_out_ptr();
memU8.set(binBuf, wPtr);

const t = meta.tensors;
ex.init_offsets(
  t.l1_dq.byte_offset, t.l1_dsc.byte_offset, t.l1_cq.byte_offset, t.l1_csc.byte_offset, t.l1_b.byte_offset,
  t.l2_dq.byte_offset, t.l2_dsc.byte_offset, t.l2_cq.byte_offset, t.l2_csc.byte_offset, t.l2_b.byte_offset,
  t.l3_dq.byte_offset, t.l3_dsc.byte_offset, t.l3_cq.byte_offset, t.l3_csc.byte_offset, t.l3_b.byte_offset,
  t.attn_WT.byte_offset, t.attn_b.byte_offset,
  t.gmm_WT.byte_offset, t.gmm_b.byte_offset,
  t.primed.byte_offset
);

function synthWord(word, styleId = 9, dkMult = 1.06, bias = 1.05, seed = 42, chunk = 2048) {
  const idxs = encodeIndices(' ' + word + '  ');
  memI32.set(idxs, cPtr >> 2);
  const maxSteps = Math.max(110, word.length * 40);
  const t0 = performance.now();
  ex.init_session(styleId, idxs.length, bias, dkMult, maxSteps, seed);
  const pts = [];
  let totalSteps = 0;
  let firstChunkMs = null;
  while (true) {
    const r = ex.step_chunk(chunk);
    const n = Math.abs(r);
    if (firstChunkMs === null) firstChunkMs = performance.now() - t0;
    const base = oPtr >> 2;
    for (let i = 0; i < n * 3; i++) pts.push(memF32[base + i]);
    totalSteps += n;
    if (r <= 0) break;
  }
  const ms = performance.now() - t0;
  return { n: totalSteps, ms, firstChunkMs, pts };
}

// Warmup V8 TurboFan Wasm compiler
synthWord('Warmup', 9, 1.06, 1.05, 1);

const tests = [
  { phrase: 'Sketch on Napkin', style: 9 },
  { phrase: 'Parse Strokes', style: 1 },
  { phrase: 'Raw Idea', style: 7 },
  { phrase: 'Export Vector SVG', style: 12 },
];

console.log('=== v5 Hand-Written WAT SIMD128 (4.8 KB, wat2wasm) ===');
const outSamples = [];
for (const tc of tests) {
  const words = tc.phrase.split(/\s+/);
  const wordRes = words.map((w, wi) => synthWord(w, tc.style, 1.06, 1.05, 42 + wi, 28));
  const sumSteps = wordRes.reduce((a, r) => a + r.n, 0);
  const sumMs = wordRes.reduce((a, r) => a + r.ms, 0);
  const parMs = Math.max(...wordRes.map(r => r.ms));
  const firstFrameMs = Math.max(...wordRes.map(r => r.firstChunkMs));
  console.log(
    `  "${tc.phrase.padEnd(18)}" (st ${String(tc.style).padStart(2)}) -> ` +
    `per-step=${(sumMs / sumSteps).toFixed(3)}ms | ` +
    `1-thread=${sumMs.toFixed(1)}ms | ` +
    `parallel-words=${parMs.toFixed(1)}ms | ` +
    `first-ink(28pt)=${firstFrameMs.toFixed(1)}ms`
  );
  outSamples.push({ phrase: tc.phrase, style: tc.style, parMs, firstFrameMs, perStep: sumMs / sumSteps, words: wordRes });
}

fs.writeFileSync('/tmp/v5_wat_samples.json', JSON.stringify(outSamples));
