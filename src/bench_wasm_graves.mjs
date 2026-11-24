/**
 * Benchmark & verify Method C (WASM SIMD128 INT8 + Pre-Cached Primed State)
 * against v2 Uncached Pure-JS FP32 and Cached Pure-JS FP32.
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

// 1. Load WASM SIMD128 engine
const wasmBytes = fs.readFileSync('/home/antigravity/napkin-ml/exports/handwriting/graves_simd.wasm');
const meta = JSON.parse(fs.readFileSync('/home/antigravity/napkin-ml/exports/handwriting/graves_fast_meta.json', 'utf8'));
const binBuf = fs.readFileSync('/home/antigravity/napkin-ml/exports/handwriting/graves_fast_int8.bin');

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
  t.lstm1_Wq.byte_offset, t.lstm1_Wsc.byte_offset, t.lstm1_b.byte_offset,
  t.lstm2_Wq.byte_offset, t.lstm2_Wsc.byte_offset, t.lstm2_b.byte_offset,
  t.lstm3_Wq.byte_offset, t.lstm3_Wsc.byte_offset, t.lstm3_b.byte_offset,
  t.attn_WT.byte_offset, t.attn_b.byte_offset,
  t.gmm_WT.byte_offset, t.gmm_b.byte_offset,
  t.primed_states.byte_offset
);

function synthWasm(text, styleId = 9, bias = 0.9, seed = 42) {
  const full = meta.tail_texts[styleId] + text + ' ';
  const idxs = encodeIndices(full);
  memI32.set(idxs, cPtr >> 2);
  const maxSteps = Math.max(220, text.length * 48);
  const t0 = performance.now();
  const n = ex.synthesize(styleId, idxs.length, bias, maxSteps, seed);
  const ms = performance.now() - t0;
  const pts = Array.from(memF32.subarray(oPtr >> 2, (oPtr >> 2) + n * 3));
  return { n, ms, pts };
}

// Warmup JIT
synthWasm('Warmup', 9, 0.9, 1);

const phrases = ['Raw Idea', 'Parse Strokes', 'Sketch on Napkin', 'Export Vector SVG'];
console.log('=== Method C: WASM SIMD128 INT8 (Cached Primed State) ===');
const samples = [];
for (let i = 0; i < phrases.length; i++) {
  const p = phrases[i];
  const r = synthWasm(p, i === 1 ? 1 : (i === 3 ? 7 : 9), 0.92, 42 + i);
  console.log(`  "${p.padEnd(18)}" -> ${String(r.n).padStart(3)} steps in ${r.ms.toFixed(1).padStart(6)} ms (${(r.ms / r.n).toFixed(3)} ms/step)`);
  samples.push({ phrase: p, style: i === 1 ? 1 : (i === 3 ? 7 : 9), n: r.n, ms: r.ms, pts: r.pts });
}

// 2. Compare against v2 pure-JS FP32 matvec speed (100 steps extrapolated to uncached & cached)
const origMeta = JSON.parse(fs.readFileSync('/home/antigravity/napkin-ml/exports/handwriting/hand_synth.json', 'utf8'));
const origRaw = new Float32Array(fs.readFileSync('/home/antigravity/napkin-ml/exports/handwriting/hand_synth.bin').buffer);
const getT = name => {
  const info = origMeta.tensors[name];
  const sz = info.shape.reduce((a, b) => a * b, 1);
  return origRaw.subarray(info.offset, info.offset + sz);
};
const l1W = getT('lstm1_W'), l2W = getT('lstm2_W'), l3W = getT('lstm3_W');
const in1 = new Float32Array(476).fill(0.1), in2 = new Float32Array(876).fill(0.1), outG = new Float32Array(1600);

function jsMatvec(x, W, inD, outD, out) {
  out.fill(0);
  for (let i = 0; i < inD; i++) {
    const xv = x[i], base = i * outD;
    for (let j = 0; j < outD; j++) out[j] += xv * W[base + j];
  }
}

const tJs0 = performance.now();
const BENCH_STEPS = 50;
for (let s = 0; s < BENCH_STEPS; s++) {
  jsMatvec(in1, l1W, 476, 1600, outG);
  jsMatvec(in2, l2W, 876, 1600, outG);
  jsMatvec(in2, l3W, 876, 1600, outG);
}
const jsPerStep = (performance.now() - tJs0) / BENCH_STEPS;
const avgWasmPerStep = samples.reduce((a, s) => a + s.ms / s.n, 0) / samples.length;
const targetSteps = samples[2].n; // "Sketch on Napkin"
const primeSteps = 775; // style 9

console.log('\n=== Speedup Summary ("Sketch on Napkin", style 9) ===');
console.log(`  v2 Pure-JS Uncached (775 prime + ${targetSteps} gen): ~${((primeSteps + targetSteps) * jsPerStep).toFixed(0)} ms (${jsPerStep.toFixed(3)} ms/step)`);
console.log(`  Pure-JS Cached      (0 prime   + ${targetSteps} gen): ~${(targetSteps * jsPerStep).toFixed(0)} ms (${jsPerStep.toFixed(3)} ms/step)`);
console.log(`  Method C WASM SIMD  (0 prime   + ${targetSteps} gen):  ${samples[2].ms.toFixed(1)} ms (${avgWasmPerStep.toFixed(3)} ms/step)`);
console.log(`  Total speedup vs v2: ${(((primeSteps + targetSteps) * jsPerStep) / samples[2].ms).toFixed(1)}x faster!`);

fs.writeFileSync('/tmp/wasm_samples.json', JSON.stringify(samples));
