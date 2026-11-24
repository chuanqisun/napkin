/**
 * Web Worker for Method C: WASM SIMD128 INT8 Cached-State Graves Handwriting
 * Keeps UI canvas drawing 100% non-blocking (0ms main-thread jank).
 */
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

let ex = null;
let meta = null;
let memI32 = null;
let memF32 = null;
let cPtr = 0;
let oPtr = 0;
let initPromise = null;

function encodeIndices(str) {
  const cleaned = str.replace(/Q/g, 'q').replace(/X/g, 'x').replace(/Z/g, 'z');
  const out = new Int32Array(cleaned.length + 1);
  for (let i = 0; i < cleaned.length; i++) out[i] = CHAR_MAP.get(cleaned[i]) ?? 1;
  out[cleaned.length] = 0;
  return out;
}

async function ensureReady() {
  if (ex) return;
  if (!initPromise) {
    initPromise = (async () => {
      const t0 = performance.now();
      const [metaRes, binRes, wasmRes] = await Promise.all([
        fetch('../exports/handwriting/graves_fast_meta.json'),
        fetch('../exports/handwriting/graves_fast_int8.bin'),
        fetch('../exports/handwriting/graves_simd.wasm')
      ]);
      meta = await metaRes.json();
      const [binBuf, wasmBytes] = await Promise.all([binRes.arrayBuffer(), wasmRes.arrayBuffer()]);
      const { instance } = await WebAssembly.instantiate(wasmBytes, {});
      ex = instance.exports;
      const memU8 = new Uint8Array(ex.memory.buffer);
      memI32 = new Int32Array(ex.memory.buffer);
      memF32 = new Float32Array(ex.memory.buffer);

      const wPtr = ex.get_weights_ptr();
      cPtr = ex.get_chars_ptr();
      oPtr = ex.get_out_ptr();
      memU8.set(new Uint8Array(binBuf), wPtr);

      const t = meta.tensors;
      ex.init_offsets(
        t.lstm1_Wq.byte_offset, t.lstm1_Wsc.byte_offset, t.lstm1_b.byte_offset,
        t.lstm2_Wq.byte_offset, t.lstm2_Wsc.byte_offset, t.lstm2_b.byte_offset,
        t.lstm3_Wq.byte_offset, t.lstm3_Wsc.byte_offset, t.lstm3_b.byte_offset,
        t.attn_WT.byte_offset, t.attn_b.byte_offset,
        t.gmm_WT.byte_offset, t.gmm_b.byte_offset,
        t.primed_states.byte_offset
      );
      const loadMs = (performance.now() - t0).toFixed(0);
      self.postMessage({ type: 'ready', loadMs, styleLabels: meta.style_labels, bytes: meta.total_bytes });
    })();
  }
  await initPromise;
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'init') {
    try {
      await ensureReady();
    } catch (err) {
      self.postMessage({ type: 'error', error: String(err) });
    }
    return;
  }
  if (msg.type === 'synth') {
    try {
      await ensureReady();
      const { id, text, styleId = 9, bias = 1.0, seed = 42 } = msg;
      const full = meta.tail_texts[styleId] + text + ' ';
      const idxs = encodeIndices(full);
      memI32.set(idxs, cPtr >> 2);
      const maxSteps = Math.max(220, text.length * 48);
      const t0 = performance.now();
      const n = ex.synthesize(styleId, idxs.length, bias, maxSteps, seed >>> 0);
      const ms = performance.now() - t0;
      // Copy out [dx, dy, eos] flat Float32Array and transfer zero-copy
      const out = new Float32Array(n * 3);
      out.set(memF32.subarray(oPtr >> 2, (oPtr >> 2) + n * 3));
      self.postMessage({ type: 'result', id, n, ms, out }, [out.buffer]);
    } catch (err) {
      self.postMessage({ type: 'error', id: msg.id, error: String(err) });
    }
  }
};
