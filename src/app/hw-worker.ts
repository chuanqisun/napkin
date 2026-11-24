/// <reference lib="webworker" />
import type { WorkerRequest, WorkerResponse } from './types';

interface GravesWasmExports {
  memory: WebAssembly.Memory;
  get_weights_ptr(): number;
  get_chars_ptr(): number;
  get_out_ptr(): number;
  init_offsets(
    l1_dq: number,
    l1_dsc: number,
    l1_cq: number,
    l1_csc: number,
    l1_b: number,
    l2_dq: number,
    l2_dsc: number,
    l2_cq: number,
    l2_csc: number,
    l2_b: number,
    l3_dq: number,
    l3_dsc: number,
    l3_cq: number,
    l3_csc: number,
    l3_b: number,
    attn_WT: number,
    attn_b: number,
    gmm_WT: number,
    gmm_b: number,
    primed: number,
  ): void;
  init_session(styleId: number, len: number, bias: number, dkMult: number, maxSteps: number, seed: number): void;
  step_chunk(chunkSteps: number): number;
}

const ALPHABET = [
  '\x00', ' ', '!', '"', '#', "'", '(', ')', ',', '-', '.',
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
  '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
  'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y',
  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
  'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
  'y', 'z',
] as const;

const CHAR_MAP = new Map<string, number>(ALPHABET.map((c, i) => [c, i]));

let ex: GravesWasmExports | null = null;
let memI32: Int32Array | null = null;
let memF32: Float32Array | null = null;
let tailTexts: string[] | null = null;
let cPtr = 0;
let oPtr = 0;

function encodeIndices(str: string): Int32Array {
  const cleaned = str.replace(/Q/g, 'q').replace(/X/g, 'x').replace(/Z/g, 'z');
  const out = new Int32Array(cleaned.length + 1);
  for (let i = 0; i < cleaned.length; i++) {
    out[i] = CHAR_MAP.get(cleaned[i]!) ?? 1;
  }
  out[cleaned.length] = 0;
  return out;
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  if (msg.type === 'init') {
    try {
      const instance = await WebAssembly.instantiate(msg.wasmModule, {});
      ex = instance.exports as unknown as GravesWasmExports;
      const memU8 = new Uint8Array(ex.memory.buffer);
      memI32 = new Int32Array(ex.memory.buffer);
      memF32 = new Float32Array(ex.memory.buffer);

      const wPtr = ex.get_weights_ptr();
      cPtr = ex.get_chars_ptr();
      oPtr = ex.get_out_ptr();
      memU8.set(new Uint8Array(msg.binBuf), wPtr);

      const t = msg.meta.tensors;
      tailTexts = Array.isArray(msg.meta.tail_texts) ? msg.meta.tail_texts : null;
      ex.init_offsets(
        t.l1_dq.byte_offset, t.l1_dsc.byte_offset, t.l1_cq.byte_offset, t.l1_csc.byte_offset, t.l1_b.byte_offset,
        t.l2_dq.byte_offset, t.l2_dsc.byte_offset, t.l2_cq.byte_offset, t.l2_csc.byte_offset, t.l2_b.byte_offset,
        t.l3_dq.byte_offset, t.l3_dsc.byte_offset, t.l3_cq.byte_offset, t.l3_csc.byte_offset, t.l3_b.byte_offset,
        t.attn_WT.byte_offset, t.attn_b.byte_offset,
        t.gmm_WT.byte_offset, t.gmm_b.byte_offset,
        t.primed.byte_offset,
      );
      const reply: WorkerResponse = { type: 'ready' };
      self.postMessage(reply);
    } catch (err) {
      const reply: WorkerResponse = { type: 'error', error: String(err) };
      self.postMessage(reply);
    }
    return;
  }

  if (msg.type === 'synth_word') {
    if (!ex || !memI32 || !memF32) return;
    const { reqId, wordIdx, word, styleId, dkMult, bias, seed, chunkSteps } = msg;
    const prefix = tailTexts?.[styleId] ?? ' ';
    const idxs = encodeIndices(prefix + word + '   ');
    memI32.set(idxs, cPtr >> 2);
    const maxSteps = Math.max(160, word.length * 56);

    const t0 = performance.now();
    ex.init_session(styleId, idxs.length, bias, dkMult, maxSteps, seed >>> 0);

    let totalSteps = 0;
    let firstChunkMs = 0;
    const base = oPtr >> 2;

    while (true) {
      const r = ex.step_chunk(chunkSteps);
      const n = Math.abs(r);
      const done = r <= 0;
      const nowMs = performance.now() - t0;
      if (totalSteps === 0) firstChunkMs = nowMs;
      totalSteps += n;

      const chunk = new Float32Array(n * 3);
      chunk.set(memF32.subarray(base, base + n * 3));
      const reply: WorkerResponse = {
        type: 'chunk',
        reqId,
        wordIdx,
        n,
        done,
        ms: nowMs,
        firstChunkMs,
        totalSteps,
        chunk,
      };
      self.postMessage(reply, [chunk.buffer]);
      if (done) break;
    }
  }
};
