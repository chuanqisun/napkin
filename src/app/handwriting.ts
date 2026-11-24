import { cachedFetchBytes, cachedFetchJSON, MODEL_URLS } from './assets';
import {
  DEFAULTS,
  WRITING_STYLES,
  type GravesMeta,
  type InkStroke,
  type PenId,
  type ShapeItem,
  type StrokePoint,
  type StyleKey,
  type TextItem,
  type TextOwner,
  type WorkerRequest,
  type WorkerResponse,
} from './types';

export interface WordLayoutResult {
  strokes: InkStroke[];
  bbox: { x: number; y: number; w: number; h: number } | null;
}

interface SynthJob {
  reqId: number;
  text: string;
  words: string[];
  cx: number;
  cy: number;
  targetH: number;
  maxW: number;
  color: string;
  align: 'center' | 'left';
  styleKey: StyleKey;
  penId: PenId;
  baseW: number;
  targetShape: ShapeItem | null;
  textItem: (TextItem | TextOwner) | null;
  wordBufs: number[][];
  wordSeeds: number[];
  wordStarted: boolean[];
  pendingCount: number;
  doneCount: number;
  resolve: (() => void) | null;
}

export function layoutWordsToStrokes(
  wordBuffers: readonly (readonly number[])[],
  anchorX: number,
  anchorY: number,
  targetH: number,
  maxW: number,
  color: string,
  baseW = 1.65,
  align: 'center' | 'left' = 'center',
  penId: PenId = 'ballpoint',
): WordLayoutResult {
  const wordBoxes: { segs: [number, number][][]; minX: number; w: number }[] = [];
  let totalRawW = 0;
  let globalMinY = 1e9;
  let globalMaxY = -1e9;
  const wordGap = 14.0;

  for (const buf of wordBuffers) {
    const segs: [number, number][][] = [];
    let x = 0;
    let y = 0;
    let cur: [number, number][] = [];
    let minX = 1e9;
    let maxX = -1e9;
    let minY = 1e9;
    let maxY = -1e9;
    const n = buf.length / 3;
    for (let i = 0; i < n; i++) {
      const dx = buf[i * 3]!;
      const dy = buf[i * 3 + 1]!;
      const eos = buf[i * 3 + 2]!;
      x += dx;
      y -= dy;
      cur.push([x, y]);
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (eos > 0.5) {
        if (cur.length >= 2) segs.push(cur);
        else if (cur.length === 1) {
          const p0 = cur[0]!;
          segs.push([
            [p0[0] - 0.16, p0[1] - 0.16],
            [p0[0] + 0.16, p0[1] + 0.16],
          ]);
        }
        cur = [];
      }
    }
    if (cur.length >= 2) segs.push(cur);
    else if (cur.length === 1) {
      const p0 = cur[0]!;
      segs.push([
        [p0[0] - 0.16, p0[1] - 0.16],
        [p0[0] + 0.16, p0[1] + 0.16],
      ]);
    }
    if (!segs.length) {
      wordBoxes.push({ segs: [], minX: 0, w: 10 });
      totalRawW += 10 + wordGap;
      continue;
    }
    if (minY < globalMinY) globalMinY = minY;
    if (maxY > globalMaxY) globalMaxY = maxY;
    const w = Math.max(4, maxX - minX);
    wordBoxes.push({ segs, minX, w });
    totalRawW += w + wordGap;
  }

  if (globalMinY > globalMaxY) return { strokes: [], bbox: null };
  totalRawW = Math.max(1, totalRawW - wordGap);

  const bh = Math.max(10, globalMaxY - globalMinY);
  let sc = targetH / bh;
  if (maxW && totalRawW * sc > maxW) sc = maxW / totalRawW;
  const cy = (globalMinY + globalMaxY) * 0.5;
  let curX = align === 'left' ? anchorX : anchorX - totalRawW * sc * 0.5;

  const strokes: InkStroke[] = [];
  let outMinX = 1e9;
  let outMaxX = -1e9;
  let outMinY = 1e9;
  let outMaxY = -1e9;

  for (const wb of wordBoxes) {
    for (const seg of wb.segs) {
      const pts: StrokePoint[] = seg.map(([px, py], idx) => {
        const prev = seg[Math.max(0, idx - 1)]!;
        const sp = Math.hypot(px - prev[0], py - prev[1]);
        const t = idx / Math.max(1, seg.length - 1);
        const taper = seg.length <= 3 ? 0.88 : Math.min(1, t * 6, (1 - t) * 6);
        const p = Math.max(0.24, Math.min(0.94, (0.88 - sp * 0.045) * (0.45 + 0.55 * taper)));
        const sx = curX + (px - wb.minX) * sc;
        const sy = anchorY + (py - cy) * sc;
        if (sx < outMinX) outMinX = sx;
        if (sx > outMaxX) outMaxX = sx;
        if (sy < outMinY) outMinY = sy;
        if (sy > outMaxY) outMaxY = sy;
        return [sx, sy, p];
      });
      strokes.push({ pts, alpha: 0.96, baseW, color, penId });
    }
    curX += (wb.w + wordGap) * sc;
  }

  return {
    strokes,
    bbox:
      outMinX < outMaxX
        ? {
            x: outMinX,
            y: outMinY,
            w: Math.max(16, outMaxX - outMinX),
            h: Math.max(20, outMaxY - outMinY),
          }
        : null,
  };
}

export interface HandwritingEngine {
  initWasmPool(): Promise<void>;
  nextReqId(): number;
  isIdle(): boolean;
  synthesizeCursiveStreaming(
    text: string,
    cx: number,
    cy: number,
    targetH: number,
    maxW: number,
    color: string,
    targetShape: ShapeItem | null,
    textItem: (TextItem | TextOwner) | null,
    styleKey: StyleKey,
    align: 'center' | 'left',
    fallbackPenId: PenId,
  ): Promise<void>;
}

export function createHandwritingEngine(onRenderNeeded: () => void): HandwritingEngine {
  const POOL_SIZE = Math.min(3, Math.max(2, navigator.hardwareConcurrency || 2));
  const readyWorkers: Worker[] = [];
  let workerRR = 0;
  let reqCounter = 0;
  const activeJobs = new Map<number, SynthJob>();
  const preReadyQueue: Array<() => void> = [];

  function isTextItem(obj: TextItem | TextOwner | null): obj is TextItem {
    return obj !== null && 'kind' in obj && obj.kind === 'text';
  }

  function applyJobStrokes(job: SynthJob): void {
    const { strokes, bbox } = layoutWordsToStrokes(
      job.wordBufs,
      job.cx,
      job.cy,
      job.targetH,
      job.maxW,
      job.color,
      job.baseW,
      job.align,
      job.penId,
    );
    const cache = job.words.map((w, i) => ({
      word: w,
      buf: job.wordBufs[i] ?? [],
      seed: job.wordSeeds[i] ?? 1,
    }));
    if (job.targetShape) {
      job.targetShape.textStrokes = strokes;
      job.targetShape.wordCache = cache;
      job.targetShape.wordCacheStyle = job.styleKey;
    } else if (job.textItem) {
      if (isTextItem(job.textItem)) {
        job.textItem.strokes = strokes;
        if (bbox) {
          job.textItem.x = bbox.x;
          job.textItem.y = bbox.y;
          job.textItem.w = bbox.w;
          job.textItem.h = bbox.h;
        }
      }
      job.textItem.wordCache = cache;
      job.textItem.wordCacheStyle = job.styleKey;
    }
    onRenderNeeded();
  }

  function onWorkerChunk(msg: Extract<WorkerResponse, { type: 'chunk' }>): void {
    const job = activeJobs.get(msg.reqId);
    if (!job) return;
    const owner = job.targetShape ?? job.textItem;
    if (owner && owner.latestReqId !== job.reqId) {
      activeJobs.delete(msg.reqId);
      return;
    }
    if (!job.wordStarted[msg.wordIdx]) {
      job.wordStarted[msg.wordIdx] = true;
      job.wordBufs[msg.wordIdx] = [];
    }
    const arr = job.wordBufs[msg.wordIdx]!;
    for (let i = 0; i < msg.chunk.length; i++) arr.push(msg.chunk[i]!);
    if (msg.done) job.doneCount++;

    applyJobStrokes(job);

    if (job.doneCount >= job.pendingCount) {
      activeJobs.delete(msg.reqId);
      job.resolve?.();
    }
  }

  function spawnOneWorker(
    wasmModule: WebAssembly.Module,
    binBuf: ArrayBuffer,
    meta: GravesMeta,
    isFirst: boolean,
    onFirstReady: (() => void) | null,
  ): void {
    const w = new Worker(new URL('./hw-worker.ts', import.meta.url), { type: 'module' });
    w.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if (msg.type === 'ready') {
        readyWorkers.push(w);
        if (isFirst) {
          onFirstReady?.();
          while (preReadyQueue.length) preReadyQueue.shift()?.();
        }
      } else if (msg.type === 'chunk') {
        onWorkerChunk(msg);
      } else if (msg.type === 'error') {
        console.error('WASM worker error:', msg.error);
      }
    };
    const copy = binBuf.slice(0);
    const initMsg: WorkerRequest = { type: 'init', wasmModule, binBuf: copy, meta };
    w.postMessage(initMsg, [copy]);
  }

  async function initWasmPool(): Promise<void> {
    const [meta, wasmBytes, binBuf] = await Promise.all([
      cachedFetchJSON<GravesMeta>(MODEL_URLS.gravesMeta),
      cachedFetchBytes(MODEL_URLS.gravesWasm),
      cachedFetchBytes(MODEL_URLS.gravesBin),
    ]);
    const wasmModule = await WebAssembly.compile(wasmBytes);
    return new Promise<void>(resolve => {
      spawnOneWorker(wasmModule, binBuf, meta, true, () => {
        resolve();
        for (let i = 1; i < POOL_SIZE; i++) {
          setTimeout(() => spawnOneWorker(wasmModule, binBuf, meta, false, null), i * 160);
        }
      });
    });
  }

  function synthesizeCursiveStreaming(
    text: string,
    cx: number,
    cy: number,
    targetH: number,
    maxW: number,
    color: string,
    targetShape: ShapeItem | null,
    textItem: (TextItem | TextOwner) | null,
    styleKey: StyleKey,
    align: 'center' | 'left',
    fallbackPenId: PenId,
  ): Promise<void> {
    const words = text.trim().split(/\s+/).filter(Boolean);
    const owner = targetShape ?? textItem;
    if (!words.length) {
      if (targetShape) {
        targetShape.textStrokes = [];
        targetShape.wordCache = [];
      } else if (isTextItem(textItem)) {
        textItem.strokes = [];
        textItem.wordCache = [];
      }
      onRenderNeeded();
      return Promise.resolve();
    }
    if (!readyWorkers.length) {
      return new Promise<void>(resolve => {
        preReadyQueue.push(() => {
          const liveText = targetShape
            ? targetShape.label || ''
            : isTextItem(textItem)
              ? textItem.text || ''
              : text;
          synthesizeCursiveStreaming(
            liveText,
            cx,
            cy,
            targetH,
            maxW,
            color,
            targetShape,
            textItem,
            styleKey,
            align,
            fallbackPenId,
          ).then(resolve);
        });
      });
    }

    const st = WRITING_STYLES[styleKey];
    const reqId = ++reqCounter;
    if (owner) owner.latestReqId = reqId;

    const prevCache =
      owner && owner.wordCacheStyle === styleKey && Array.isArray(owner.wordCache) ? owner.wordCache : [];
    const wordBufs: number[][] = [];
    const wordSeeds: number[] = [];
    const wordStarted: boolean[] = [];
    const toSynth: { wordIdx: number; word: string; seed: number }[] = [];

    for (let i = 0; i < words.length; i++) {
      const word = words[i]!;
      const prev = prevCache[i];
      if (prev && prev.word === word && prev.buf && prev.buf.length > 0) {
        wordBufs.push(prev.buf);
        wordSeeds.push(prev.seed);
        wordStarted.push(true);
      } else {
        wordBufs.push(prev?.buf ? prev.buf : []);
        const seed = prev?.seed ? prev.seed : Math.floor(Math.random() * 1e9) + 1;
        wordSeeds.push(seed);
        wordStarted.push(false);
        toSynth.push({ wordIdx: i, word, seed });
      }
    }

    const job: SynthJob = {
      reqId,
      text,
      words,
      cx,
      cy,
      targetH,
      maxW,
      color,
      align,
      styleKey,
      penId: owner?.penId ?? fallbackPenId,
      baseW: targetShape ? 1.55 : 1.72,
      targetShape,
      textItem,
      wordBufs,
      wordSeeds,
      wordStarted,
      pendingCount: toSynth.length,
      doneCount: 0,
      resolve: null,
    };

    if (toSynth.length === 0) {
      applyJobStrokes(job);
      return Promise.resolve();
    }
    applyJobStrokes(job);

    return new Promise<void>(resolve => {
      job.resolve = resolve;
      activeJobs.set(reqId, job);
      toSynth.forEach(({ wordIdx, word, seed }) => {
        const w = readyWorkers[workerRR++ % readyWorkers.length]!;
        const msg: WorkerRequest = {
          type: 'synth_word',
          reqId,
          wordIdx,
          word,
          styleId: st.wasmStyleId,
          dkMult: st.dkMult,
          bias: st.bias,
          seed,
          chunkSteps: DEFAULTS.streamChunkSteps,
        };
        w.postMessage(msg);
      });
    });
  }

  return {
    initWasmPool,
    nextReqId: () => ++reqCounter,
    isIdle: () => activeJobs.size === 0 && preReadyQueue.length === 0,
    synthesizeCursiveStreaming,
  };
}
