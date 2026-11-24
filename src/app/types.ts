export type ToolId =
  | 'sketch'
  | 'text'
  | 'box'
  | 'oval'
  | 'diamond'
  | 'arrow'
  | 'move';

export type ShapeType = 'box' | 'oval' | 'diamond';
export type DragToolId = ShapeType | 'arrow';
export type PenId = 'ballpoint' | 'pencil' | 'marker' | 'fountain' | 'mech';
export type StyleKey = 'upright' | 'cursive' | 'print' | 'journal';
export type DrinkId = 'coffee_latte' | 'black_coffee' | 'matcha_pure' | 'matcha_latte';
export type VolumeKey = '100' | '65' | '30' | '05';

export type CompanionClipId =
  | 'hmm_pause'
  | 'mm_ponder'
  | 'sigh_block'
  | 'ooo_intrigued'
  | 'aha_spark'
  | 'eureka_win'
  | 'sip_coffee'
  | 'swallow_liquid'
  | 'sip_ahhh'
  | 'cup_setdown'
  | 'refill_pour'
  | 'coffee_drip'
  | 'oops_drip'
  | 'argh_stain';

export type CupPhase =
  | 'idle'
  | 'entering'
  | 'replacing_out'
  | 'exiting'
  | 'drinking'
  | 'returning'
  | 'refilling'
  | 'crumple_lift'
  | 'crumple_wait'
  | 'crumple_return';

export type Point2D = { x: number; y: number };
export type StrokePoint = [x: number, y: number, pressure: number];

export interface InkStroke {
  pts: StrokePoint[];
  alpha: number;
  baseW: number;
  color: string;
  penId: PenId;
}

export interface WordCacheEntry {
  word: string;
  buf: number[];
  seed: number;
}

export interface TextOwner {
  latestReqId?: number;
  wordCache?: WordCacheEntry[];
  wordCacheStyle?: StyleKey;
  penId?: PenId;
}

export interface ShapeItem extends TextOwner {
  kind: 'shape';
  shapeType: ShapeType;
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  styleKey: StyleKey;
  penId: PenId;
  color: string;
  strokes: InkStroke[];
  textStrokes: InkStroke[];
}

export interface ArrowItem {
  kind: 'arrow';
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  x: number;
  y: number;
  w: number;
  h: number;
  penId: PenId;
  color: string;
  strokes: InkStroke[];
}

export interface TextItem extends TextOwner {
  kind: 'text';
  text: string;
  styleKey: StyleKey;
  penId: PenId;
  anchorX: number;
  anchorY: number;
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
  strokes: InkStroke[];
}

export interface HandwriteItem {
  kind: 'handwrite';
  styleKey: StyleKey;
  penId: PenId;
  color: string;
  rawList: Point2D[][];
  strokes: InkStroke[];
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FreehandItem {
  kind: 'free';
  penId: PenId;
  color: string;
  strokes: InkStroke[];
  x: number;
  y: number;
  w: number;
  h: number;
}

export type NapkinItem = ShapeItem | ArrowItem | TextItem | HandwriteItem | FreehandItem;

export interface WritingStyleSpec {
  readonly label: string;
  readonly wasmStyleId: number;
  readonly bias: number;
  readonly dkMult: number;
  readonly hwSlant: number;
  readonly hwChaikin: number;
  readonly hwMinDist: number;
  readonly hwBaseW: number;
  readonly hwMinP: number;
  readonly hwVelDrop: number;
  readonly hwBaselinePull: number;
}

export interface PenTypeSpec {
  readonly id: PenId;
  readonly name: string;
  readonly short: string;
  readonly hint: string;
  readonly img: string;
}

export interface VAELayer {
  w: number[][];
  b: number[];
}

export interface VAEModel {
  latent_dim: number;
  k: number;
  res_scale: number;
  layers: VAELayer[];
}

export interface GravesTensorOffset {
  byte_offset: number;
  bytes?: number;
  shape?: number[];
}

export interface GravesMeta {
  tail_texts?: string[];
  tensors: {
    l1_dq: GravesTensorOffset;
    l1_dsc: GravesTensorOffset;
    l1_cq: GravesTensorOffset;
    l1_csc: GravesTensorOffset;
    l1_b: GravesTensorOffset;
    l2_dq: GravesTensorOffset;
    l2_dsc: GravesTensorOffset;
    l2_cq: GravesTensorOffset;
    l2_csc: GravesTensorOffset;
    l2_b: GravesTensorOffset;
    l3_dq: GravesTensorOffset;
    l3_dsc: GravesTensorOffset;
    l3_cq: GravesTensorOffset;
    l3_csc: GravesTensorOffset;
    l3_b: GravesTensorOffset;
    attn_WT: GravesTensorOffset;
    attn_b: GravesTensorOffset;
    gmm_WT: GravesTensorOffset;
    gmm_b: GravesTensorOffset;
    primed: GravesTensorOffset;
  };
}

export type WorkerRequest =
  | {
      type: 'init';
      wasmModule: WebAssembly.Module;
      binBuf: ArrayBuffer;
      meta: GravesMeta;
    }
  | {
      type: 'synth_word';
      reqId: number;
      wordIdx: number;
      word: string;
      styleId: number;
      dkMult: number;
      bias: number;
      seed: number;
      chunkSteps: number;
    };

export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'error'; error: string }
  | {
      type: 'chunk';
      reqId: number;
      wordIdx: number;
      n: number;
      done: boolean;
      ms: number;
      firstChunkMs: number;
      totalSteps: number;
      chunk: Float32Array;
    };

export const DEFAULTS = {
  isPres: 0.75,
  tau: 0.65,
  underSketch: true,
  debounceMs: 600,
  streamChunkSteps: 28,
} as const;

export const WRITING_STYLES = {
  upright: {
    label: 'Clean Upright',
    wasmStyleId: 7,
    bias: 0.98,
    dkMult: 1.0,
    hwSlant: 0.0,
    hwChaikin: 1,
    hwMinDist: 2.4,
    hwBaseW: 1.72,
    hwMinP: 0.3,
    hwVelDrop: 0.03,
    hwBaselinePull: 0.14,
  },
  cursive: {
    label: 'Designer Cursive',
    wasmStyleId: 9,
    bias: 0.95,
    dkMult: 1.0,
    hwSlant: -0.1,
    hwChaikin: 1,
    hwMinDist: 2.3,
    hwBaseW: 1.75,
    hwMinP: 0.22,
    hwVelDrop: 0.038,
    hwBaselinePull: 0.0,
  },
  print: {
    label: 'Crisp Print',
    wasmStyleId: 1,
    bias: 0.98,
    dkMult: 1.0,
    hwSlant: 0.02,
    hwChaikin: 0,
    hwMinDist: 2.6,
    hwBaseW: 1.62,
    hwMinP: 0.42,
    hwVelDrop: 0.022,
    hwBaselinePull: 0.08,
  },
  journal: {
    label: 'Fast Journal',
    wasmStyleId: 12,
    bias: 0.94,
    dkMult: 1.0,
    hwSlant: -0.2,
    hwChaikin: 2,
    hwMinDist: 2.2,
    hwBaseW: 1.58,
    hwMinP: 0.18,
    hwVelDrop: 0.048,
    hwBaselinePull: 0.0,
  },
} as const satisfies Record<StyleKey, WritingStyleSpec>;

export const STYLE_KEYS = Object.keys(WRITING_STYLES) as StyleKey[];
