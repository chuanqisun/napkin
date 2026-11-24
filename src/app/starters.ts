import type { ArrowItem, ShapeItem, ShapeType, TextItem } from './types';

export interface StarterDiagramAPI {
  shape(type: ShapeType, x: number, y: number, w: number, h: number, label: string, color?: string): ShapeItem;
  arrow(x1: number, y1: number, x2: number, y2: number, color?: string): ArrowItem;
  note(cx: number, cy: number, text: string, targetH?: number, maxW?: number, color?: string): TextItem;
}

export interface StarterDiagramSpec {
  readonly name: string;
  readonly w: number;
  readonly h: number;
  build(ox: number, oy: number, api: StarterDiagramAPI): void;
}

const INK = '#1d3557';
const RED = '#9b2226';

export const STARTER_DIAGRAMS: readonly StarterDiagramSpec[] = [
  {
    name: 'startup_plan',
    w: 580,
    h: 220,
    build(ox, oy, api) {
      api.note(ox + 280, oy + 14, 'Billion Dollar Plan', 26, 300, INK);
      api.shape('oval', ox + 0, oy + 76, 138, 60, 'Big Idea', INK);
      api.arrow(ox + 144, oy + 106, ox + 194, oy + 106, INK);
      api.shape('box', ox + 200, oy + 72, 146, 68, 'Magic', INK);
      api.arrow(ox + 352, oy + 106, ox + 402, oy + 106, INK);
      api.shape('oval', ox + 408, oy + 76, 150, 60, 'Profit', RED);
      api.note(ox + 273, oy + 172, 'figure out later', 22, 190, RED);
    },
  },
  {
    name: 'coffee_loop',
    w: 500,
    h: 270,
    build(ox, oy, api) {
      api.shape('oval', ox + 20, oy + 12, 150, 58, 'Tired', INK);
      api.arrow(ox + 178, oy + 41, ox + 288, oy + 41, INK);
      api.shape('box', ox + 296, oy + 10, 182, 62, 'Drink Coffee', RED);
      api.arrow(ox + 387, oy + 78, ox + 387, oy + 176, INK);
      api.shape('box', ox + 296, oy + 184, 182, 62, 'Too Awake', INK);
      api.arrow(ox + 288, oy + 215, ox + 178, oy + 215, INK);
      api.shape('box', ox + 14, oy + 184, 158, 62, 'Overthink', INK);
      api.arrow(ox + 94, oy + 176, ox + 94, oy + 78, INK);
      api.note(ox + 236, oy + 128, 'daily cycle', 23, 150, INK);
    },
  },
  {
    name: 'should_i_tree',
    w: 520,
    h: 260,
    build(ox, oy, api) {
      api.shape('diamond', ox + 24, oy + 18, 196, 94, 'Should I', INK);
      api.arrow(ox + 228, oy + 65, ox + 324, oy + 65, INK);
      api.note(ox + 274, oy + 44, 'Yes', 21, 60, INK);
      api.shape('box', ox + 332, oy + 32, 166, 66, 'Do It', INK);
      api.arrow(ox + 122, oy + 118, ox + 122, oy + 176, INK);
      api.note(ox + 150, oy + 145, 'No', 21, 50, RED);
      api.shape('box', ox + 12, oy + 184, 232, 66, 'Do It Anyway', RED);
    },
  },
  {
    name: 'meeting_flow',
    w: 595,
    h: 265,
    build(ox, oy, api) {
      api.shape('oval', ox + 0, oy + 34, 158, 62, 'Quick Sync', INK);
      api.arrow(ox + 164, oy + 65, ox + 218, oy + 65, INK);
      api.shape('diamond', ox + 224, oy + 16, 206, 98, 'Need Meeting', INK);
      api.arrow(ox + 436, oy + 65, ox + 494, oy + 65, INK);
      api.note(ox + 462, oy + 44, 'No', 21, 50, INK);
      api.arrow(ox + 327, oy + 120, ox + 327, oy + 174, INK);
      api.note(ox + 358, oy + 144, 'Yes', 21, 60, INK);
      api.shape('box', ox + 210, oy + 182, 236, 66, 'Send An Email', RED);
      api.note(ox + 536, oy + 66, 'Email', 24, 90, RED);
    },
  },
  {
    name: 'software_bugs',
    w: 600,
    h: 230,
    build(ox, oy, api) {
      api.shape('box', ox + 0, oy + 24, 152, 66, 'One Bug', INK);
      api.arrow(ox + 158, oy + 57, ox + 210, oy + 57, INK);
      api.shape('box', ox + 216, oy + 24, 158, 66, 'Clever Fix', INK);
      api.arrow(ox + 380, oy + 57, ox + 430, oy + 57, INK);
      api.shape('oval', ox + 436, oy + 24, 156, 66, 'Three Bugs', RED);
      api.arrow(ox + 514, oy + 98, ox + 514, oy + 162, RED);
      api.arrow(ox + 506, oy + 166, ox + 298, oy + 166, RED);
      api.arrow(ox + 295, oy + 162, ox + 295, oy + 98, RED);
      api.note(ox + 404, oy + 194, 'ship it anyway', 23, 180, INK);
    },
  },
  {
    name: 'cat_physics',
    w: 520,
    h: 250,
    build(ox, oy, api) {
      api.note(ox + 252, oy + 14, 'Cat Logic', 27, 180, INK);
      api.shape('box', ox + 10, oy + 62, 180, 62, 'Fancy Bed', INK);
      api.arrow(ox + 198, oy + 93, ox + 286, oy + 93, INK);
      api.shape('oval', ox + 294, oy + 62, 166, 62, 'Ignored', INK);
      api.shape('box', ox + 10, oy + 164, 180, 62, 'Empty Box', INK);
      api.arrow(ox + 198, oy + 195, ox + 286, oy + 195, RED);
      api.shape('oval', ox + 294, oy + 164, 166, 62, 'Pure Joy', RED);
    },
  },
  {
    name: 'weekend_plan',
    w: 590,
    h: 270,
    build(ox, oy, api) {
      api.shape('box', ox + 0, oy + 30, 150, 62, 'Big Plans', INK);
      api.arrow(ox + 156, oy + 61, ox + 206, oy + 61, INK);
      api.shape('diamond', ox + 212, oy + 14, 174, 94, 'Energy', INK);
      api.arrow(ox + 392, oy + 61, ox + 444, oy + 61, INK);
      api.note(ox + 416, oy + 40, 'Yes', 20, 50, INK);
      api.shape('box', ox + 450, oy + 30, 134, 62, 'Snacks', INK);
      api.arrow(ox + 299, oy + 114, ox + 299, oy + 178, INK);
      api.note(ox + 274, oy + 146, 'No', 20, 50, INK);
      api.arrow(ox + 516, oy + 98, ox + 392, oy + 198, INK);
      api.shape('oval', ox + 218, oy + 186, 164, 64, 'Take Nap', RED);
    },
  },
  {
    name: 'brainstorm_funnel',
    w: 440,
    h: 290,
    build(ox, oy, api) {
      api.shape('box', ox + 40, oy + 4, 330, 60, '100 Wild Ideas', INK);
      api.arrow(ox + 205, oy + 70, ox + 205, oy + 112, INK);
      api.shape('box', ox + 85, oy + 118, 240, 58, '3 Okay Ones', INK);
      api.arrow(ox + 205, oy + 182, ox + 205, oy + 222, RED);
      api.shape('oval', ox + 108, oy + 228, 194, 60, 'Order Pizza', RED);
    },
  },
  {
    name: 'triangle_tradeoff',
    w: 460,
    h: 270,
    build(ox, oy, api) {
      api.shape('oval', ox + 154, oy + 4, 146, 58, 'Good', INK);
      api.shape('oval', ox + 10, oy + 196, 146, 58, 'Fast', INK);
      api.shape('oval', ox + 298, oy + 196, 146, 58, 'Cheap', INK);
      api.arrow(ox + 184, oy + 68, ox + 102, oy + 188, INK);
      api.arrow(ox + 270, oy + 68, ox + 352, oy + 188, INK);
      api.arrow(ox + 166, oy + 226, ox + 288, oy + 226, INK);
      api.note(ox + 226, oy + 144, 'pick one maybe', 23, 175, RED);
    },
  },
  {
    name: 'napkin_classic',
    w: 560,
    h: 285,
    build(ox, oy, api) {
      api.shape('oval', ox + 16, oy + 0, 190, 60, 'Weird Thought', INK);
      api.arrow(ox + 110, oy + 66, ox + 110, oy + 112, INK);
      api.shape('box', ox + 0, oy + 118, 224, 64, 'Sketch On Napkin', INK);
      api.arrow(ox + 230, oy + 150, ox + 306, oy + 150, INK);
      api.shape('diamond', ox + 312, oy + 104, 178, 92, 'Genius', INK);
      api.arrow(ox + 401, oy + 202, ox + 401, oy + 240, RED);
      api.note(ox + 401, oy + 260, 'Crumple It', 23, 140, RED);
    },
  },
];

export function pickStarterDiagram(): StarterDiagramSpec {
  let lastIdx = -1;
  try {
    const raw = sessionStorage.getItem('napkin_starter_idx');
    if (raw !== null) lastIdx = parseInt(raw, 10);
  } catch {
    // Ignore storage errors
  }
  let idx = Math.floor(Math.random() * STARTER_DIAGRAMS.length);
  if (STARTER_DIAGRAMS.length > 1 && idx === lastIdx) {
    idx = (idx + 1 + Math.floor(Math.random() * (STARTER_DIAGRAMS.length - 1))) % STARTER_DIAGRAMS.length;
  }
  try {
    sessionStorage.setItem('napkin_starter_idx', String(idx));
  } catch {
    // Ignore storage errors
  }
  return STARTER_DIAGRAMS[idx]!;
}
