import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const SR = 48000;

function writeWavStereo(filePath, left, right) {
  const n = left.length;
  const buf = Buffer.alloc(44 + n * 4);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + n * 4, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); // PCM
  buf.writeUInt16LE(2, 22); // stereo
  buf.writeUInt32LE(SR, 24);
  buf.writeUInt32LE(SR * 4, 28);
  buf.writeUInt16LE(4, 32);
  buf.writeUInt16LE(16, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(n * 4, 40);
  for (let i = 0; i < n; i++) {
    const l = Math.max(-0.98, Math.min(0.98, left[i]));
    const r = Math.max(-0.98, Math.min(0.98, right[i]));
    buf.writeInt16LE(Math.round(l * 32767), 44 + i * 4);
    buf.writeInt16LE(Math.round(r * 32767), 44 + i * 4 + 2);
  }
  fs.writeFileSync(filePath, buf);
}

// Decode any mp3/wav to Float32 stereo 48kHz via ffmpeg
function decodeAudioToStereo(filePath) {
  const res = spawnSync('ffmpeg', [
    '-v', 'error',
    '-i', filePath,
    '-f', 'f32le',
    '-ac', '2',
    '-ar', String(SR),
    'pipe:1',
  ], { maxBuffer: 64 * 1024 * 1024 });
  if (res.status !== 0) {
    throw new Error(`ffmpeg decode failed for ${filePath}: ${res.stderr?.toString()}`);
  }
  const f32 = new Float32Array(res.stdout.buffer, res.stdout.byteOffset, res.stdout.byteLength / 4);
  const n = f32.length / 2;
  const L = new Float32Array(n);
  const R = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    L[i] = f32[i * 2];
    R[i] = f32[i * 2 + 1];
  }
  return { L, R, n };
}

function mixClip(outL, outR, clip, startSec, gain = 1.0, maxDurSec = null, fadeOutSec = 0.04) {
  const startIdx = Math.round(startSec * SR);
  const maxN = maxDurSec ? Math.min(clip.n, Math.round(maxDurSec * SR)) : clip.n;
  const fadeN = Math.round(fadeOutSec * SR);
  for (let i = 0; i < maxN; i++) {
    const idx = startIdx + i;
    if (idx < 0 || idx >= outL.length) continue;
    let env = 1.0;
    if (maxDurSec && i > maxN - fadeN) {
      env = Math.max(0, (maxN - i) / fadeN);
    }
    outL[idx] += clip.L[i] * gain * env;
    outR[idx] += clip.R[i] * gain * env;
  }
}

// Deterministic PRNG for warm synthesis
function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Karplus-Strong plucked nylon acoustic guitar note
function addGuitarPluck(outL, outR, startSec, midiNote, durSec, vel = 0.25, pan = 0, rng) {
  const freq = 440 * Math.pow(2, (midiNote - 69) / 12);
  const period = Math.max(8, Math.round(SR / freq));
  const buf = new Float32Array(period);
  // Warm nylon pluck excitation (low-passed noise burst)
  let lp = 0;
  for (let i = 0; i < period; i++) {
    const w = rng() * 2 - 1;
    lp = lp * 0.55 + w * 0.45;
    buf[i] = lp;
  }
  const n = Math.round(durSec * SR);
  const startIdx = Math.round(startSec * SR);
  const gL = Math.cos(((pan + 1) * Math.PI) / 4) * vel;
  const gR = Math.sin(((pan + 1) * Math.PI) / 4) * vel;
  const damp = 0.9962;
  let ptr = 0;
  for (let i = 0; i < n; i++) {
    const cur = buf[ptr];
    const next = buf[(ptr + 1) % period];
    buf[ptr] = (cur * 0.52 + next * 0.48) * damp;
    ptr = (ptr + 1) % period;
    const attack = i < 96 ? i / 96 : 1;
    const rel = i > n - 1400 ? Math.max(0, (n - i) / 1400) : 1;
    const s = cur * attack * rel;
    const idx = startIdx + i;
    if (idx >= 0 && idx < outL.length) {
      outL[idx] += s * gL;
      outR[idx] += s * gR;
    }
  }
}

// Warm upright acoustic bass pluck
function addBassNote(outL, outR, startSec, midiNote, durSec, vel = 0.28) {
  const freq = 440 * Math.pow(2, (midiNote - 69) / 12);
  const n = Math.round(durSec * SR);
  const startIdx = Math.round(startSec * SR);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const env = (i < 240 ? i / 240 : 1) * Math.exp(-t * 3.4);
    const s = (Math.sin(2 * Math.PI * freq * t) * 0.8 + Math.sin(4 * Math.PI * freq * t) * 0.22 * Math.exp(-t * 6)) * env * vel;
    const idx = startIdx + i;
    if (idx >= 0 && idx < outL.length) {
      outL[idx] += s;
      outR[idx] += s;
    }
  }
}

// Warm Rhodes electric piano chord stab (for Eureka & End Card!)
function addRhodesNote(outL, outR, startSec, midiNote, durSec, vel = 0.14, pan = 0) {
  const freq = 440 * Math.pow(2, (midiNote - 69) / 12);
  const n = Math.round(durSec * SR);
  const startIdx = Math.round(startSec * SR);
  const gL = Math.cos(((pan + 1) * Math.PI) / 4) * vel;
  const gR = Math.sin(((pan + 1) * Math.PI) / 4) * vel;
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const env = (i < 160 ? i / 160 : 1) * Math.exp(-t * 1.45) * (i > n - 2400 ? Math.max(0, (n - i) / 2400) : 1);
    const tine = Math.sin(2 * Math.PI * freq * 3 * t) * Math.exp(-t * 14) * 0.18;
    const trem = 0.88 + 0.12 * Math.sin(2 * Math.PI * 4.8 * t);
    const s = (Math.sin(2 * Math.PI * freq * t) + 0.15 * Math.sin(4 * Math.PI * freq * t) + tine) * env * trem;
    const idx = startIdx + i;
    if (idx >= 0 && idx < outL.length) {
      outL[idx] += s * gL;
      outR[idx] += s * gR;
    }
  }
}

// Soft brushed shaker / snare percussive hit
function addBrushHit(outL, outR, startSec, durSec, vel, accent, rng) {
  const n = Math.round(durSec * SR);
  const startIdx = Math.round(startSec * SR);
  let hpPrev = 0;
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const w = rng() * 2 - 1;
    const hp = w - hpPrev * (accent ? 0.72 : 0.88);
    hpPrev = w;
    const env = (i < 90 ? i / 90 : 1) * Math.exp(-t * (accent ? 26 : 42));
    const s = hp * env * vel;
    const idx = startIdx + i;
    if (idx >= 0 && idx < outL.length) {
      outL[idx] += s * 0.85;
      outR[idx] += s * 1.1;
    }
  }
}

// Crisp UI click / pen tap / marker scratch / crash-zoom whoosh / wet splat
function addUiClick(outL, outR, startSec, vel = 0.22) {
  const n = Math.round(0.028 * SR);
  const startIdx = Math.round(startSec * SR);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const s = (Math.sin(2 * Math.PI * 1650 * t) * 0.65 + Math.sin(2 * Math.PI * 3400 * t) * 0.35) * Math.exp(-t * 180) * vel;
    const idx = startIdx + i;
    if (idx >= 0 && idx < outL.length) {
      outL[idx] += s;
      outR[idx] += s;
    }
  }
}

function addPenScratch(outL, outR, startSec, durSec, style = 'fountain', vel = 0.24, rng) {
  const n = Math.round(durSec * SR);
  const startIdx = Math.round(startSec * SR);
  let lp = 0;
  let hpPrev = 0;
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const u = i / Math.max(1, n - 1);
    const w = rng() * 2 - 1;
    const hp = w - hpPrev * 0.78;
    hpPrev = w;
    lp = lp * 0.62 + hp * 0.38;
    const env = Math.sin(u * Math.PI) * (0.72 + 0.28 * Math.sin(t * 28));
    let s = lp;
    if (style === 'marker') {
      // Damp fiber + subtle marker squeak!
      s = lp * 0.65 + Math.sin(2 * Math.PI * (2250 + 240 * Math.sin(t * 14)) * t) * 0.12;
    } else if (style === 'fountain') {
      // Metallic crisp nib grain
      s = hp * 0.55 + lp * 0.5;
    }
    s *= env * vel;
    const idx = startIdx + i;
    if (idx >= 0 && idx < outL.length) {
      outL[idx] += s * 0.95;
      outR[idx] += s * 1.05;
    }
  }
}

function addWetSplat(outL, outR, startSec, vel = 0.45, rng) {
  const n = Math.round(0.24 * SR);
  const startIdx = Math.round(startSec * SR);
  let lp = 0;
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    // Liquid bloop pitch sweep + wet splash noise burst
    const freq = 310 + 780 * Math.min(1, t / 0.065);
    const bloop = Math.sin(2 * Math.PI * freq * t) * Math.exp(-t * 24) * 0.65;
    const w = rng() * 2 - 1;
    lp = lp * 0.72 + w * 0.28;
    const splash = lp * Math.exp(-t * 19) * 0.75;
    const s = (bloop + splash) * vel;
    const idx = startIdx + i;
    if (idx >= 0 && idx < outL.length) {
      outL[idx] += s;
      outR[idx] += s;
    }
  }
}

function addWhoosh(outL, outR, startSec, durSec = 0.16, vel = 0.22, rng) {
  const n = Math.round(durSec * SR);
  const startIdx = Math.round(startSec * SR);
  let lp = 0;
  for (let i = 0; i < n; i++) {
    const u = i / Math.max(1, n - 1);
    const w = rng() * 2 - 1;
    const cut = 0.88 - 0.55 * Math.sin(u * Math.PI);
    lp = lp * cut + w * (1 - cut);
    const env = Math.pow(Math.sin(u * Math.PI), 1.3);
    const s = lp * env * vel;
    const idx = startIdx + i;
    if (idx >= 0 && idx < outL.length) {
      outL[idx] += s * (1.1 - u * 0.2);
      outR[idx] += s * (0.9 + u * 0.2);
    }
  }
}

function buildCafeBossaMusic(totalSec, cues) {
  const n = Math.ceil(totalSec * SR);
  const L = new Float32Array(n);
  const R = new Float32Array(n);
  const rng = makeRng(20261124);

  const bpm = 116;
  const beat = 60 / bpm; // ~0.5172s
  const eighth = beat / 2;

  // Classic cozy Bossa Nova progression: Cmaj9 -> A7 -> Dm9 -> G13
  const chords = [
    { bass: 48, notes: [60, 64, 67, 71, 74] }, // Cmaj9
    { bass: 45, notes: [57, 61, 64, 67, 72] }, // A7
    { bass: 50, notes: [62, 65, 69, 72, 76] }, // Dm9
    { bass: 43, notes: [55, 59, 62, 65, 69] }, // G13
  ];

  // Bossa clave rhythm (in 8th notes per 2-bar / 8-eighth bar): [0, 1.5, 3, 4.5, 6]
  const barDur = beat * 4;
  const totalBars = Math.ceil(totalSec / barDur) + 1;

  for (let bar = 0; bar < totalBars; bar++) {
    const barT = bar * barDur;
    const ch = chords[bar % chords.length];

    // Upright bass (1, 2.5, 3, 4.5)
    addBassNote(L, R, barT + 0 * beat, ch.bass, beat * 1.4, 0.22);
    addBassNote(L, R, barT + 1.5 * beat, ch.bass + 7, beat * 0.9, 0.16);
    addBassNote(L, R, barT + 2.0 * beat, ch.bass, beat * 1.3, 0.20);
    addBassNote(L, R, barT + 3.5 * beat, ch.bass + 5, beat * 0.6, 0.15);

    // Syncopated nylon guitar bossa comping
    const hits = [0, 1.5, 3, 4.5, 6, 7];
    for (const h of hits) {
      const ht = barT + h * eighth;
      if (ht >= totalSec - 1.2) break;
      const isAccent = h === 0 || h === 3 || h === 6;
      ch.notes.slice(0, 4).forEach((note, idx) => {
        addGuitarPluck(L, R, ht + idx * 0.011, note, isAccent ? 0.85 : 0.42, isAccent ? 0.11 : 0.075, -0.25 + idx * 0.16, rng);
      });
    }

    // Playful plucked melody motif on top!
    const melOffsets = [
      [0.5, 76], [2.0, 79], [3.5, 74], [5.0, 76], [6.5, 72],
    ];
    for (const [off8, note] of melOffsets) {
      const mt = barT + off8 * eighth;
      if (mt >= totalSec - 1.5) break;
      const transpose = bar % 4 === 2 ? -2 : bar % 4 === 3 ? -3 : 0;
      addGuitarPluck(L, R, mt, note + transpose, 0.65, 0.085, 0.28, rng);
    }

    // Brushed shaker every eighth note
    for (let e = 0; e < 8; e++) {
      const et = barT + e * eighth;
      if (et >= totalSec - 1.5) break;
      const accent = e === 2 || e === 6;
      addBrushHit(L, R, et, accent ? 0.095 : 0.055, accent ? 0.085 : 0.042, accent, rng);
    }
  }

  // Apply dynamic musical ducks & comedic HARD-CUT TO SILENCE!
  const splatT = cues.splatCutStart ?? 8.2;
  const resumeT = cues.musicResume ?? 13.3;
  const eurekaT = cues.eurekaHit ?? 17.5;
  const endT = cues.endCardStart ?? 19.2;

  // Hard-cut music cleanly to silence on coffee splat -> stays silent through "Oops!" + coffee ring + 3D crumple!
  // Resumes with punchy energy right when the fresh napkin & latte clink land!
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    let mul = 1.0;
    if (t >= splatT && t < resumeT) {
      if (t < splatT + 0.025) mul = Math.max(0, 1 - (t - splatT) / 0.025);
      else mul = 0.0;
    } else if (t >= resumeT && t < resumeT + 0.12) {
      mul = (t - resumeT) / 0.12;
    }
    // Gentle dip right before end card resolve
    if (t >= endT + 1.35) {
      mul *= Math.max(0, 1 - (t - (endT + 1.35)) / 0.9);
    }
    L[i] *= mul;
    R[i] *= mul;
  }

  // Triumphant warm Rhodes + Guitar Cmaj9 stab right on EUREKA!!
  for (const [idx, note] of [48, 60, 64, 67, 71, 74, 79].entries()) {
    addRhodesNote(L, R, eurekaT, note, 1.9, idx === 0 ? 0.15 : 0.11, -0.4 + idx * 0.13);
  }
  // Cozy resolving Cmaj9 final strum on the End Card!
  for (const [idx, note] of [48, 60, 64, 67, 71, 74].entries()) {
    addGuitarPluck(L, R, endT + 0.05 + idx * 0.022, note, 2.2, 0.14, -0.35 + idx * 0.14, rng);
    addRhodesNote(L, R, endT + 0.06, note, 2.1, 0.085, -0.3 + idx * 0.12);
  }

  return { L, R };
}

function main() {
  const timelinePath = path.resolve('trailer/edit/timeline.json');
  if (!fs.existsSync(timelinePath)) {
    throw new Error(`Missing ${timelinePath} — run compose_reels.py first!`);
  }
  const tl = JSON.parse(fs.readFileSync(timelinePath, 'utf8'));
  const totalSec = tl.totalDurationSec;
  const n = Math.ceil(totalSec * SR);
  const outL = new Float32Array(n);
  const outR = new Float32Array(n);
  const rng = makeRng(998877);

  // 1. Build & save bespoke café-bossa music bed
  const music = buildCafeBossaMusic(totalSec, tl.musicCues || {});
  fs.mkdirSync(path.resolve('trailer/audio/music'), { recursive: true });
  writeWavStereo(path.resolve('trailer/audio/music/cafe_bossa.wav'), music.L, music.R);
  for (let i = 0; i < n; i++) {
    outL[i] += music.L[i] * 0.56;
    outR[i] += music.R[i] * 0.56;
  }

  // 2. Load real game SFX clips
  const sfxCache = new Map();
  const loadClip = (id) => {
    if (sfxCache.has(id)) return sfxCache.get(id);
    const p = id === 'crumble'
      ? path.resolve('web/crumble.mp3')
      : path.resolve(`web/sfx/${id}.mp3`);
    if (!fs.existsSync(p)) return null;
    const dec = decodeAudioToStereo(p);
    sfxCache.set(id, dec);
    return dec;
  };

  // 3. Place exact timeline SFX & procedural pen foley
  const clipGains = {
    sigh_block: 0.92,
    cup_setdown: 0.95,
    sip_ahhh: 1.05,
    oops_drip: 0.95,
    crumble: 1.15,
    argh_stain: 0.88,
    aha_spark: 0.78,
    eureka_win: 1.0,
    mm_ponder: 0.75,
    ooo_intrigued: 0.75,
  };

  for (const ev of tl.audioEvents || []) {
    if (ev.kind === 'clip') {
      const clip = loadClip(ev.id);
      if (clip) {
        mixClip(outL, outR, clip, ev.t, (ev.gain ?? clipGains[ev.id] ?? 0.85), ev.maxDur ?? null);
      }
    } else if (ev.kind === 'click') {
      addUiClick(outL, outR, ev.t, ev.gain ?? 0.26);
    } else if (ev.kind === 'scratch') {
      addPenScratch(outL, outR, ev.t, ev.dur ?? 0.28, ev.style ?? 'fountain', ev.gain ?? 0.24, rng);
    } else if (ev.kind === 'splat') {
      addWetSplat(outL, outR, ev.t, ev.gain ?? 0.48, rng);
    } else if (ev.kind === 'whoosh') {
      addWhoosh(outL, outR, ev.t, ev.dur ?? 0.16, ev.gain ?? 0.22, rng);
    }
  }

  // Soft-clip master limiter so mix is punchy and broadcast-clean with zero distortion
  for (let i = 0; i < n; i++) {
    outL[i] = Math.tanh(outL[i] * 1.12) * 0.92;
    outR[i] = Math.tanh(outR[i] * 1.12) * 0.92;
  }

  const masterWav = path.resolve('trailer/audio/master_mix.wav');
  writeWavStereo(masterWav, outL, outR);
  console.log(`[AUDIO] Wrote ${masterWav} (${totalSec.toFixed(2)}s, ${(tl.audioEvents || []).length} synced SFX events)`);
}

main();
