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
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(2, 22);
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

function decodeAudioToStereo(filePath) {
  const res = spawnSync('ffmpeg', [
    '-v', 'error',
    '-i', filePath,
    '-f', 'f32le',
    '-ac', '2',
    '-ar', String(SR),
    'pipe:1',
  ], { maxBuffer: 128 * 1024 * 1024 });
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

function normalizePeak(clip, targetPeak = 0.88) {
  let peak = 1e-6;
  for (let i = 0; i < clip.n; i++) {
    const a = Math.max(Math.abs(clip.L[i]), Math.abs(clip.R[i]));
    if (a > peak) peak = a;
  }
  const mul = targetPeak / peak;
  for (let i = 0; i < clip.n; i++) {
    clip.L[i] *= mul;
    clip.R[i] *= mul;
  }
  return clip;
}

function mixClip(outL, outR, clip, startSec, gain = 1.0, maxDurSec = null, fadeOutSec = 0.05) {
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

function main() {
  const tlPath = path.resolve('trailer/edit/timeline_v2.json');
  const tl = JSON.parse(fs.readFileSync(tlPath, 'utf8'));
  const totalSec = tl.totalDurationSec;
  const n = Math.ceil(totalSec * SR);

  // Separate stems so we can sidechain-duck music cleanly behind VO & in-game SFX!
  const voL = new Float32Array(n);
  const voR = new Float32Array(n);
  const sfxL = new Float32Array(n);
  const sfxR = new Float32Array(n);
  const musL = new Float32Array(n);
  const musR = new Float32Array(n);

  // 1. Load Gemini Lyria 3 café-bossa music track (trim/offset cleanly past any 0.1s lead-in)
  const musicPath = path.resolve('trailer/audio/music/lyria_cafe_bossa.mp3');
  const lyria = normalizePeak(decodeAudioToStereo(musicPath), 0.85);
  const cues = tl.musicCues || {};
  const splatT = cues.splatCutStart ?? 9.4;
  const resumeT = cues.musicResume ?? 16.0;
  const endT = cues.endCardStart ?? 21.5;

  // Map Lyria music onto timeline with comedic HARD-CUT to silence on splat, crisp resume on fresh napkin clink, and warm fadeout!
  const musicStartOffset = Math.round(0.15 * SR);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const srcIdx = (musicStartOffset + i) % lyria.n;
    let mul = 1.0;
    if (t < 0.08) mul = t / 0.08;
    if (t >= splatT && t < resumeT) {
      mul = t < splatT + 0.025 ? Math.max(0, 1 - (t - splatT) / 0.025) : 0.0;
    } else if (t >= resumeT && t < resumeT + 0.10) {
      mul = (t - resumeT) / 0.10;
    }
    if (t >= endT + 1.9) {
      mul *= Math.max(0, 1 - (t - (endT + 1.9)) / 0.95);
    }
    musL[i] = lyria.L[srcIdx] * mul;
    musR[i] = lyria.R[srcIdx] * mul;
  }

  // 2. Mix ElevenLabs v4 custom VO + Real In-Game SFX + ElevenLabs Studio Foley
  const cache = new Map();
  const load = (key, filePath, normPeak = null) => {
    if (cache.has(key)) return cache.get(key);
    if (!fs.existsSync(filePath)) return null;
    const c = decodeAudioToStereo(filePath);
    if (normPeak) normalizePeak(c, normPeak);
    cache.set(key, c);
    return c;
  };

  for (const ev of tl.audioEvents || []) {
    if (ev.kind === 'vo') {
      const clip = load(`vo:${ev.id}`, path.resolve(`trailer/audio/vo/${ev.id}.mp3`), 0.92);
      if (clip) mixClip(voL, voR, clip, ev.t, ev.gain ?? 1.0);
    } else if (ev.kind === 'clip') {
      const p = ev.id === 'crumble'
        ? path.resolve('web/crumble.mp3')
        : path.resolve(`web/sfx/${ev.id}.mp3`);
      const clip = load(`game:${ev.id}`, p, 0.88);
      if (clip) mixClip(sfxL, sfxR, clip, ev.t, ev.gain ?? 0.95, ev.maxDur ?? null);
    } else if (ev.kind === 'el_sfx') {
      const clip = load(`el:${ev.id}`, path.resolve(`trailer/audio/sfx/${ev.id}.mp3`), 0.85);
      if (clip) mixClip(sfxL, sfxR, clip, ev.t, ev.gain ?? 0.65, ev.maxDur ?? null);
    }
  }

  // 3. Smooth sidechain ducking so music NEVER overwhelms VO or in-game SFX!
  // Compute fast-attack / smooth-release envelope from VO (strong duck) + SFX (moderate duck)
  const outL = new Float32Array(n);
  const outR = new Float32Array(n);
  const baseMusicGain = 0.24; // Gentle, airy café bed (~-12.5dB base)
  let duckEnv = 0.0;
  const atk = 1 - Math.exp(-1 / (0.025 * SR));
  const rel = 1 - Math.exp(-1 / (0.240 * SR));

  for (let i = 0; i < n; i++) {
    const voAbs = Math.max(Math.abs(voL[i]), Math.abs(voR[i]));
    const sfxAbs = Math.max(Math.abs(sfxL[i]), Math.abs(sfxR[i]));
    const targetDuck = Math.min(1.0, voAbs * 3.8 + sfxAbs * 1.9);
    duckEnv += (targetDuck - duckEnv) * (targetDuck > duckEnv ? atk : rel);
    // Duck music down to 35% of baseMusicGain during VO/SFX peaks -> super clear dialogue & foley!
    const mGain = baseMusicGain * (1.0 - 0.65 * duckEnv);

    const l = musL[i] * mGain + sfxL[i] * 0.92 + voL[i] * 1.06;
    const r = musR[i] * mGain + sfxR[i] * 0.92 + voR[i] * 1.06;
    // Gentle soft-clipping master bus limiter
    outL[i] = Math.tanh(l * 1.08) * 0.93;
    outR[i] = Math.tanh(r * 1.08) * 0.93;
  }

  const outWav = path.resolve('trailer/audio/master_mix_v2.wav');
  writeWavStereo(outWav, outL, outR);
  console.log(`[AUDIO V2] Wrote ${outWav} (${totalSec.toFixed(2)}s, Lyria music ducked cleanly behind custom VO + SFX)`);
}

main();
