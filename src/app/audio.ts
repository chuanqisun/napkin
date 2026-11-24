import { getSfxUrl, MODEL_URLS } from './assets';
import type { CompanionClipId, PenId } from './types';

interface DrawPresetSpec {
  readonly id: PenId;
  readonly bodyFMin: number;
  readonly bodyFMax: number;
  readonly bodyQ: number;
  readonly bodyGain: number;
  readonly grainFMin: number;
  readonly grainFMax: number;
  readonly grainQ: number;
  readonly grainGain: number;
  readonly rateMin: number;
  readonly rateMax: number;
  readonly tipType: BiquadFilterType;
  readonly tipFMin: number;
  readonly tipFMax: number;
  readonly tipQ: number;
  readonly tipGain: number;
  readonly tapFreq: number;
  readonly tapDur: number;
  readonly tapGain: number;
  readonly curveBoost: number;
  readonly velScale: number;
  readonly squeak: number;
}

const DRAW_PRESETS = {
  ballpoint: {
    id: 'ballpoint',
    bodyFMin: 720, bodyFMax: 2150, bodyQ: 1.35, bodyGain: 0.24,
    grainFMin: 1550, grainFMax: 3900, grainQ: 2.2, grainGain: 0.44, rateMin: 0.55, rateMax: 1.95,
    tipType: 'bandpass', tipFMin: 3100, tipFMax: 6500, tipQ: 2.7, tipGain: 0.22,
    tapFreq: 1450, tapDur: 0.014, tapGain: 0.16, curveBoost: 0.38, velScale: 1.0, squeak: 0,
  },
  pencil: {
    id: 'pencil',
    bodyFMin: 500, bodyFMax: 1650, bodyQ: 1.5, bodyGain: 0.34,
    grainFMin: 1300, grainFMax: 3450, grainQ: 2.5, grainGain: 0.62, rateMin: 0.45, rateMax: 1.85,
    tipType: 'bandpass', tipFMin: 2550, tipFMax: 5600, tipQ: 2.0, tipGain: 0.34,
    tapFreq: 880, tapDur: 0.018, tapGain: 0.15, curveBoost: 0.48, velScale: 1.05, squeak: 0,
  },
  marker: {
    id: 'marker',
    bodyFMin: 320, bodyFMax: 1120, bodyQ: 1.2, bodyGain: 0.56,
    grainFMin: 740, grainFMax: 1850, grainQ: 1.7, grainGain: 0.34, rateMin: 0.38, rateMax: 1.35,
    tipType: 'bandpass', tipFMin: 1950, tipFMax: 3150, tipQ: 6.2, tipGain: 0.16,
    tapFreq: 460, tapDur: 0.022, tapGain: 0.18, curveBoost: 0.25, velScale: 1.0, squeak: 0.22,
  },
  fountain: {
    id: 'fountain',
    bodyFMin: 900, bodyFMax: 2450, bodyQ: 1.8, bodyGain: 0.18,
    grainFMin: 2350, grainFMax: 5300, grainQ: 3.5, grainGain: 0.48, rateMin: 0.65, rateMax: 2.25,
    tipType: 'bandpass', tipFMin: 4200, tipFMax: 7700, tipQ: 4.1, tipGain: 0.34,
    tapFreq: 2600, tapDur: 0.011, tapGain: 0.18, curveBoost: 0.62, velScale: 0.88, squeak: 0,
  },
  mech: {
    id: 'mech',
    bodyFMin: 1080, bodyFMax: 2800, bodyQ: 2.1, bodyGain: 0.16,
    grainFMin: 2600, grainFMax: 5700, grainQ: 3.9, grainGain: 0.46, rateMin: 0.72, rateMax: 2.4,
    tipType: 'bandpass', tipFMin: 4600, tipFMax: 8300, tipQ: 4.8, tipGain: 0.32,
    tapFreq: 3150, tapDur: 0.01, tapGain: 0.19, curveBoost: 0.52, velScale: 0.85, squeak: 0,
  },
} as const satisfies Record<PenId, DrawPresetSpec>;

interface ClipSpec {
  readonly file: string;
  readonly gain: number;
  readonly dur: number;
}

const COMPANION_CLIPS = {
  hmm_pause:      { file: getSfxUrl('hmm_pause'),      gain: 0.72, dur: 1.76 },
  mm_ponder:      { file: getSfxUrl('mm_ponder'),      gain: 1.4,  dur: 2.0 },
  sigh_block:     { file: getSfxUrl('sigh_block'),     gain: 0.88, dur: 2.0 },
  ooo_intrigued:  { file: getSfxUrl('ooo_intrigued'),  gain: 0.4,  dur: 1.48 },
  aha_spark:      { file: getSfxUrl('aha_spark'),      gain: 0.54, dur: 1.36 },
  eureka_win:     { file: getSfxUrl('eureka_win'),     gain: 0.46, dur: 2.08 },
  sip_coffee:     { file: getSfxUrl('sip_coffee'),     gain: 2.35, dur: 2.0 },
  swallow_liquid: { file: getSfxUrl('swallow_liquid'), gain: 2.45, dur: 1.76 },
  sip_ahhh:       { file: getSfxUrl('sip_ahhh'),       gain: 1.75, dur: 2.48 },
  cup_setdown:    { file: getSfxUrl('cup_setdown'),    gain: 0.8,  dur: 1.2 },
  refill_pour:    { file: getSfxUrl('refill_pour'),    gain: 2.45, dur: 2.4 },
  coffee_drip:    { file: getSfxUrl('coffee_drip'),    gain: 3.5,  dur: 1.08 },
  oops_drip:      { file: getSfxUrl('oops_drip'),      gain: 0.54, dur: 0.88 },
  argh_stain:     { file: getSfxUrl('argh_stain'),     gain: 0.48, dur: 1.36 },
} as const satisfies Record<CompanionClipId, ClipSpec>;

interface VoiceGraph {
  outGain: GainNode;
  bodySrc: AudioBufferSourceNode;
  bodyFlt: BiquadFilterNode;
  bodyGain: GainNode;
  grainSrc: AudioBufferSourceNode;
  grainFlt: BiquadFilterNode;
  grainGain: GainNode;
  tipSrc: AudioBufferSourceNode;
  tipFlt: BiquadFilterNode;
  tipGain: GainNode;
  sqOsc: OscillatorNode;
  sqFlt: BiquadFilterNode;
  sqGain: GainNode;
}

export interface PlayClipOptions {
  channel?: 'vocal' | 'foley';
  interrupt?: boolean;
  volumeMul?: number;
}

export interface AudioEventRecord {
  t: number;
  type: 'clip' | 'crumple' | 'draw_start' | 'draw_stop' | 'type_pulse' | 'pen';
  id: string;
}

export interface NapkinAudioController {
  initUI(): void;
  setPreset(id: PenId): void;
  startDraw(x: number, y: number): void;
  updateDraw(x: number, y: number): void;
  stopDraw(): void;
  triggerTypingPulse(): void;
  prepareCrumple(): Promise<void>;
  playCrumple(): void;
  stopCrumple(): void;
  playClip(id: CompanionClipId, opts?: PlayClipOptions): Promise<{ played: boolean; duration: number }>;
  stopCompanion(): void;
  isVocalBusy(): boolean;
  toggleMute(): void;
  setEventLogger?(fn: ((ev: AudioEventRecord) => void) | null): void;
  getState(): {
    muted: boolean;
    presetId: PenId;
    hasCrumpleBuffer: boolean;
    companionLoaded: number;
    companionTotal: number;
  };
}

export function createNapkinAudio(): NapkinAudioController {
  const DRAW_VOL = 0.75;
  const SPEED_SENS = 0.75;
  const CRUMPLE_VOL = 0.85;
  let activePresetId: PenId = 'ballpoint';
  let muted = false;
  let eventLogger: ((ev: AudioEventRecord) => void) | null = null;
  const logEv = (type: AudioEventRecord['type'], id: string) => {
    eventLogger?.({ t: performance.now(), type, id });
  };

  const currentPreset = (): DrawPresetSpec => DRAW_PRESETS[activePresetId];

  let actx: AudioContext | null = null;
  let masterGain: GainNode | null = null;
  let compressor: DynamicsCompressorNode | null = null;
  let buffers: { pink: AudioBuffer; grain: AudioBuffer; crisp: AudioBuffer } | null = null;
  let voice: VoiceGraph | null = null;

  let crumbleRawBytes: ArrayBuffer | null = null;
  let crumbleFetchPromise: Promise<ArrayBuffer | null> | null = null;
  let crumbleBuffer: AudioBuffer | null = null;
  let crumbleDecodePromise: Promise<AudioBuffer | null> | null = null;
  let activeCrumpleSource: AudioBufferSourceNode | null = null;
  let fallbackCrumpleEl: HTMLAudioElement | null = null;

  let isDrawing = false;
  let lastX = 0;
  let lastY = 0;
  let lastT = 0;
  let lastAng: number | null = null;
  let smoothedSpeed = 0;
  let smoothedCurve = 0;
  let decayRaf: number | null = null;

  let lastTypeTime = 0;
  let typingPulseTimer: ReturnType<typeof setTimeout> | null = null;

  const clipRaw = new Map<CompanionClipId, ArrayBuffer>();
  const clipBuf = new Map<CompanionClipId, AudioBuffer>();
  let activeVocalSrc: AudioBufferSourceNode | null = null;
  const activeFoleySrcs = new Set<AudioBufferSourceNode>();
  let loadedCompanionCount = 0;

  function preloadCrumple(): Promise<ArrayBuffer | null> {
    if (crumbleFetchPromise) return crumbleFetchPromise;
    try {
      fallbackCrumpleEl = new Audio(MODEL_URLS.crumbleMp3);
      fallbackCrumpleEl.preload = 'auto';
    } catch {
      // Ignore audio element errors
    }
    crumbleFetchPromise = fetch(MODEL_URLS.crumbleMp3)
      .then(res => (res.ok ? res.arrayBuffer() : null))
      .then(buf => {
        crumbleRawBytes = buf;
        if (actx && buf && !crumbleBuffer) void decodeCrumple();
        return buf;
      })
      .catch(() => null);
    return crumbleFetchPromise;
  }

  function decodeCrumple(): Promise<AudioBuffer | null> {
    if (crumbleBuffer) return Promise.resolve(crumbleBuffer);
    if (crumbleDecodePromise) return crumbleDecodePromise;
    if (!actx || !crumbleRawBytes) return Promise.resolve(null);
    crumbleDecodePromise = actx
      .decodeAudioData(crumbleRawBytes.slice(0))
      .then(decoded => {
        crumbleBuffer = decoded;
        return decoded;
      })
      .catch(() => {
        crumbleDecodePromise = null;
        return null;
      });
    return crumbleDecodePromise;
  }

  function buildTextureBuffers(ctx: AudioContext) {
    const sr = ctx.sampleRate;
    const len = Math.floor(sr * 3.0);

    const pink = ctx.createBuffer(1, len, sr);
    const pd = pink.getChannelData(0);
    let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
    for (let i = 0; i < len; i++) {
      const w = Math.random() * 2 - 1;
      b0 = 0.99886 * b0 + w * 0.0555179;
      b1 = 0.99332 * b1 + w * 0.0750759;
      b2 = 0.969 * b2 + w * 0.153852;
      b3 = 0.8665 * b3 + w * 0.3104856;
      b4 = 0.55 * b4 + w * 0.5329522;
      b5 = -0.7616 * b5 - w * 0.016898;
      pd[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11;
      b6 = w * 0.115926;
    }

    const grain = ctx.createBuffer(1, len, sr);
    const gd = grain.getChannelData(0);
    let env = 0.5;
    let imp = 0;
    for (let i = 0; i < len; i++) {
      const emboss = 0.65 + 0.35 * Math.sin(i * 0.0038 + Math.sin(i * 0.00043) * 2.5);
      env += (Math.random() - 0.5) * 0.03;
      env = Math.max(0.25, Math.min(1.0, env));
      if (Math.random() < 0.018) imp += (Math.random() * 2 - 1) * 1.35;
      imp *= 0.84;
      const micro = (Math.random() * 2 - 1) * 0.35;
      gd[i] = Math.max(-1, Math.min(1, (imp + micro) * env * emboss));
    }

    const crisp = ctx.createBuffer(1, len, sr);
    const cd = crisp.getChannelData(0);
    let prev = 0;
    for (let i = 0; i < len; i++) {
      const w = Math.random() * 2 - 1;
      const hp = (w - prev) * 0.52;
      prev = w;
      const sparkle = Math.random() < 0.006 ? (Math.random() * 2 - 1) * 0.9 : 0;
      cd[i] = Math.max(-1, Math.min(1, hp + sparkle));
    }

    const fadeN = Math.floor(sr * 0.01);
    for (const arr of [pd, gd, cd]) {
      for (let i = 0; i < fadeN; i++) {
        const k = i / fadeN;
        arr[len - fadeN + i] = arr[len - fadeN + i]! * (1 - k) + arr[i]! * k;
      }
    }
    return { pink, grain, crisp };
  }

  function ensureContext(): AudioContext | null {
    if (!actx) {
      const AC =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AC) return null;
      actx = new AC();
      compressor = actx.createDynamicsCompressor();
      compressor.threshold.value = -10;
      compressor.knee.value = 18;
      compressor.ratio.value = 4;
      compressor.attack.value = 0.003;
      compressor.release.value = 0.12;

      masterGain = actx.createGain();
      masterGain.gain.value = muted ? 0 : 1;
      masterGain.connect(compressor);
      compressor.connect(actx.destination);

      buffers = buildTextureBuffers(actx);
      if (crumbleRawBytes && !crumbleBuffer) void decodeCrumple();
    }
    if (actx.state === 'suspended') {
      actx.resume().catch(() => {});
    }
    return actx;
  }

  function configureVoice(): void {
    if (!voice || !actx) return;
    const p = currentPreset();
    const now = actx.currentTime;
    voice.bodyFlt.Q.setTargetAtTime(p.bodyQ, now, 0.02);
    voice.grainFlt.Q.setTargetAtTime(p.grainQ, now, 0.02);
    voice.tipFlt.type = p.tipType;
    voice.tipFlt.Q.setTargetAtTime(p.tipQ, now, 0.02);
    voice.bodyGain.gain.setTargetAtTime(p.bodyGain, now, 0.02);
    voice.grainGain.gain.setTargetAtTime(p.grainGain, now, 0.02);
    voice.tipGain.gain.setTargetAtTime(p.tipGain, now, 0.02);
  }

  function ensureVoice(): VoiceGraph | null {
    const ctx = ensureContext();
    if (!ctx || !buffers || !masterGain) return null;
    if (voice) return voice;

    const p = currentPreset();
    const outGain = ctx.createGain();
    outGain.gain.value = 0;
    outGain.connect(masterGain);

    const makeLoop = (buf: AudioBuffer) => {
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.loop = true;
      src.start(0, Math.random() * 2.5);
      return src;
    };

    const bodySrc = makeLoop(buffers.pink);
    const bodyFlt = ctx.createBiquadFilter();
    bodyFlt.type = 'bandpass';
    bodyFlt.Q.value = p.bodyQ;
    const bodyGain = ctx.createGain();
    bodyGain.gain.value = p.bodyGain;
    bodySrc.connect(bodyFlt);
    bodyFlt.connect(bodyGain);
    bodyGain.connect(outGain);

    const grainSrc = makeLoop(buffers.grain);
    const grainFlt = ctx.createBiquadFilter();
    grainFlt.type = 'bandpass';
    grainFlt.Q.value = p.grainQ;
    const grainGain = ctx.createGain();
    grainGain.gain.value = p.grainGain;
    grainSrc.connect(grainFlt);
    grainFlt.connect(grainGain);
    grainGain.connect(outGain);

    const tipSrc = makeLoop(buffers.crisp);
    const tipFlt = ctx.createBiquadFilter();
    tipFlt.type = p.tipType;
    tipFlt.Q.value = p.tipQ;
    const tipGain = ctx.createGain();
    tipGain.gain.value = p.tipGain;
    tipSrc.connect(tipFlt);
    tipFlt.connect(tipGain);
    tipGain.connect(outGain);

    const sqOsc = ctx.createOscillator();
    sqOsc.type = 'triangle';
    sqOsc.frequency.value = 2350;
    const sqFlt = ctx.createBiquadFilter();
    sqFlt.type = 'bandpass';
    sqFlt.Q.value = 10;
    sqFlt.frequency.value = 2350;
    const sqGain = ctx.createGain();
    sqGain.gain.value = 0;
    sqOsc.connect(sqFlt);
    sqFlt.connect(sqGain);
    sqGain.connect(outGain);
    sqOsc.start();

    voice = {
      outGain,
      bodySrc,
      bodyFlt,
      bodyGain,
      grainSrc,
      grainFlt,
      grainGain,
      tipSrc,
      tipFlt,
      tipGain,
      sqOsc,
      sqFlt,
      sqGain,
    };
    return voice;
  }

  function playPenTap(strength = 1.0): void {
    if (muted) return;
    const ctx = ensureContext();
    if (!ctx || !buffers || !masterGain) return;
    const p = currentPreset();
    const now = ctx.currentTime;
    const src = ctx.createBufferSource();
    src.buffer = buffers.crisp;
    const flt = ctx.createBiquadFilter();
    flt.type = 'bandpass';
    flt.frequency.value = p.tapFreq * (0.9 + Math.random() * 0.2);
    flt.Q.value = 2.4;
    const g = ctx.createGain();
    const peak = p.tapGain * DRAW_VOL * strength * 0.65;
    g.gain.setValueAtTime(0.0001, now);
    g.gain.linearRampToValueAtTime(peak, now + 0.002);
    g.gain.exponentialRampToValueAtTime(0.0001, now + p.tapDur);
    src.connect(flt);
    flt.connect(g);
    g.connect(masterGain);
    src.start(now, Math.random() * 2.0, p.tapDur + 0.01);
  }

  function applySpeedToVoice(speedPxPerSec: number, curveAccent = 0): void {
    if (muted) {
      if (voice && actx) voice.outGain.gain.setTargetAtTime(0, actx.currentTime, 0.02);
      return;
    }
    const v = ensureVoice();
    if (!v || !actx) return;
    const p = currentPreset();
    const now = actx.currentTime;

    if (speedPxPerSec < 10) {
      v.outGain.gain.setTargetAtTime(0, now, 0.026);
      v.sqGain.gain.setTargetAtTime(0, now, 0.02);
      return;
    }

    const rawU = Math.max(0, Math.min(1, (speedPxPerSec - 10) / 1300));
    const pitchU = Math.pow(rawU, 0.72) * (0.28 + 0.72 * SPEED_SENS);
    const gainU = Math.pow(rawU, 0.45 + 0.48 * SPEED_SENS);
    const logLerp = (a: number, b: number, t: number) => a * Math.pow(b / a, Math.max(0, Math.min(1.15, t)));

    v.bodyFlt.frequency.setTargetAtTime(logLerp(p.bodyFMin, p.bodyFMax, pitchU), now, 0.018);
    v.grainFlt.frequency.setTargetAtTime(logLerp(p.grainFMin, p.grainFMax, pitchU), now, 0.016);
    v.tipFlt.frequency.setTargetAtTime(logLerp(p.tipFMin, p.tipFMax, pitchU), now, 0.016);

    const gRate = p.rateMin + (p.rateMax - p.rateMin) * Math.pow(rawU, 0.78);
    v.grainSrc.playbackRate.setTargetAtTime(Math.max(0.25, Math.min(3.2, gRate)), now, 0.02);
    v.tipSrc.playbackRate.setTargetAtTime(0.75 + rawU * 0.85, now, 0.025);

    v.grainGain.gain.setTargetAtTime(p.grainGain * (1 + curveAccent * p.curveBoost), now, 0.015);
    if (p.squeak > 0 && rawU > 0.35) {
      const sqFreq = 2050 + rawU * 880 + curveAccent * 160;
      v.sqOsc.frequency.setTargetAtTime(sqFreq, now, 0.02);
      v.sqFlt.frequency.setTargetAtTime(sqFreq, now, 0.02);
      v.sqGain.gain.setTargetAtTime(p.squeak * Math.pow((rawU - 0.35) / 0.65, 1.4) * 0.14, now, 0.02);
    } else {
      v.sqGain.gain.setTargetAtTime(0, now, 0.02);
    }

    const targetGain = (0.07 + 0.68 * gainU) * DRAW_VOL * p.velScale;
    v.outGain.gain.setTargetAtTime(Math.min(1.15, targetGain), now, 0.014);
  }

  function tickDecay(): void {
    if (!isDrawing) {
      decayRaf = null;
      return;
    }
    const idleMs = performance.now() - lastT;
    if (idleMs > 28 && smoothedSpeed > 0) {
      smoothedSpeed *= Math.pow(0.74, Math.min(8, idleMs / 16));
      smoothedCurve *= 0.7;
      if (smoothedSpeed < 8) smoothedSpeed = 0;
      applySpeedToVoice(smoothedSpeed, smoothedCurve);
    }
    decayRaf = requestAnimationFrame(tickDecay);
  }

  function startDraw(x: number, y: number): void {
    ensureContext();
    logEv('draw_start', activePresetId);
    isDrawing = true;
    lastX = x;
    lastY = y;
    lastT = performance.now();
    lastAng = null;
    smoothedSpeed = 0;
    smoothedCurve = 0;
    playPenTap(1.0);
    if (!decayRaf) decayRaf = requestAnimationFrame(tickDecay);
  }

  function updateDraw(x: number, y: number): void {
    if (!isDrawing) return;
    const now = performance.now();
    const dt = Math.max(4, now - lastT);
    const dx = x - lastX;
    const dy = y - lastY;
    const dist = Math.hypot(dx, dy);
    if (dist < 0.45) return;

    const instSpeed = Math.min(2600, (dist / dt) * 1000);
    const ang = Math.atan2(dy, dx);
    let curve = 0;
    if (lastAng !== null && dist > 1.5) {
      const dAng = Math.abs(Math.atan2(Math.sin(ang - lastAng), Math.cos(ang - lastAng)));
      curve = Math.min(1, dAng * 1.25);
    }
    const alpha = instSpeed > smoothedSpeed ? 0.46 : 0.28;
    smoothedSpeed = smoothedSpeed * (1 - alpha) + instSpeed * alpha;
    smoothedCurve = smoothedCurve * 0.65 + curve * 0.35;

    lastX = x;
    lastY = y;
    lastT = now;
    lastAng = ang;
    applySpeedToVoice(smoothedSpeed, smoothedCurve);
  }

  function stopDraw(): void {
    if (!isDrawing) return;
    logEv('draw_stop', activePresetId);
    isDrawing = false;
    smoothedSpeed = 0;
    smoothedCurve = 0;
    if (decayRaf) {
      cancelAnimationFrame(decayRaf);
      decayRaf = null;
    }
    applySpeedToVoice(0, 0);
  }

  function triggerTypingPulse(): void {
    if (muted || isDrawing) return;
    logEv('type_pulse', activePresetId);
    const now = performance.now();
    const dt = Math.max(45, Math.min(600, now - lastTypeTime));
    lastTypeTime = now;
    const speed = Math.max(240, Math.min(960, 145000 / dt));
    const durMs = Math.max(58, Math.min(115, dt * 0.45));
    ensureVoice();
    playPenTap(0.55);
    applySpeedToVoice(speed, 0.35);
    if (typingPulseTimer) clearTimeout(typingPulseTimer);
    typingPulseTimer = setTimeout(() => {
      typingPulseTimer = null;
      if (!isDrawing) applySpeedToVoice(0, 0);
    }, durMs);
  }

  async function prepareCrumple(): Promise<void> {
    if (muted) return;
    ensureContext();
    if (!crumbleRawBytes) await preloadCrumple();
    if (actx && crumbleRawBytes && !crumbleBuffer) await decodeCrumple();
    if (actx && actx.state === 'suspended') {
      try {
        await Promise.race([actx.resume(), new Promise(r => setTimeout(r, 120))]);
      } catch {
        // Ignore resume rejection
      }
    }
  }

  function stopCrumple(): void {
    if (activeCrumpleSource) {
      try {
        activeCrumpleSource.stop();
      } catch {
        // Already stopped
      }
      activeCrumpleSource = null;
    }
    if (fallbackCrumpleEl) {
      try {
        fallbackCrumpleEl.pause();
        fallbackCrumpleEl.currentTime = 0;
      } catch {
        // Ignore fallback error
      }
    }
  }

  function playCrumple(): void {
    stopCrumple();
    if (muted) return;
    logEv('crumple', 'crumble');
    const ctx = ensureContext();
    if (ctx && crumbleBuffer && masterGain) {
      const src = ctx.createBufferSource();
      src.buffer = crumbleBuffer;
      const g = ctx.createGain();
      g.gain.value = CRUMPLE_VOL * 3.4;
      src.connect(g);
      g.connect(masterGain);
      src.onended = () => {
        if (activeCrumpleSource === src) activeCrumpleSource = null;
      };
      activeCrumpleSource = src;
      src.start(ctx.currentTime);
      return;
    }
    if (fallbackCrumpleEl) {
      fallbackCrumpleEl.currentTime = 0;
      fallbackCrumpleEl.volume = CRUMPLE_VOL;
      fallbackCrumpleEl.play().catch(() => {});
    }
  }

  function prefetchCompanionClips(): void {
    for (const [id, spec] of Object.entries(COMPANION_CLIPS) as [CompanionClipId, ClipSpec][]) {
      fetch(spec.file)
        .then(res => (res.ok ? res.arrayBuffer() : null))
        .then(ab => {
          if (!ab) return;
          clipRaw.set(id, ab);
          loadedCompanionCount++;
        })
        .catch(() => {});
    }
  }

  async function getCompanionBuffer(id: CompanionClipId): Promise<AudioBuffer | null> {
    const cached = clipBuf.get(id);
    if (cached) return cached;
    const spec = COMPANION_CLIPS[id];
    const ctx = ensureContext();
    if (!ctx) return null;
    if (!clipRaw.has(id)) {
      try {
        const res = await fetch(spec.file);
        if (!res.ok) return null;
        clipRaw.set(id, await res.arrayBuffer());
        loadedCompanionCount++;
      } catch {
        return null;
      }
    }
    const raw = clipRaw.get(id);
    if (!raw) return null;
    try {
      const dec = await ctx.decodeAudioData(raw.slice(0));
      clipBuf.set(id, dec);
      return dec;
    } catch {
      return null;
    }
  }

  function stopCompanion(): void {
    if (activeVocalSrc) {
      try {
        activeVocalSrc.stop();
      } catch {
        // Ignore
      }
      activeVocalSrc = null;
    }
    for (const s of activeFoleySrcs) {
      try {
        s.stop();
      } catch {
        // Ignore
      }
    }
    activeFoleySrcs.clear();
  }

  async function playClip(
    id: CompanionClipId,
    opts: PlayClipOptions = {},
  ): Promise<{ played: boolean; duration: number }> {
    const spec = COMPANION_CLIPS[id];
    const channel = opts.channel ?? 'vocal';
    const volMul = opts.volumeMul ?? 1.0;
    if (muted) return { played: false, duration: spec.dur };
    if (channel === 'vocal' && activeVocalSrc && !opts.interrupt) {
      return { played: false, duration: spec.dur };
    }
    logEv('clip', id);
    const ctx = ensureContext();
    if (!ctx || !masterGain) return { played: false, duration: spec.dur };
    const buf = await getCompanionBuffer(id);
    if (!buf || muted) return { played: false, duration: spec.dur };

    if (channel === 'vocal' && activeVocalSrc) {
      try {
        activeVocalSrc.stop();
      } catch {
        // Ignore
      }
      activeVocalSrc = null;
    }

    const src = ctx.createBufferSource();
    src.buffer = buf;
    const g = ctx.createGain();
    g.gain.value = spec.gain * volMul;
    src.connect(g);
    g.connect(masterGain);

    if (channel === 'vocal') {
      activeVocalSrc = src;
      src.onended = () => {
        if (activeVocalSrc === src) activeVocalSrc = null;
      };
    } else {
      activeFoleySrcs.add(src);
      src.onended = () => {
        activeFoleySrcs.delete(src);
      };
    }
    src.start(ctx.currentTime);
    return { played: true, duration: buf.duration };
  }

  function syncUI(): void {
    const tog = document.getElementById('sound-toggle');
    if (!tog) return;
    tog.classList.toggle('muted', muted);
    tog.setAttribute('aria-pressed', muted ? 'false' : 'true');
    tog.title = muted ? 'Sound Off (click to unmute)' : 'Sound On (click to mute)';
    tog.setAttribute('aria-label', muted ? 'Sound Off' : 'Sound On');
  }

  function toggleMute(): void {
    muted = !muted;
    if (masterGain && actx) {
      masterGain.gain.setTargetAtTime(muted ? 0 : 1, actx.currentTime, 0.015);
    }
    if (muted) {
      stopCrumple();
      stopCompanion();
    }
    syncUI();
  }

  function initUI(): void {
    const unlockOnce = () => {
      ensureContext();
      if (crumbleRawBytes && !crumbleBuffer) void decodeCrumple();
    };
    window.addEventListener('pointerdown', unlockOnce, { once: true, passive: true });
    window.addEventListener('keydown', unlockOnce, { once: true, passive: true });

    void preloadCrumple();
    setTimeout(prefetchCompanionClips, 250);
    syncUI();
  }

  function setPreset(id: PenId): void {
    activePresetId = id;
    logEv('pen', id);
    configureVoice();
  }

  return {
    initUI,
    setPreset,
    startDraw,
    updateDraw,
    stopDraw,
    triggerTypingPulse,
    prepareCrumple,
    playCrumple,
    stopCrumple,
    playClip,
    stopCompanion,
    isVocalBusy: () => !!activeVocalSrc,
    toggleMute,
    setEventLogger: fn => {
      eventLogger = fn;
    },
    getState: () => ({
      muted,
      presetId: activePresetId,
      hasCrumpleBuffer: !!crumbleBuffer,
      companionLoaded: loadedCompanionCount,
      companionTotal: Object.keys(COMPANION_CLIPS).length,
    }),
  };
}
