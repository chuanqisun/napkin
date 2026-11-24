import fs from 'node:fs';
import path from 'node:path';

// Load .env cleanly without leaking keys
const env = {};
if (fs.existsSync('.env')) {
  for (const line of fs.readFileSync('.env', 'utf8').split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Za-z0-9_]+)\s*=\s*(.*)\s*$/);
    if (m) env[m[1]] = m[2].replace(/^['"]|['"]$/g, '');
  }
}

const EL_KEY = env.ELEVENLABS_API_KEY || process.env.ELEVENLABS_API_KEY;
const GEMINI_KEY = env.GEMINI_API_KEY || process.env.GEMINI_API_KEY;
const VOICE_ID = 'LRBMsOzRzI614MxPFGIY';

if (!EL_KEY) throw new Error('Missing ELEVENLABS_API_KEY');
if (!GEMINI_KEY) throw new Error('Missing GEMINI_API_KEY');

const VO_DIR = path.resolve('trailer/audio/vo');
const SFX_DIR = path.resolve('trailer/audio/sfx');
const MUSIC_DIR = path.resolve('trailer/audio/music');
fs.mkdirSync(VO_DIR, { recursive: true });
fs.mkdirSync(SFX_DIR, { recursive: true });
fs.mkdirSync(MUSIC_DIR, { recursive: true });

// Convert ElevenLabs character alignment to crisp word-level timestamps!
function charsToWords(alignment) {
  if (!alignment || !alignment.characters) return [];
  const { characters, character_start_times_seconds: starts, character_end_times_seconds: ends } = alignment;
  const words = [];
  let cur = '';
  let wStart = null;
  let wEnd = null;
  for (let i = 0; i < characters.length; i++) {
    const ch = characters[i];
    if (/\s/.test(ch)) {
      if (cur.length > 0) {
        words.push({ word: cur, start: +wStart.toFixed(4), end: +wEnd.toFixed(4) });
        cur = '';
        wStart = null;
        wEnd = null;
      }
    } else {
      if (wStart === null) wStart = starts[i];
      wEnd = ends[i];
      cur += ch;
    }
  }
  if (cur.length > 0) {
    words.push({ word: cur, start: +wStart.toFixed(4), end: +wEnd.toFixed(4) });
  }
  return words;
}

const VO_LINES = [
  { id: 'vo_01', text: 'Billion-dollar idea. Step one: ridiculously fancy pen.' },
  { id: 'vo_02', text: 'Step two: caffeine.' },
  { id: 'vo_03', text: 'Step three... ruin it.' },
  { id: 'vo_04', text: "Good thing there's a whole stack." },
  { id: 'vo_05', text: 'There it is.' },
  { id: 'vo_06', text: 'Napkin. Play the free beta now.' },
];

const SFX_PROMPTS = [
  {
    id: 'fountain_scratch',
    duration: 1.8,
    prompt: 'Close-up ASMR foley of a fountain pen nib sketching briskly on thick textured kraft paper napkin, crisp tactile paper grain, clean dry studio recording, no music',
  },
  {
    id: 'marker_sketch',
    duration: 1.4,
    prompt: 'Close-up ASMR foley of a bold felt-tip marker drawing quick confident strokes on thick paper napkin, subtle marker squeak and damp paper texture, clean studio recording, no music',
  },
  {
    id: 'ui_click',
    duration: 0.5,
    prompt: 'Single subtle minimal tactile wooden UI click tap, crisp clean soft button press, studio foley',
  },
  {
    id: 'coffee_splat',
    duration: 0.8,
    prompt: 'Single comedic wet coffee drop splattering onto a thick paper napkin, quick wet liquid plop and blot, crisp dry foley, no music',
  },
  {
    id: 'zoom_whoosh',
    duration: 0.5,
    prompt: 'Short snappy subtle modern camera whip-pan air whoosh swish, clean crisp light transition sound effect',
  },
  {
    id: 'eureka_chime',
    duration: 1.4,
    prompt: 'Warm cozy triumphant acoustic Rhodes electric piano and soft marimba major ninth chord sparkle accent, classy eureka lightbulb moment, clean studio',
  },
];

async function generateVO(item) {
  const mp3Path = path.join(VO_DIR, `${item.id}.mp3`);
  const jsonPath = path.join(VO_DIR, `${item.id}.json`);
  const r = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}/with-timestamps?output_format=mp3_44100_128`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': EL_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: item.text,
        model_id: 'eleven_v4',
        voice_settings: {
          stability: 0.45,
          similarity_boost: 0.80,
          style: 0.35,
          use_speaker_boost: true,
        },
      }),
    },
  );
  if (!r.ok) {
    throw new Error(`VO ${item.id} failed (${r.status}): ${await r.text()}`);
  }
  const data = await r.json();
  const audioBuf = Buffer.from(data.audio_base64, 'base64');
  fs.writeFileSync(mp3Path, audioBuf);
  const alignment = data.normalized_alignment || data.alignment;
  const words = charsToWords(alignment);
  const duration = words.length ? words[words.length - 1].end : 0;
  const meta = { id: item.id, text: item.text, model: 'eleven_v4', voice_id: VOICE_ID, duration, words };
  fs.writeFileSync(jsonPath, JSON.stringify(meta, null, 2));
  console.log(`[VO] ${item.id}: "${item.text}" -> ${duration.toFixed(2)}s (${words.length} words)`);
  return meta;
}

async function generateSFX(item) {
  const mp3Path = path.join(SFX_DIR, `${item.id}.mp3`);
  const r = await fetch('https://api.elevenlabs.io/v1/sound-generation', {
    method: 'POST',
    headers: {
      'xi-api-key': EL_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: item.prompt,
      duration_seconds: item.duration,
      prompt_influence: 0.45,
    }),
  });
  if (!r.ok) {
    throw new Error(`SFX ${item.id} failed (${r.status}): ${await r.text()}`);
  }
  const buf = Buffer.from(await r.arrayBuffer());
  fs.writeFileSync(mp3Path, buf);
  console.log(`[SFX] ${item.id}.mp3 (${buf.byteLength} bytes)`);
}

async function generateLyriaMusic() {
  const outPath = path.join(MUSIC_DIR, 'lyria_cafe_bossa.mp3');
  const prompt =
    'Instrumental cozy upbeat Parisian café bossa nova and playful light acoustic jazz, 118 BPM, warm plucked nylon acoustic guitar, gentle upright walking bass, soft brushed jazz snare and light shaker, subtle warm Rhodes chords, sparse airy open arrangement leaving plenty of room for voiceover and sound effects, cheerful witty indie game vibe, strictly instrumental, no vocals, no singing.';
  const r = await fetch('https://generativelanguage.googleapis.com/v1beta/interactions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Api-Revision': '2026-05-20',
      'x-goog-api-key': GEMINI_KEY,
    },
    body: JSON.stringify({
      model: 'lyria-3-clip-preview',
      input: prompt,
    }),
  });
  if (!r.ok) {
    throw new Error(`Lyria music failed (${r.status}): ${await r.text()}`);
  }
  const data = await r.json();
  // Locate audio output across interactions schema variants
  let b64 = null;
  let mime = 'audio/mpeg';
  const scan = (obj) => {
    if (!obj || typeof obj !== 'object' || b64) return;
    if (obj.type === 'audio' && typeof obj.data === 'string' && obj.data.length > 1000) {
      b64 = obj.data;
      mime = obj.mime_type || mime;
      return;
    }
    if (obj.inlineData?.data && /audio/i.test(obj.inlineData.mimeType || '')) {
      b64 = obj.inlineData.data;
      mime = obj.inlineData.mimeType;
      return;
    }
    if (Array.isArray(obj)) {
      for (const el of obj) scan(el);
    } else {
      for (const v of Object.values(obj)) scan(v);
    }
  };
  scan(data);
  if (!b64) {
    fs.writeFileSync(path.join(MUSIC_DIR, 'lyria_debug.json'), JSON.stringify(data, null, 2));
    throw new Error('Lyria response contained no audio payload (saved lyria_debug.json)');
  }
  const ext = mime.includes('wav') ? 'wav' : 'mp3';
  const finalPath = path.join(MUSIC_DIR, `lyria_cafe_bossa.${ext}`);
  const buf = Buffer.from(b64, 'base64');
  fs.writeFileSync(finalPath, buf);
  console.log(`[MUSIC] Lyria 3 clip saved: ${finalPath} (${buf.byteLength} bytes, ${mime})`);
  return finalPath;
}

async function runPool(items, limit, fn) {
  const results = new Array(items.length);
  let nextIdx = 0;
  async function worker() {
    while (nextIdx < items.length) {
      const i = nextIdx++;
      results[i] = await fn(items[i]);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

async function main() {
  console.log('Generating ElevenLabs v4 custom VO, ElevenLabs SFX (max 3 concurrent), and Gemini Lyria 3 music...');
  const elPromise = (async () => {
    const voMetas = await runPool(VO_LINES, 3, generateVO);
    await runPool(SFX_PROMPTS, 3, generateSFX);
    return voMetas;
  })();
  const musicPromise = generateLyriaMusic();

  const [voMetas, musicPath] = await Promise.all([elPromise, musicPromise]);
  fs.writeFileSync(
    path.join(VO_DIR, 'vo_manifest.json'),
    JSON.stringify({ voice_id: VOICE_ID, model: 'eleven_v4', lines: voMetas, musicPath }, null, 2),
  );
  console.log('ALL V2 AUDIO ASSETS GENERATED SUCCESSFULLY!');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
