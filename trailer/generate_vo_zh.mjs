import fs from 'node:fs';
import path from 'node:path';

const env = {};
if (fs.existsSync('.env')) {
  for (const line of fs.readFileSync('.env', 'utf8').split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Za-z0-9_]+)\s*=\s*(.*)\s*$/);
    if (m) env[m[1]] = m[2].replace(/^['"]|['"]$/g, '');
  }
}

const EL_KEY = env.ELEVENLABS_API_KEY || process.env.ELEVENLABS_API_KEY;
const VOICE_ID = 'LRBMsOzRzI614MxPFGIY';
if (!EL_KEY) throw new Error('Missing ELEVENLABS_API_KEY');

const VO_DIR = path.resolve('trailer/audio/vo_zh');
fs.mkdirSync(VO_DIR, { recursive: true });

// Keep character-level timestamps for exact Chinese karaoke highlighting!
function extractChars(alignment) {
  if (!alignment || !alignment.characters) return [];
  const { characters, character_start_times_seconds: starts, character_end_times_seconds: ends } = alignment;
  const out = [];
  for (let i = 0; i < characters.length; i++) {
    out.push({
      ch: characters[i],
      start: +starts[i].toFixed(4),
      end: +ends[i].toFixed(4),
    });
  }
  return out;
}

const VO_LINES_ZH = [
  { id: 'vo_01a', text: '值一个亿的灵感。' },
  { id: 'vo_01b', text: '第一步，先挑支有排面的笔。' },
  { id: 'vo_02', text: '第二步，咖啡续命。' },
  { id: 'vo_03', text: '第三步……完美翻车。' },
  { id: 'vo_04', text: '还好，纸巾管够。' },
  { id: 'vo_05', text: '对，就是它！' },
  { id: 'vo_06', text: 'Napkin，免费公测中，现在就能玩。' },
];

async function generateOne(item) {
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
        language_code: 'zh',
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
    throw new Error(`ZH VO ${item.id} failed (${r.status}): ${await r.text()}`);
  }
  const data = await r.json();
  fs.writeFileSync(mp3Path, Buffer.from(data.audio_base64, 'base64'));
  const alignment = data.normalized_alignment || data.alignment;
  const chars = extractChars(alignment);
  const duration = chars.length ? chars[chars.length - 1].end : 0;
  const meta = { id: item.id, text: item.text, model: 'eleven_v4', voice_id: VOICE_ID, language: 'zh', duration, chars };
  fs.writeFileSync(jsonPath, JSON.stringify(meta, null, 2));
  console.log(`[VO-ZH] ${item.id}: "${item.text}" -> ${duration.toFixed(2)}s (${chars.length} chars)`);
  return meta;
}

async function runPool(items, limit, fn) {
  const results = new Array(items.length);
  let idx = 0;
  async function worker() {
    while (idx < items.length) {
      const i = idx++;
      results[i] = await fn(items[i]);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

async function main() {
  const metas = await runPool(VO_LINES_ZH, 3, generateOne);
  fs.writeFileSync(
    path.join(VO_DIR, 'vo_manifest_zh.json'),
    JSON.stringify({ voice_id: VOICE_ID, model: 'eleven_v4', language: 'zh', lines: metas }, null, 2),
  );
  console.log('ALL CHINESE VO LINES GENERATED!');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
