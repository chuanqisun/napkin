import ballpointImg from '../../docs/vfx/pens/ballpoint.webp?url';
import fountainImg from '../../docs/vfx/pens/fountain.webp?url';
import markerImg from '../../docs/vfx/pens/marker.webp?url';
import mechImg from '../../docs/vfx/pens/mech.webp?url';
import pencilImg from '../../docs/vfx/pens/pencil.webp?url';
import vaeModelUrl from '../../exports/opensketch_residual_vae.json?url';
import gravesBinUrl from '../../exports/handwriting/graves_v5.bin?url';
import gravesMetaUrl from '../../exports/handwriting/graves_v5_meta.json?url';
import gravesWasmUrl from '../../exports/handwriting/graves_v5.wasm?url';
import crumbleMp3Url from '../../web/crumble.mp3?url';
import type { CompanionClipId, DrinkId, PenId, PenTypeSpec, VolumeKey } from './types';

const cupGlob = import.meta.glob<string>('../../docs/vfx/cups/*.webp', {
  eager: true,
  query: '?url',
  import: 'default',
});

const sfxGlob = import.meta.glob<string>('../../web/sfx/*.mp3', {
  eager: true,
  query: '?url',
  import: 'default',
});

function extractStem(map: Record<string, string>, ext: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [path, url] of Object.entries(map)) {
    const file = path.slice(path.lastIndexOf('/') + 1);
    const stem = file.endsWith(ext) ? file.slice(0, -ext.length) : file;
    out[stem] = url;
  }
  return out;
}

const CUP_URLS = extractStem(cupGlob, '.webp');
const SFX_URLS = extractStem(sfxGlob, '.mp3');

export const PEN_IMAGES = {
  ballpoint: ballpointImg,
  pencil: pencilImg,
  marker: markerImg,
  fountain: fountainImg,
  mech: mechImg,
} as const satisfies Record<PenId, string>;

export const PEN_TYPES = [
  { id: 'ballpoint', name: 'Ballpoint Pen', short: 'Ballpoint', hint: 'Smooth crisp roll', img: PEN_IMAGES.ballpoint },
  { id: 'pencil', name: '2B Graphite Pencil', short: '2B Pencil', hint: 'Soft dry paper tooth', img: PEN_IMAGES.pencil },
  { id: 'marker', name: 'Felt-Tip Marker', short: 'Marker', hint: 'Bold translucent ink', img: PEN_IMAGES.marker },
  { id: 'fountain', name: 'Fountain Pen', short: 'Fountain', hint: 'Thick/thin calligraphy nib', img: PEN_IMAGES.fountain },
  { id: 'mech', name: '0.5mm Mechanical', short: '0.5mm', hint: 'Fine precision lead', img: PEN_IMAGES.mech },
] as const satisfies readonly PenTypeSpec[];

export function getCupSpriteUrl(drinkId: DrinkId, volumeKey: VolumeKey): string {
  return CUP_URLS[`${drinkId}_${volumeKey}`] ?? '';
}

export function getSfxUrl(clipId: CompanionClipId): string {
  return SFX_URLS[clipId] ?? '';
}

export const MODEL_URLS = {
  vae: vaeModelUrl,
  gravesMeta: gravesMetaUrl,
  gravesWasm: gravesWasmUrl,
  gravesBin: gravesBinUrl,
  crumbleMp3: crumbleMp3Url,
} as const;

const CACHE_NAME = 'napkin-ml-v7-2';

export async function cachedFetchBytes(url: string): Promise<ArrayBuffer> {
  let cache: Cache | null = null;
  try {
    if ('caches' in window) cache = await caches.open(CACHE_NAME);
  } catch {
    // Ignore storage quota or private-mode cache errors
  }
  if (cache) {
    const hit = await cache.match(url);
    if (hit) return await hit.arrayBuffer();
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  const buf = await res.arrayBuffer();
  if (cache) {
    try {
      await cache.put(
        url,
        new Response(buf.slice(0), {
          headers: { 'Content-Type': res.headers.get('Content-Type') || 'application/octet-stream' },
        }),
      );
    } catch {
      // Ignore cache write errors
    }
  }
  return buf;
}

export async function cachedFetchJSON<T>(url: string): Promise<T> {
  const buf = await cachedFetchBytes(url);
  return JSON.parse(new TextDecoder().decode(buf)) as T;
}
