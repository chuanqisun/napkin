import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { setTimeout as realSleep } from 'node:timers/promises';

const CHROME_BIN = '/home/stack/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome';
const PORT = 5179;
const CDP_PORT = 9229;
const VIEW_W = 720;
const VIEW_H = 1280;
const DPR = 2; // 1440x2560 raw capture!
const FPS = 30;
const DT_MS = 1000 / FPS;

const RAW_DIR = path.resolve('trailer/raw');
fs.mkdirSync(RAW_DIR, { recursive: true });

// Virtual clock script injected BEFORE page scripts execute
const VCLOCK_INIT_SCRIPT = `
(() => {
  const realPerfNow = performance.now.bind(performance);
  const realDateNow = Date.now.bind(Date);
  const realRAF = window.requestAnimationFrame.bind(window);
  const realCAF = window.cancelAnimationFrame.bind(window);
  const realSetTimeout = window.setTimeout.bind(window);
  const realClearTimeout = window.clearTimeout.bind(window);
  const realSetInterval = window.setInterval.bind(window);
  const realClearInterval = window.clearInterval.bind(window);

  let enabled = false;
  let vNow = 0;
  let dateBase = realDateNow();
  let nextRafId = 1;
  let nextTimerId = 1;
  let rafQueue = new Map();
  let timerQueue = new Map();

  function getNow() {
    return enabled ? vNow : realPerfNow();
  }

  Object.defineProperty(performance, 'now', {
    configurable: true,
    value: () => getNow(),
  });
  Date.now = () => (enabled ? Math.floor(dateBase + vNow) : realDateNow());

  window.requestAnimationFrame = (cb) => {
    if (!enabled) return realRAF(cb);
    const id = nextRafId++;
    rafQueue.set(id, cb);
    return id;
  };
  window.cancelAnimationFrame = (id) => {
    if (!enabled) return realCAF(id);
    rafQueue.delete(id);
  };
  window.setTimeout = (fn, ms = 0, ...args) => {
    if (!enabled) return realSetTimeout(fn, ms, ...args);
    const id = nextTimerId++;
    timerQueue.set(id, {
      fn: typeof fn === 'function' ? () => fn(...args) : () => eval(fn),
      due: vNow + Math.max(0, Number(ms) || 0),
      interval: null,
    });
    return id;
  };
  window.clearTimeout = (id) => {
    if (!enabled) return realClearTimeout(id);
    timerQueue.delete(id);
  };
  window.setInterval = (fn, ms = 0, ...args) => {
    if (!enabled) return realSetInterval(fn, ms, ...args);
    const id = nextTimerId++;
    const iv = Math.max(1, Number(ms) || 1);
    timerQueue.set(id, {
      fn: typeof fn === 'function' ? () => fn(...args) : () => eval(fn),
      due: vNow + iv,
      interval: iv,
    });
    return id;
  };
  window.clearInterval = (id) => {
    if (!enabled) return realClearInterval(id);
    timerQueue.delete(id);
  };

  window.__vclock = {
    enable() {
      if (enabled) return;
      vNow = realPerfNow();
      dateBase = realDateNow() - vNow;
      enabled = true;
    },
    now() {
      return getNow();
    },
    step(dtMs) {
      if (!enabled) return;
      const target = vNow + dtMs;
      // Fire due timers in chronological order
      let safety = 0;
      while (safety++ < 120) {
        let earliestId = null;
        let earliestDue = target + 1e-6;
        for (const [id, t] of timerQueue.entries()) {
          if (t.due <= earliestDue) {
            earliestDue = t.due;
            earliestId = id;
          }
        }
        if (earliestId === null || earliestDue > target) break;
        const entry = timerQueue.get(earliestId);
        if (!entry) break;
        vNow = entry.due;
        if (entry.interval !== null) {
          entry.due = vNow + entry.interval;
        } else {
          timerQueue.delete(earliestId);
        }
        try { entry.fn(); } catch (err) { console.error('vclock timer err:', err); }
      }
      vNow = target;
      // Run one batch of rAF callbacks
      const batch = Array.from(rafQueue.entries());
      rafQueue.clear();
      for (const [, cb] of batch) {
        try { cb(vNow); } catch (err) { console.error('vclock raf err:', err); }
      }
      return vNow;
    },
  };
})();
`;

class CDPClient {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.id = 0;
    this.pending = new Map();
    this.events = new Map();
  }
  async connect() {
    this.ws = new WebSocket(this.wsUrl);
    await new Promise((resolve, reject) => {
      this.ws.addEventListener('open', resolve, { once: true });
      this.ws.addEventListener('error', reject, { once: true });
    });
    this.ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(`${msg.error.code}: ${msg.error.message}`));
        else resolve(msg.result);
      } else if (msg.method && this.events.has(msg.method)) {
        for (const fn of this.events.get(msg.method)) fn(msg.params);
      }
    });
  }
  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  async eval(expression) {
    const res = await this.send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (res.exceptionDetails) {
      throw new Error(`JS exception: ${res.exceptionDetails.text} ${JSON.stringify(res.exceptionDetails.exception)}`);
    }
    return res.result?.value;
  }
  close() {
    try { this.ws?.close(); } catch {}
  }
}

async function waitForHttp(url, timeoutMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const r = await fetch(url);
      if (r.ok) return true;
    } catch {}
    await realSleep(150);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

class TakeRecorder {
  constructor(cdp, name) {
    this.cdp = cdp;
    this.name = name;
    this.mp4Path = path.join(RAW_DIR, `${name}.mp4`);
    this.metaPath = path.join(RAW_DIR, `${name}.json`);
    this.frames = [];
    this.markers = [];
    this.frameIdx = 0;
    this.vStart = 0;
    this.curX = 360;
    this.curY = 640;
    this.curMode = 'arrow';
    this.curDown = false;
  }

  async start() {
    this.vStart = await this.cdp.eval(`window.__vclock.now()`);
    await this.cdp.eval(`window.__napkinTrailer.clearAudioLog()`);
    this.ff = spawn('ffmpeg', [
      '-y',
      '-f', 'image2pipe',
      '-vcodec', 'mjpeg',
      '-r', String(FPS),
      '-i', '-',
      '-c:v', 'libx264',
      '-preset', 'fast',
      '-crf', '16',
      '-pix_fmt', 'yuv420p',
      '-movflags', '+faststart',
      this.mp4Path,
    ], { stdio: ['pipe', 'ignore', 'pipe'] });
    this.ffErr = '';
    this.ff.stderr.on('data', d => { this.ffErr += d.toString(); });
  }

  mark(label, extra = {}) {
    this.markers.push({
      label,
      frame: this.frameIdx,
      t: +(this.frameIdx / FPS).toFixed(4),
      ...extra,
    });
  }

  async setCursor(x, y, mode = this.curMode, down = this.curDown) {
    this.curX = x;
    this.curY = y;
    this.curMode = mode;
    this.curDown = down;
    await this.cdp.eval(`window.__napkinTrailer.setCursor(${x.toFixed(1)}, ${y.toFixed(1)}, ${JSON.stringify(mode)}, ${down})`);
  }

  async captureFrame(workerYieldMs = 0) {
    if (workerYieldMs > 0) await realSleep(workerYieldMs);
    await this.cdp.eval(`window.__vclock.step(${DT_MS.toFixed(4)})`);
    const { data } = await this.cdp.send('Page.captureScreenshot', {
      format: 'jpeg',
      quality: 94,
    });
    const buf = Buffer.from(data, 'base64');
    if (!this.ff.stdin.write(buf)) {
      await new Promise(r => this.ff.stdin.once('drain', r));
    }
    this.frames.push({
      f: this.frameIdx,
      t: +(this.frameIdx / FPS).toFixed(4),
      cx: +this.curX.toFixed(1),
      cy: +this.curY.toFixed(1),
      mode: this.curMode,
      down: this.curDown,
    });
    this.frameIdx++;
  }

  async hold(ms, workerYieldMs = 0) {
    const n = Math.max(1, Math.round(ms / DT_MS));
    for (let i = 0; i < n; i++) await this.captureFrame(workerYieldMs);
  }

  async moveTo(tx, ty, ms, mode = this.curMode) {
    const sx = this.curX;
    const sy = this.curY;
    const n = Math.max(2, Math.round(ms / DT_MS));
    // Gentle quadratic curve arc so cursor feels human, not robotic
    const mx = (sx + tx) * 0.5 + (ty - sy) * 0.08;
    const my = (sy + ty) * 0.5 - (tx - sx) * 0.05;
    for (let i = 1; i <= n; i++) {
      const u = i / n;
      // Smooth ease-in-out cubic
      const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
      const omt = 1 - e;
      const x = omt * omt * sx + 2 * omt * e * mx + e * e * tx;
      const y = omt * omt * sy + 2 * omt * e * my + e * e * ty;
      await this.setCursor(x, y, mode, this.curDown);
      await this.cdp.send('Input.dispatchMouseEvent', {
        type: 'mouseMoved',
        x,
        y,
        buttons: this.curDown ? 1 : 0,
      });
      await this.captureFrame();
    }
  }

  async clickSelector(sel, moveMs = 260, mode = 'hand', postPauseMs = 120) {
    const rect = await this.cdp.eval(`(() => {
      const el = document.querySelector(${JSON.stringify(sel)});
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return { x: r.left + r.width * 0.5, y: r.top + r.height * 0.5, w: r.width, h: r.height };
    })()`);
    if (!rect) throw new Error(`Element not found: ${sel}`);
    await this.moveTo(rect.x, rect.y, moveMs, mode);
    await this.setCursor(rect.x, rect.y, mode, true);
    await this.cdp.send('Input.dispatchMouseEvent', {
      type: 'mousePressed',
      x: rect.x,
      y: rect.y,
      button: 'left',
      clickCount: 1,
    });
    await this.captureFrame();
    await this.setCursor(rect.x, rect.y, mode, false);
    await this.cdp.send('Input.dispatchMouseEvent', {
      type: 'mouseReleased',
      x: rect.x,
      y: rect.y,
      button: 'left',
      clickCount: 1,
    });
    await this.hold(postPauseMs);
    return rect;
  }

  async clickPoint(x, y, moveMs = 220, mode = 'cross', postPauseMs = 80) {
    await this.moveTo(x, y, moveMs, mode);
    await this.setCursor(x, y, mode, true);
    await this.cdp.send('Input.dispatchMouseEvent', {
      type: 'mousePressed',
      x,
      y,
      button: 'left',
      clickCount: 1,
    });
    await this.captureFrame();
    await this.setCursor(x, y, mode, false);
    await this.cdp.send('Input.dispatchMouseEvent', {
      type: 'mouseReleased',
      x,
      y,
      button: 'left',
      clickCount: 1,
    });
    await this.hold(postPauseMs);
  }

  async strokePath(pts, durMs, mode = 'cross') {
    if (pts.length < 2) return;
    const p0 = pts[0];
    await this.moveTo(p0[0], p0[1], 140, mode);
    await this.setCursor(p0[0], p0[1], mode, true);
    await this.cdp.send('Input.dispatchMouseEvent', {
      type: 'mousePressed',
      x: p0[0],
      y: p0[1],
      button: 'left',
      clickCount: 1,
      buttons: 1,
    });
    await this.captureFrame();

    const nFrames = Math.max(3, Math.round(durMs / DT_MS));
    for (let f = 1; f <= nFrames; f++) {
      const u = f / nFrames;
      const idxF = u * (pts.length - 1);
      const i0 = Math.floor(idxF);
      const i1 = Math.min(pts.length - 1, i0 + 1);
      const frac = idxF - i0;
      const x = pts[i0][0] + (pts[i1][0] - pts[i0][0]) * frac;
      const y = pts[i0][1] + (pts[i1][1] - pts[i0][1]) * frac;
      await this.setCursor(x, y, mode, true);
      await this.cdp.send('Input.dispatchMouseEvent', {
        type: 'mouseMoved',
        x,
        y,
        button: 'left',
        buttons: 1,
      });
      await this.captureFrame();
    }

    const last = pts[pts.length - 1];
    await this.setCursor(last[0], last[1], mode, false);
    await this.cdp.send('Input.dispatchMouseEvent', {
      type: 'mouseReleased',
      x: last[0],
      y: last[1],
      button: 'left',
      clickCount: 1,
      buttons: 0,
    });
    await this.captureFrame();
  }

  async dragLinear(x1, y1, x2, y2, durMs, mode = 'cross') {
    const pts = [];
    const steps = 16;
    for (let i = 0; i <= steps; i++) {
      const u = i / steps;
      const e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
      pts.push([x1 + (x2 - x1) * e, y1 + (y2 - y1) * e]);
    }
    await this.strokePath(pts, durMs, mode);
  }

  async typeLive(text, charFrames = 2) {
    for (const ch of text) {
      await this.cdp.send('Input.insertText', { text: ch });
      for (let k = 0; k < charFrames; k++) {
        await this.captureFrame(22);
      }
    }
    // Let streaming handwriting worker finish cleanly while capturing frames
    for (let i = 0; i < 40; i++) {
      const idle = await this.cdp.eval(`window.__napkinTrailer.isHandwritingIdle()`);
      await this.captureFrame(18);
      if (idle && i >= 2) break;
    }
    // Commit via Enter
    await this.cdp.send('Input.dispatchKeyEvent', {
      type: 'keyDown',
      key: 'Enter',
      code: 'Enter',
      windowsVirtualKeyCode: 13,
    });
    await this.cdp.send('Input.dispatchKeyEvent', {
      type: 'keyUp',
      key: 'Enter',
      code: 'Enter',
      windowsVirtualKeyCode: 13,
    });
    await this.hold(80);
  }

  async finish() {
    const rawAudio = await this.cdp.eval(`window.__napkinTrailer.getAudioLog()`);
    const audioEvents = rawAudio.map(ev => ({
      ...ev,
      t: +Math.max(0, (ev.t - this.vStart) / 1000).toFixed(4),
    }));
    const anchors = await this.cdp.eval(`({
      canvas: window.__napkinTrailer.getCanvasRect(),
      items: window.__napkinTrailer.getItemRects(),
      cup: window.__napkinTrailer.cup.getCupScreenPos(),
      cupState: window.__napkinTrailer.cup.getState(),
    })`);

    this.ff.stdin.end();
    await new Promise((resolve, reject) => {
      this.ff.on('close', code => {
        if (code === 0) resolve();
        else reject(new Error(`ffmpeg exited ${code}: ${this.ffErr}`));
      });
    });

    const meta = {
      name: this.name,
      fps: FPS,
      width: VIEW_W * DPR,
      height: VIEW_H * DPR,
      logicalW: VIEW_W,
      logicalH: VIEW_H,
      dpr: DPR,
      frameCount: this.frameIdx,
      durationSec: +(this.frameIdx / FPS).toFixed(4),
      markers: this.markers,
      audioEvents,
      anchors,
      frames: this.frames,
    };
    fs.writeFileSync(this.metaPath, JSON.stringify(meta, null, 2));
    console.log(`[CAPTURED] ${this.name}.mp4 (${meta.frameCount} frames, ${meta.durationSec}s, ${audioEvents.length} audio events)`);
    return meta;
  }
}

// Organic hand-drawn shape point generators (look authentically rough before auto-beautify!)
function makeRoughOval(cx, cy, rx, ry, n = 42) {
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const a = -Math.PI * 0.55 + (i / n) * Math.PI * 2.04;
    const wob = 1 + 0.045 * Math.sin(a * 3 + 0.7) + 0.03 * Math.cos(a * 5);
    pts.push([cx + Math.cos(a) * rx * wob, cy + Math.sin(a) * ry * wob]);
  }
  return pts;
}

function makeRoughBox(x, y, w, h) {
  const pts = [];
  const seg = (x1, y1, x2, y2, k = 10) => {
    for (let i = 0; i <= k; i++) {
      const u = i / k;
      const jx = Math.sin(u * Math.PI) * 3.2;
      const jy = Math.sin(u * Math.PI * 2) * 2.4;
      pts.push([x1 + (x2 - x1) * u + jx, y1 + (y2 - y1) * u + jy]);
    }
  };
  seg(x, y, x + w, y);
  seg(x + w, y, x + w, y + h);
  seg(x + w, y + h, x, y + h);
  seg(x, y + h, x, y + 3);
  return pts;
}

function makeDownArrow(x, y1, y2) {
  const shaft = [];
  for (let i = 0; i <= 12; i++) {
    const u = i / 12;
    shaft.push([x + Math.sin(u * Math.PI) * 2.5, y1 + (y2 - y1) * u]);
  }
  const head = [
    [x - 16, y2 - 18],
    [x, y2 + 2],
    [x + 17, y2 - 17],
  ];
  return { shaft, head };
}

async function main() {
  console.log('Starting Vite dev server...');
  const vite = spawn('npx', ['vite', '--port', String(PORT), '--strictPort', '--host', '127.0.0.1'], {
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let chrome = null;
  let cdp = null;

  try {
    await waitForHttp(`http://127.0.0.1:${PORT}/napkin/`);
    console.log('Vite ready. Launching headless Chromium...');

    chrome = spawn(CHROME_BIN, [
      '--headless=new',
      `--remote-debugging-port=${CDP_PORT}`,
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--autoplay-policy=no-user-gesture-required',
      `--window-size=${VIEW_W},${VIEW_H}`,
      '--hide-scrollbars',
      'about:blank',
    ], { stdio: ['ignore', 'ignore', 'pipe'] });

    await waitForHttp(`http://127.0.0.1:${CDP_PORT}/json/list`);
    const targets = await (await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`)).json();
    const pageTarget = targets.find(t => t.type === 'page');
    if (!pageTarget?.webSocketDebuggerUrl) throw new Error('No page target found');

    cdp = new CDPClient(pageTarget.webSocketDebuggerUrl);
    await cdp.connect();
    await cdp.send('Page.enable');
    await cdp.send('Runtime.enable');
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width: VIEW_W,
      height: VIEW_H,
      deviceScaleFactor: DPR,
      mobile: false,
    });
    await cdp.send('Page.addScriptToEvaluateOnNewDocument', { source: VCLOCK_INIT_SCRIPT });
    await cdp.send('Page.navigate', { url: `http://127.0.0.1:${PORT}/napkin/?trailer=1` });

    // Wait for app + VAE + WASM handwriting + cup sprites + audio buffers
    for (let i = 0; i < 80; i++) {
      const ready = await cdp.eval(`(() => {
        const t = window.__napkinTrailer;
        if (!t) return false;
        return t.cup.getState().loadedCups >= 16 && t.isHandwritingIdle();
      })()`);
      if (ready) break;
      await realSleep(150);
    }
    // Pre-warm audio & crumple buffer before virtual clock starts
    await cdp.eval(`window.__napkinTrailer.audio.prepareCrumple()`);
    await realSleep(600);
    // Warm up handwriting worker pool with a tiny invisible check so first real keystroke is fast
    await cdp.eval(`window.__vclock.enable()`);
    console.log('App loaded & virtual clock enabled. Recording 5 raw 9:16 takes...');

    // =========================================================================
    // TAKE 1 (v3): Draw "Big Idea" ("Billion-dollar idea.") -> OPEN PEN MENU ("Step one: ridiculously fancy pen.") -> Arrow + "Magic"
    // =========================================================================
    const only45 = process.argv.includes('--only45');
    let m1, m2, m3;
    let lastCurX = 360, lastCurY = 640;

    if (!only45) {
    const t1 = new TakeRecorder(cdp, '01_pen_and_idea');
    await t1.setCursor(335, 360, 'cross', false);
    await t1.start();
    t1.mark('start');

    // Phase 1A ("Billion-dollar idea."): Freehand rough oval -> SNAP -> type "Big Idea"
    const ovalPts = makeRoughOval(335, 420, 140, 58, 34);
    t1.mark('draw_oval_start');
    await t1.strokePath(ovalPts, 420, 'cross');
    await t1.hold(650); // auto-beautify snap!
    t1.mark('oval_snapped');

    await t1.clickPoint(335, 420, 100, 'cross', 35);
    t1.mark('type_big_idea');
    await t1.typeLive('Big Idea', 2);
    await t1.hold(120);
    t1.mark('big_idea_done');

    // Phase 1B ("Step one: ridiculously fancy pen."): Open Pen & Writing Style menu -> dramatic Fountain Pen pick!
    await t1.clickSelector('#pen-btn', 200, 'hand', 180);
    t1.mark('pen_menu_open');

    // Snobbish micro-hover on Ballpoint -> glide down past Pencil & Marker -> click gold Fountain Pen!
    const bp = await cdp.eval(`(() => { const r = document.querySelector('button.pen-opt[data-pen="ballpoint"]').getBoundingClientRect(); return { x: r.left + r.width*0.62, y: r.top + r.height*0.5 }; })()`);
    await t1.moveTo(bp.x, bp.y, 180, 'hand');
    await t1.hold(160);
    const ftn = await t1.clickSelector('button.pen-opt[data-pen="fountain"]', 260, 'hand', 380);
    t1.mark('fountain_selected', { x: ftn.x, y: ftn.y });

    await t1.clickSelector('button.style-opt[data-style="cursive"]', 200, 'hand', 160);
    await t1.clickSelector('#pen-close-btn', 150, 'hand', 60);
    t1.mark('pen_menu_closed');

    // Phase 1C: Arrow down + Box -> SNAP -> freehand-draw a bold "?" question mark inside the box!
    await t1.clickSelector('#t-arrow', 140, 'hand', 35);
    await t1.dragLinear(335, 488, 335, 578, 200, 'cross');
    t1.mark('arrow_drawn');

    await t1.clickSelector('#t-sketch', 130, 'hand', 35);
    const boxPts = makeRoughBox(190, 592, 290, 114);
    await t1.strokePath(boxPts, 420, 'cross');
    await t1.hold(650);
    t1.mark('magic_box_snapped');

    // Freehand-draw a crisp, expressive "?" question mark right inside the rectangle box!
    t1.mark('draw_question_mark');
    const qHook = [
      [318, 626], [321, 616], [330, 610], [342, 610], [352, 616],
      [355, 626], [351, 636], [341, 643], [335, 650], [335, 661],
    ];
    await t1.strokePath(qHook, 260, 'cross');
    // Expressive dot beneath the hook (tiny closed dab so it reads crisp in Fountain Pen ink)
    const qDot = [
      [333, 678], [337, 678], [337, 683], [333, 683], [334, 680],
    ];
    await t1.strokePath(qDot, 90, 'cross');
    await cdp.eval(`window.__napkinTrailer.flushPendingNow()`);
    t1.mark('question_mark_done');
    await t1.hold(200);

    m1 = await t1.finish();

    // =========================================================================
    // TAKE 2: Writer's Block -> Order Coffee Latte -> Sip -> SPLAT -> Cup Ring
    // =========================================================================
    const t2 = new TakeRecorder(cdp, '02_block_order_spill');
    t2.curX = t1.curX;
    t2.curY = t1.curY;
    t2.curMode = 'cross';
    await t2.start();
    t2.mark('start');

    // Hesitant squiggle below Magic -> Undo -> weary sigh!
    const squiggle = [];
    for (let i = 0; i <= 20; i++) {
      const u = i / 20;
      squiggle.push([250 + u * 165, 775 + Math.sin(u * Math.PI * 5) * 26]);
    }
    await t2.strokePath(squiggle, 360, 'cross');
    await t2.hold(180);
    await t2.clickSelector('#undo-btn', 180, 'hand', 120);
    t2.mark('undo_block');
    await cdp.eval(`window.__napkinTrailer.audio.playClip('sigh_block', { channel: 'vocal', interrupt: true })`);
    await t2.hold(420);

    // Order Drink: click pulsing amber Menu button -> Coffee Latte!
    const menuBtn = await t2.clickSelector('#drink-menu-btn', 200, 'hand', 200);
    t2.mark('drink_menu_open', { x: menuBtn.x, y: menuBtn.y });

    const latteBtn = await t2.clickSelector('button.drink-opt[data-drink="coffee_latte"]', 240, 'hand', 80);
    t2.mark('latte_ordered', { x: latteBtn.x, y: latteBtn.y });

    // Wait for latte mug to glide onto upper-right napkin & clink!
    await t2.hold(680);
    t2.mark('latte_landed');

    // Configure deterministic SPLAT right across the "?" box!
    const magicCenter = await cdp.eval(`(() => {
      const r = window.__napkinTrailer.getCanvasRect();
      const shapes = window.__napkinTrailer.getItemRects().filter(x => x.kind === 'shape');
      const it = shapes[shapes.length - 1];
      return {
        canvasX: it ? (it.cx - r.left) : 335,
        canvasY: it ? (it.cy - r.top) : 649,
        screenX: it ? it.cx : 335,
        screenY: it ? it.cy : 649,
      };
    })()`);
    await cdp.eval(`(() => {
      window.__napkinTrailer.cup.setTrailerDripOverride({ x: ${magicCenter.canvasX - 24}, y: ${magicCenter.canvasY - 6}, kind: 'trail', scale: 1.7 });
      window.__napkinTrailer.cup.setTrailerForceRing(true);
    })()`);

    // Click Coffee Latte mug to take a sip!
    const cupPos = await cdp.eval(`window.__napkinTrailer.cup.getCupScreenPos()`);
    await t2.clickPoint(cupPos.bowlX, cupPos.bowlY, 240, 'hand', 40);
    t2.mark('sip_clicked', { cupX: cupPos.bowlX, cupY: cupPos.bowlY });

    // Exiting (420ms) + Drinking until SPLAT (~52% of 2150ms = ~1120ms) -> SPLAT lands at +1540ms!
    await t2.hold(1500);
    t2.mark('splat_hit', { x: magicCenter.screenX, y: magicCenter.screenY });

    // Hold during glistening spread + "Oops!" + mug returning to corner (~1600ms)
    await t2.hold(1750);
    t2.mark('mug_returned');

    // Slowly drag & swivel the mug down-right using Mickey-glove GRAB fist to reveal a dark coffee ring!
    const cupPos2 = await cdp.eval(`window.__napkinTrailer.cup.getCupScreenPos()`);
    await t2.moveTo(cupPos2.bowlX, cupPos2.bowlY, 240, 'hand');
    await t2.setCursor(cupPos2.bowlX, cupPos2.bowlY, 'grab', true);
    await cdp.send('Input.dispatchMouseEvent', {
      type: 'mousePressed',
      x: cupPos2.bowlX,
      y: cupPos2.bowlY,
      button: 'left',
      clickCount: 1,
      buttons: 1,
    });
    await t2.hold(220); // hold threshold lifts mug
    const dragSteps = 16;
    const dx = -12;
    const dy = 210;
    for (let i = 1; i <= dragSteps; i++) {
      const u = i / dragSteps;
      const e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
      const nx = cupPos2.bowlX + dx * e;
      const ny = cupPos2.bowlY + dy * e;
      await t2.setCursor(nx, ny, 'grab', true);
      await cdp.send('Input.dispatchMouseEvent', {
        type: 'mouseMoved',
        x: nx,
        y: ny,
        button: 'left',
        buttons: 1,
      });
      await t2.captureFrame();
    }
    const endX = cupPos2.bowlX + dx;
    const endY = cupPos2.bowlY + dy;
    await t2.setCursor(endX, endY, 'hand', false);
    await cdp.send('Input.dispatchMouseEvent', {
      type: 'mouseReleased',
      x: endX,
      y: endY,
      button: 'left',
      clickCount: 1,
      buttons: 0,
    });
    t2.mark('coffee_ring_revealed', { x: cupPos2.bowlX, y: cupPos2.bowlY });
    await t2.hold(900);

    m2 = await t2.finish();

    if (process.argv.includes('--only12')) {
      console.log('Re-captured Takes 1 & 2 cleanly!');
      return;
    }

    // =========================================================================
    // TAKE 3 (v3): Trash an Idea + 3x RAPID-FIRE "WHOLE STACK" CRUMPLE MONTAGE!
    // =========================================================================
    const t3 = new TakeRecorder(cdp, '03_crumple_reset');
    t3.curX = t2.curX;
    t3.curY = t2.curY;
    t3.curMode = 'hand';
    await cdp.eval(`(() => {
      const t = window.__napkinTrailer;
      t.queueNapkin({ paletteIdx: 2, embossStyle: 'linen', borderStyle: 'double', cupRing: null });   // Toasted Oat
      t.queueNapkin({ paletteIdx: 5, embossStyle: 'diamond', borderStyle: 'dotted', cupRing: null }); // Raw Sand
      t.queueNapkin({ paletteIdx: 0, embossStyle: 'linen', borderStyle: 'double', cupRing: null });   // Warm Kraft
      t.queueNapkin({ paletteIdx: 4, embossStyle: 'dimple', borderStyle: 'stitch', cupRing: null });  // Final clean Diner Ivory
    })()`);
    await t3.start();
    t3.mark('start');

    // Crumple #1 (Hero real-time crumple of the coffee-stained Big Idea ↓ Magic sheet!)
    await t3.hold(180);
    const trashBtn = await t3.clickSelector('#clear-btn', 220, 'hand', 40);
    t3.mark('trash_clicked', { x: trashBtn.x, y: trashBtn.y });
    await t3.hold(430);
    t3.mark('crumple_start');
    await t3.hold(1450);
    t3.mark('crumple_done');
    await t3.hold(480); // mug returns cleanly onto Toasted Oat

    // Rapid-fire Stack Crumples #2, #3, #4 (will be sped up ~3.4x during "Good thing there's a whole stack."!)
    t3.mark('stack_start');
    const badDoodles = [
      { color: '#1b3d8f', pen: 'ballpoint', strokes: [[[170, 420], [500, 470], [180, 560], [510, 620]], [[210, 680], [460, 400]]] },
      { color: '#b91c1c', pen: 'marker', strokes: [[[220, 460], [450, 460], [450, 660], [220, 660], [220, 460]], [[180, 430], [490, 690]], [[490, 430], [180, 690]]] },
      { color: '#1c1e22', pen: 'fountain', strokes: [[[180, 520], [260, 410], [340, 610], [420, 410], [500, 580]]] },
    ];
    for (let si = 0; si < badDoodles.length; si++) {
      const d = badDoodles[si];
      await cdp.eval(`window.__napkinTrailer.setTool('sketch'); window.__napkinTrailer.setPen(${JSON.stringify(d.pen)}); window.__napkinTrailer.setColor(${JSON.stringify(d.color)});`);
      for (const pts of d.strokes) {
        await t3.strokePath(pts, 180, 'cross');
      }
      await cdp.eval(`window.__napkinTrailer.flushPendingNow()`);
      await t3.hold(80);
      await t3.clickSelector('#clear-btn', 120, 'hand', 30);
      t3.mark(`stack_crumple_${si + 2}_start`);
      await t3.hold(430 + 1450 + 480);
      t3.mark(`stack_crumple_${si + 2}_done`);
    }
    await t3.hold(320);

    m3 = await t3.finish();
    if (process.argv.includes('--only123')) {
      console.log('Re-captured Takes 1, 2, 3 cleanly!');
      return;
    }
    lastCurX = t3.curX;
    lastCurY = t3.curY;
    } else {
      m1 = JSON.parse(fs.readFileSync(path.join(RAW_DIR, '01_pen_and_idea.json'), 'utf8'));
      m2 = JSON.parse(fs.readFileSync(path.join(RAW_DIR, '02_block_order_spill.json'), 'utf8'));
      m3 = JSON.parse(fs.readFileSync(path.join(RAW_DIR, '03_crumple_reset.json'), 'utf8'));
      await cdp.eval(`(() => {
        const t = window.__napkinTrailer;
        t.setNapkin({ paletteIdx: 4, embossStyle: 'dimple', borderStyle: 'stitch', cupRing: null });
        t.cup.orderDrink('coffee_latte');
        t.cup.setTrailerCupPos(0.86, 0.22, 22);
      })()`);
      for (let i = 0; i < 20; i++) await cdp.eval(`window.__vclock.step(35)`);
    }

    // =========================================================================
    // TAKE 4: Matcha Swap + Red Marker Vertical "Weird Thought ↓ Napkin ↓ Ship It!" + EUREKA!
    // =========================================================================
    const t4 = new TakeRecorder(cdp, '04_matcha_eureka');
    t4.curX = lastCurX;
    t4.curY = lastCurY;
    t4.curMode = 'hand';
    await cdp.eval(`window.__napkinTrailer.cup.setTrailerForceRing(false)`);
    await cdp.eval(`window.__napkinTrailer.cup.setTrailerCupPos(0.88, 0.16, 22)`);
    await t4.start();
    t4.mark('start');

    // Quick swap to Ceremonial Matcha!
    await t4.clickSelector('#drink-menu-btn', 160, 'hand', 120);
    await t4.clickSelector('button.drink-opt[data-drink="matcha_pure"]', 180, 'hand', 60);
    t4.mark('matcha_ordered');
    await t4.hold(860); // replacing_out (350ms) + entering (480ms)
    t4.mark('matcha_landed');

    // Switch to Ballpoint + Clean Upright for crisp legible top items (placed cleanly clear of Matcha!)
    await cdp.eval(`window.__napkinTrailer.setPen('ballpoint'); window.__napkinTrailer.setWritingStyle('upright'); window.__napkinTrailer.setColor('#1b3d8f');`);

    // 1. Oval: "Weird Thought" (shifted cleanly below/left of Matcha rim!)
    await t4.clickSelector('#t-oval', 130, 'hand', 30);
    await t4.dragLinear(125, 365, 445, 470, 220, 'cross');
    await t4.clickPoint(285, 418, 90, 'cross', 30);
    t4.mark('type_weird_thought');
    await t4.typeLive('Weird Thought', 2);

    // 2. Arrow down
    await t4.clickSelector('#t-arrow', 110, 'hand', 30);
    await t4.dragLinear(285, 478, 285, 555, 160, 'cross');

    // 3. Box: "Napkin"
    await t4.clickSelector('#t-box', 110, 'hand', 30);
    await t4.dragLinear(155, 565, 415, 670, 210, 'cross');
    await t4.clickPoint(285, 618, 90, 'cross', 30);
    t4.mark('type_napkin');
    await t4.typeLive('Napkin', 2);

    // 4. Switch to Red Ink + Felt-Tip Marker (Clean Upright style so "Ship It!" reads crystal clear!)
    await t4.clickSelector('button.swatch[data-color="#b91c1c"]', 140, 'hand', 40);
    await cdp.eval(`window.__napkinTrailer.setPen('marker'); window.__napkinTrailer.setWritingStyle('upright');`);

    // Bold Red Arrow down + "Aha!"
    await t4.clickSelector('#t-arrow', 100, 'hand', 30);
    await t4.dragLinear(285, 678, 315, 760, 160, 'cross');
    await cdp.eval(`window.__napkinTrailer.audio.playClip('aha_spark', { channel: 'vocal', interrupt: true })`);
    t4.mark('aha_spark');

    // 5. Big bold Red Marker Box: "Ship It!"
    await t4.clickSelector('#t-box', 100, 'hand', 30);
    await t4.dragLinear(145, 770, 495, 905, 220, 'cross');
    await t4.clickPoint(320, 838, 90, 'cross', 30);
    t4.mark('type_ship_it');
    await t4.typeLive('Ship It!', 2);

    // 6. Switch to Sketch and slash TWO energetic red underlines beneath Ship It!
    await t4.clickSelector('#t-sketch', 110, 'hand', 20);
    const u1 = [];
    const u2 = [];
    for (let i = 0; i <= 14; i++) {
      const u = i / 14;
      u1.push([165 + u * 310, 932 - Math.sin(u * Math.PI) * 6]);
      u2.push([185 + u * 275, 952 - Math.sin(u * Math.PI) * 5]);
    }
    await t4.strokePath(u1, 160, 'cross');
    await t4.strokePath(u2, 150, 'cross');
    await cdp.eval(`window.__napkinTrailer.flushPendingNow()`);

    // EUREKA!!
    await cdp.eval(`window.__napkinTrailer.audio.playClip('eureka_win', { channel: 'vocal', interrupt: true })`);
    t4.mark('eureka_hit', { x: 320, y: 845 });
    await t4.hold(2100);

    const m4 = await t4.finish();

    // =========================================================================
    // TAKE 5: Clean Full-Bleed Sunlit Napkin End Card (Live-Writing "Napkin")
    // =========================================================================
    const t5 = new TakeRecorder(cdp, '05_endcard_napkin');
    await cdp.eval(`(() => {
      const t = window.__napkinTrailer;
      t.setHideToolbar(true);
      t.clearInstant();
      t.setNapkin({
        paletteIdx: 0,
        embossStyle: 'linen',
        borderStyle: 'double',
        cupRing: { nx: 0.16, ny: 0.78, radius: 52, alpha: 0.042, rot: 1.1 },
      });
      t.cup.orderDrink('coffee_latte');
      t.cup.setTrailerCupPos(0.84, 0.17, 24);
      t.setPen('fountain');
      t.setWritingStyle('upright');
      t.setColor('#1b3d8f');
      t.setCursor(-100, -100, 'hidden', false);
    })()`);
    // Let latte settle instantly into idle before end card starts
    for (let i = 0; i < 20; i++) await cdp.eval(`window.__vclock.step(35)`);

    await t5.start();
    t5.mark('start');
    await t5.hold(160);

    // Kick off live streaming cursive "Napkin" in big gorgeous ink right in center safe zone!
    t5.mark('write_title_start');
    const titlePromise = cdp.eval(`window.__napkinTrailer.typeTextDirect('Napkin', 360, 560, 128, 480, '#1d3557')`);
    // Step frames while the neural cursive streams stroke-by-stroke!
    for (let i = 0; i < 42; i++) {
      await t5.captureFrame(24);
      const idle = await cdp.eval(`window.__napkinTrailer.isHandwritingIdle()`);
      if (idle && i >= 18) break;
    }
    await titlePromise;
    t5.mark('write_title_done');

    // Add a graceful sweeping red underline swoosh directly beneath "Napkin"
    await cdp.eval(`window.__napkinTrailer.setColor('#b91c1c'); window.__napkinTrailer.setTool('sketch');`);
    const swoosh = [];
    for (let i = 0; i <= 18; i++) {
      const u = i / 18;
      swoosh.push([185 + u * 350, 646 - Math.sin(u * Math.PI) * 11]);
    }
    await t5.strokePath(swoosh, 240, 'hidden');
    await cdp.eval(`window.__napkinTrailer.flushPendingNow(); window.__napkinTrailer.setCursor(-100, -100, 'hidden', false);`);
    t5.mark('swoosh_done');
    await t5.hold(2200);

    const m5 = await t5.finish();

    // Save combined capture manifest
    fs.writeFileSync(
      path.join(RAW_DIR, 'manifest.json'),
      JSON.stringify({ fps: FPS, width: VIEW_W * DPR, height: VIEW_H * DPR, takes: [m1, m2, m3, m4, m5] }, null, 2),
    );
    console.log('ALL 5 RAW 9:16 TAKES CAPTURED SUCCESSFULLY!');
  } finally {
    cdp?.close();
    try { chrome?.kill('SIGKILL'); } catch {}
    try { vite?.kill('SIGKILL'); } catch {}
    process.exit(0);
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
