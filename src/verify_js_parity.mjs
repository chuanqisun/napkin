/**
 * Verify that pure-JS inference for both exported models (GestureMLP and ConditionalStrokeGRU-MDN)
 * executes cleanly, produces finite outputs, and samples valid vector strokes for every class.
 */
import fs from 'fs';

const mlp = JSON.parse(fs.readFileSync('/home/antigravity/napkin-ml/exports/gesture_mlp.json', 'utf8'));
const gru = JSON.parse(fs.readFileSync('/home/antigravity/napkin-ml/exports/stroke_gru_mdn.json', 'utf8'));

function softmax(arr, temp = 1.0) {
  const scaled = arr.map(v => v / Math.max(temp, 1e-4));
  const m = Math.max(...scaled);
  const exps = scaled.map(v => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / s);
}

function runMLP(x68) {
  let h = x68;
  for (let li = 0; li < mlp.layers.length; li++) {
    const { w, b } = mlp.layers[li];
    const out = new Array(b.length);
    for (let i = 0; i < b.length; i++) {
      let sum = b[i];
      const row = w[i];
      for (let j = 0; j < row.length; j++) sum += row[j] * h[j];
      out[i] = li < mlp.layers.length - 1 ? Math.max(0, sum) : sum;
    }
    h = out;
  }
  return softmax(h);
}

function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function sampleCat(probs) {
  let r = Math.random(), acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r <= acc) return i;
  }
  return probs.length - 1;
}

function sampleGRU(clsIdx, temperature = 0.35) {
  const H = gru.hidden_size;
  const M = gru.num_mixtures;
  const W = gru.weights;
  const emb = W.class_emb[clsIdx];
  let h = new Array(H).fill(0);
  let x = [0, 0, 0, 1, 0];
  let cx = 0, cy = 0;
  const strokes = [];
  let cur = [];

  for (let step = 0; step < gru.max_seq_len; step++) {
    const inp = [...x, ...emb];
    const r = new Array(H), z = new Array(H), n = new Array(H), hNext = new Array(H);

    // Precompute ih and hh gates
    for (let i = 0; i < H; i++) {
      let ir = W.gru_b_ih[i], hr = W.gru_b_hh[i];
      let iz = W.gru_b_ih[H + i], hz = W.gru_b_hh[H + i];
      const w_ir = W.gru_w_ih[i], w_hr = W.gru_w_hh[i];
      const w_iz = W.gru_w_ih[H + i], w_hz = W.gru_w_hh[H + i];
      for (let j = 0; j < inp.length; j++) { ir += w_ir[j] * inp[j]; iz += w_iz[j] * inp[j]; }
      for (let j = 0; j < H; j++) { hr += w_hr[j] * h[j]; hz += w_hz[j] * h[j]; }
      r[i] = 1 / (1 + Math.exp(-(ir + hr)));
      z[i] = 1 / (1 + Math.exp(-(iz + hz)));
    }
    for (let i = 0; i < H; i++) {
      let in_v = W.gru_b_ih[2 * H + i], hn_v = W.gru_b_hh[2 * H + i];
      const w_in = W.gru_w_ih[2 * H + i], w_hn = W.gru_w_hh[2 * H + i];
      for (let j = 0; j < inp.length; j++) in_v += w_in[j] * inp[j];
      for (let j = 0; j < H; j++) hn_v += w_hn[j] * h[j];
      n[i] = Math.tanh(in_v + r[i] * hn_v);
      hNext[i] = (1 - z[i]) * n[i] + z[i] * h[i];
    }
    h = hNext;

    // Linear head -> 6*M + 3
    const p = new Array(W.head_b.length);
    for (let i = 0; i < p.length; i++) {
      let sum = W.head_b[i];
      const row = W.head_w[i];
      for (let j = 0; j < H; j++) sum += row[j] * h[j];
      p[i] = sum;
    }

    const pi = softmax(p.slice(0, M), temperature);
    const penProbs = softmax(p.slice(6 * M, 6 * M + 3), Math.max(temperature, 0.85));
    let penState = sampleCat(penProbs);
    if (penState === 2 && strokes.length + (cur.length >= 2 ? 1 : 0) >= 1) break;
    if (penState === 2) penState = 0;

    const k = sampleCat(pi);
    const mx = p[M + k], my = p[2 * M + k];
    const sx = (Math.exp(Math.max(-6, Math.min(2, p[3 * M + k]))) + 1e-3) * Math.sqrt(temperature);
    const sy = (Math.exp(Math.max(-6, Math.min(2, p[4 * M + k]))) + 1e-3) * Math.sqrt(temperature);
    const rho = Math.tanh(p[5 * M + k]) * 0.95;

    const z1 = randn(), z2 = randn();
    const dx = mx + sx * z1;
    const dy = my + sy * (rho * z1 + Math.sqrt(Math.max(1e-4, 1 - rho * rho)) * z2);
    cx += dx; cy += dy;

    if (penState === 1) {
      if (cur.length >= 2) strokes.push(cur);
      cur = [[cx, cy]];
    } else {
      if (cur.length === 0) cur = [[cx - dx, cy - dy]];
      cur.push([cx, cy]);
    }
    x = [dx, dy, penState === 0 ? 1 : 0, penState === 1 ? 1 : 0, 0];
  }
  if (cur.length >= 2) strokes.push(cur);
  return strokes;
}

console.log('MLP test check:', runMLP(new Array(68).fill(0)).map(v => +v.toFixed(3)));
for (let c = 0; c < gru.classes.length; c++) {
  const s = sampleGRU(c, 0.3);
  const pts = s.reduce((a, st) => a + st.length, 0);
  console.log(`GRU JS sample [${gru.classes[c]}]: ${s.length} strokes, ${pts} pts`);
}
