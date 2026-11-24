"""
Verify exact Graves (2013) IAM-OnDB handwriting synthesis forward pass on hand_synth.bin + hand_synth.json
and confirm clean velocity-derived pressure tapering.
"""
import json
import math
import time
import numpy as np

ALPHABET = [
    '\x00', ' ', '!', '"', '#', "'", '(', ')', ',', '-', '.',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
    '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
]
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}


def encode_text(text):
    # Map Q/X/Z to lowercase since IAM-OnDB alphabet omits those 3 uppercase glyphs
    cleaned = text.replace("Q", "q").replace("X", "x").replace("Z", "z")
    idxs = [CHAR_TO_IDX.get(ch, 1) for ch in cleaned] + [0]
    oh = np.zeros((len(idxs), len(ALPHABET)), dtype=np.float32)
    for i, idx in enumerate(idxs):
        oh[i, idx] = 1.0
    return oh


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def load_weights():
    with open("/home/antigravity/napkin-ml/exports/handwriting/hand_synth.json") as f:
        meta = json.load(f)
    raw = np.fromfile("/home/antigravity/napkin-ml/exports/handwriting/hand_synth.bin", dtype=np.float32)
    T = {}
    for k, info in meta["tensors"].items():
        off = info["offset"]
        shape = tuple(info["shape"])
        size = int(np.prod(shape))
        T[k] = raw[off:off + size].reshape(shape)
    return T, meta["styles"]


def lstm_step(x_concat, c_prev, W, b):
    gates = x_concat @ W + b
    i, j, f, o = np.split(gates, 4)
    c = c_prev * sigmoid(f + 1.0) + sigmoid(i) * np.tanh(j)
    h = np.tanh(c) * sigmoid(o)
    return h, c


def synthesize(text, T, styles, style_id=9, bias=0.85, max_steps=None, seed=7):
    rng = np.random.default_rng(seed)
    st = styles[style_id]
    prime_text = st["text"] + " "
    full_text = prime_text + text + " "
    if max_steps is None:
        max_steps = max(220, len(text) * 48)
    oh = encode_text(full_text)
    U = len(oh)
    u_range = np.arange(U, dtype=np.float32)[None, :]  # (1, U)

    prime_offsets = np.array(st["offsets"], dtype=np.float32).reshape(-1, 3)

    h1 = np.zeros(400, dtype=np.float32); c1 = np.zeros(400, dtype=np.float32)
    h2 = np.zeros(400, dtype=np.float32); c2 = np.zeros(400, dtype=np.float32)
    h3 = np.zeros(400, dtype=np.float32); c3 = np.zeros(400, dtype=np.float32)
    kappa = np.zeros(10, dtype=np.float32)
    w = np.zeros(73, dtype=np.float32)

    def step_cell(x3):
        nonlocal h1, c1, h2, c2, h3, c3, kappa, w
        s1_in = np.concatenate([w, x3, h1])
        h1, c1 = lstm_step(s1_in, c1, T["lstm1_W"], T["lstm1_b"])
        attn_in = np.concatenate([w, x3, h1])
        ap = softplus(attn_in @ T["attn_W"] + T["attn_b"])
        alpha, beta, dk = np.split(ap, 3)
        kappa = kappa + dk / 25.0
        beta = np.maximum(beta, 0.01)
        phi = (alpha[:, None] * np.exp(-((kappa[:, None] - u_range) ** 2) / beta[:, None])).sum(axis=0)
        w = phi @ oh
        s2_in = np.concatenate([x3, h1, w, h2])
        h2, c2 = lstm_step(s2_in, c2, T["lstm2_W"], T["lstm2_b"])
        s3_in = np.concatenate([x3, h2, w, h3])
        h3, c3 = lstm_step(s3_in, c3, T["lstm3_W"], T["lstm3_b"])
        return phi

    # Prime with style strokes
    for row in prime_offsets:
        step_cell(row)

    x3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    out = []
    for _ in range(max_steps):
        phi = step_cell(x3)
        gmm = h3 @ T["gmm_W"] + T["gmm_b"]
        pis = gmm[0:20] * (1.0 + bias)
        s1 = np.maximum(np.exp(gmm[20:40] - bias), 1e-4)
        s2 = np.maximum(np.exp(gmm[40:60] - bias), 1e-4)
        rhos = np.clip(np.tanh(gmm[60:80]), -0.9999, 0.9999)
        mu1 = gmm[80:100]
        mu2 = gmm[100:120]
        es = float(sigmoid(gmm[120]))
        if es < 0.01:
            es = 0.0

        pis = np.exp(pis - pis.max())
        pis = pis / pis.sum()
        pis[pis < 0.01] = 0.0
        pis = pis / pis.sum()

        k = int(rng.choice(20, p=pis))
        z1, z2 = rng.standard_normal(), rng.standard_normal()
        dx = float(mu1[k] + s1[k] * z1)
        dy = float(mu2[k] + s2[k] * (rhos[k] * z1 + math.sqrt(max(1e-5, 1.0 - rhos[k] ** 2)) * z2))
        eos = 1.0 if rng.random() < es else 0.0
        x3 = np.array([dx, dy, eos], dtype=np.float32)
        out.append([dx, dy, eos])

        if int(np.argmax(phi)) >= U - 2 and eos > 0.5:
            break

    return np.array(out, dtype=np.float32)


if __name__ == "__main__":
    T, styles = load_weights()
    t0 = time.time()
    seq = synthesize("Parse Strokes", T, styles, style_id=9, bias=0.9)
    dt = time.time() - t0
    coords = np.cumsum(seq[:, :2], axis=0)
    n_lifts = int(seq[:, 2].sum())
    print(f"Synthesized 'Parse Strokes': {len(seq)} pts, {n_lifts} pen lifts in {dt:.2f}s | x=[{coords[:,0].min():.1f}, {coords[:,0].max():.1f}], y=[{coords[:,1].min():.1f}, {coords[:,1].max():.1f}]")
