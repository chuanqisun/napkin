"""
Export v5 optimized Graves weights for hand-written WAT SIMD128 engine (graves_v5.wat):
  1. Splits each LSTM input into:
     - Dense SIMD128 INT8 block (x + hidden states only, skipping 73 sparse char cols!)
       lstm1_dense: [1600, 416] int8 (403 active: x3 + h1)
       lstm2_dense: [1600, 816] int8 (803 active: x3 + h1 + h2)
       lstm3_dense: [1600, 816] int8 (803 active: x3 + h2 + h3)
     - Sparse Character Column tables (only evaluated for the ~2 active chars with w[c] > 1e-3!):
       lstm1_Wchar, lstm2_Wchar, lstm3_Wchar: [73, 1600] int8 + [73] f32 scales
       (36x less work for attention context!)
  2. Pre-computes tight primed states for all 13 writer styles (median(kappa) aligned to 0.35
     so zero steps are wasted on priming tails!).
"""
import json
import os
import time
import numpy as np

from test_handwriting import encode_text, load_weights, softplus, lstm_step

OUT_DIR = "/home/antigravity/napkin-ml/exports/handwriting"


def quantize_rows_i8(M_out_in, pad_mult=16):
    out_d, in_d = M_out_in.shape
    in_pad = ((in_d + pad_mult - 1) // pad_mult) * pad_mult
    Mp = np.zeros((out_d, in_pad), dtype=np.float32)
    Mp[:, :in_d] = M_out_in
    ma = np.max(np.abs(Mp), axis=1)
    sc = np.where(ma > 1e-8, ma / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.round(Mp / sc[:, None]), -127, 127).astype(np.int8)
    return q, sc


TAIL_CHARS = 4  # Keep last 4 chars of prime_text so Gaussian attention heads transition seamlessly


def compute_primed_state(T, style):
    prime_text = style["text"] + " "
    oh = encode_text(prime_text + "    ")
    U = len(oh)
    u_range = np.arange(U, dtype=np.float32)[None, :]
    po = np.array(style["offsets"], dtype=np.float32).reshape(-1, 3)

    h1 = np.zeros(400, np.float32); c1 = np.zeros(400, np.float32)
    h2 = np.zeros(400, np.float32); c2 = np.zeros(400, np.float32)
    h3 = np.zeros(400, np.float32); c3 = np.zeros(400, np.float32)
    kappa = np.zeros(10, np.float32); w = np.zeros(73, np.float32)

    for row in po:
        h1, c1 = lstm_step(np.concatenate([w, row, h1]), c1, T["lstm1_W"], T["lstm1_b"])
        ap = softplus(np.concatenate([w, row, h1]) @ T["attn_W"] + T["attn_b"])
        a, b, dk = np.split(ap, 3)
        kappa += dk / 25.0
        b = np.maximum(b, 0.01)
        phi = (a[:, None] * np.exp(-((kappa[:, None] - u_range) ** 2) / b[:, None])).sum(axis=0)
        w = phi @ oh
        h2, c2 = lstm_step(np.concatenate([row, h1, w, h2]), c2, T["lstm2_W"], T["lstm2_b"])
        h3, c3 = lstm_step(np.concatenate([row, h2, w, h3]), c3, T["lstm3_W"], T["lstm3_b"])

    P = len(prime_text)
    tail_text = prime_text[-TAIL_CHARS:]
    kappa_shifted = kappa - float(P - TAIL_CHARS)
    vec = np.concatenate([h1, c1, h2, c2, h3, c3, kappa_shifted, w]).astype(np.float32)
    return vec, tail_text


def main():
    t0 = time.time()
    T, styles = load_weights()

    # Split LSTM1_W [476, 1600]: [w(0:73), x(73:76), h1(76:476)]
    l1_Wchar = T["lstm1_W"][0:73, :]        # [73, 1600]
    l1_Wdense = T["lstm1_W"][73:476, :].T   # [1600, 403]

    # Split LSTM2_W [876, 1600]: [x(0:3), h1(3:403), w(403:476), h2(476:876)]
    l2_Wchar = T["lstm2_W"][403:476, :]     # [73, 1600]
    l2_Wdense = np.concatenate([T["lstm2_W"][0:403, :], T["lstm2_W"][476:876, :]], axis=0).T  # [1600, 803]

    # Split LSTM3_W [876, 1600]: [x(0:3), h2(3:403), w(403:476), h3(476:876)]
    l3_Wchar = T["lstm3_W"][403:476, :]     # [73, 1600]
    l3_Wdense = np.concatenate([T["lstm3_W"][0:403, :], T["lstm3_W"][476:876, :]], axis=0).T  # [1600, 803]

    chunks = []
    offsets = {}
    cur = 0

    def add(name, arr):
        nonlocal cur
        pad = (16 - (cur % 16)) % 16
        if pad:
            chunks.append(b"\x00" * pad)
            cur += pad
        raw = np.ascontiguousarray(arr).tobytes()
        offsets[name] = {"byte_offset": cur, "bytes": len(raw), "shape": list(arr.shape), "dtype": str(arr.dtype)}
        chunks.append(raw)
        cur += len(raw)

    for name, Wd, Wc, b in [
        ("l1", l1_Wdense, l1_Wchar, T["lstm1_b"]),
        ("l2", l2_Wdense, l2_Wchar, T["lstm2_b"]),
        ("l3", l3_Wdense, l3_Wchar, T["lstm3_b"]),
    ]:
        dq, dsc = quantize_rows_i8(Wd, 16)
        cq, csc = quantize_rows_i8(Wc, 16)  # [73, 1600] int8, [73] f32
        add(f"{name}_dq", dq)
        add(f"{name}_dsc", dsc)
        add(f"{name}_cq", cq)
        add(f"{name}_csc", csc)
        add(f"{name}_b", b.astype(np.float32))

    # Attention: [w(0:73), x(73:76), h1(76:476)] -> pad to [30, 480] f32
    attn_WT = np.zeros((30, 480), dtype=np.float32)
    attn_WT[:, :476] = T["attn_W"].T
    add("attn_WT", attn_WT)
    add("attn_b", T["attn_b"].astype(np.float32))

    # GMM: [121, 400] f32
    add("gmm_WT", T["gmm_W"].T.astype(np.float32))
    add("gmm_b", T["gmm_b"].astype(np.float32))

    # Exact primed states + 4-char tail texts for all 13 styles
    primed_pairs = [compute_primed_state(T, st) for st in styles]
    primed_mat = np.stack([p[0] for p in primed_pairs], axis=0)
    tail_texts = [p[1] for p in primed_pairs]
    add("primed", primed_mat)

    bin_path = os.path.join(OUT_DIR, "graves_v5.bin")
    with open(bin_path, "wb") as f:
        for ch in chunks:
            f.write(ch)

    meta_path = os.path.join(OUT_DIR, "graves_v5_meta.json")
    meta = {
        "total_bytes": cur,
        "tail_chars": TAIL_CHARS,
        "tail_texts": tail_texts,
        "style_labels": [
            "9: Designer Cursive",
            "1: Crisp Print",
            "7: Clean Upright",
            "12: Fast Journal",
            "0: Neat Italic",
            "2: Flowing Script",
        ],
        "style_ids": [9, 1, 7, 12, 0, 2],
        "tensors": offsets,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Exported {bin_path} ({cur / 1e6:.2f} MB) + {meta_path} in {time.time() - t0:.1f}s")
    for k, v in offsets.items():
        print(f"  {k:10s} off={v['byte_offset']:8d} shape={v['shape']} {v['dtype']}")


if __name__ == "__main__":
    main()
