"""
Train Idea 2 Model: 32-Point Semantic Operator Classifier MLP (ml5.neuralNetwork / TF.js compatible).
Classes (5 semantic transformation symbols):
  0: emphasize (real Quick, Draw! 'circle' strokes)
  1: extract   (real Quick, Draw! 'triangle' + inverted V-funnel strokes)
  2: summarize (humanized '}' right-bracket strokes with natural warp & speed jitter)
  3: compare   (real Quick, Draw! 'line' strokes)
  4: itemize   (real Quick, Draw! 'zigzag' strokes)

Input features (68 floats, identical in Python & JS):
  - 32 equidistant resampled normalized (x, y) coords -> 64 floats
  - [log_aspect_ratio, start_end_dist, path_length_norm, bbox_area_norm] -> 4 floats
"""
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = "/home/antigravity/napkin-ml/data"
EXPORT_DIR = "/home/antigravity/napkin-ml/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

OP_CLASSES = ["emphasize", "extract", "summarize", "compare", "itemize"]
N_PTS = 32
EPOCHS = 35
BATCH_SIZE = 128
SEED = 42


def flatten_drawing(drawing):
    pts = []
    for stroke in drawing:
        for x, y in zip(stroke[0], stroke[1]):
            pts.append([float(x), float(y)])
    return pts


def extract_68_features(raw_pts, n=N_PTS):
    """Arc-length resample to n points + normalize + 4 geometric invariants -> (68,) float32."""
    if len(raw_pts) < 2:
        return None
    pts = np.asarray(raw_pts, dtype=np.float32)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    total_len = float(seg_lens.sum())
    if total_len < 1e-3:
        return None

    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    targets = np.linspace(0.0, total_len, n)
    rx = np.interp(targets, cum, pts[:, 0])
    ry = np.interp(targets, cum, pts[:, 1])

    min_x, max_x = float(rx.min()), float(rx.max())
    min_y, max_y = float(ry.min()), float(ry.max())
    w = max(max_x - min_x, 1e-3)
    h = max(max_y - min_y, 1e-3)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    scale = max(w, h) / 2.0

    nx = (rx - cx) / scale
    ny = (ry - cy) / scale

    log_aspect = math.log((w + 1e-3) / (h + 1e-3))
    log_aspect = max(-3.0, min(3.0, log_aspect)) / 3.0
    start_end = float(math.hypot(nx[-1] - nx[0], ny[-1] - ny[0])) / 2.83
    norm_path = min(total_len / (scale * 12.0), 2.0)
    area_norm = (w * h) / (4.0 * scale * scale)

    coords = np.empty((n * 2,), dtype=np.float32)
    coords[0::2] = nx
    coords[1::2] = ny
    feats = np.concatenate([coords, np.array([log_aspect, start_end, norm_path, area_norm], dtype=np.float32)])
    return feats


def synthesize_bracket(rng):
    """Generate realistic human right-bracket '}' strokes (top-down or bottom-up)."""
    h = rng.uniform(60, 140)
    w = rng.uniform(16, 42)
    tip = rng.uniform(0.38, 0.62)
    # Control path for '}'
    pts = [
        (-w * 0.5, -h * 0.5),
        (w * 0.12, -h * 0.44),
        (w * 0.08, -h * (0.5 - tip) - h * 0.09),
        (w * 0.55, (tip - 0.5) * h),
        (w * 0.08, (tip - 0.5) * h + h * 0.09),
        (w * 0.12, h * 0.44),
        (-w * 0.5, h * 0.5),
    ]
    # Interpolate smoothly + add hand wobble + slight rotation
    pts = np.array(pts, dtype=np.float32)
    t_orig = np.linspace(0, 1, len(pts))
    t_dense = np.linspace(0, 1, 28)
    x = np.interp(t_dense, t_orig, pts[:, 0])
    y = np.interp(t_dense, t_orig, pts[:, 1])
    x += rng.normal(0, w * 0.06, size=len(x))
    y += rng.normal(0, h * 0.018, size=len(y))

    ang = rng.uniform(-0.22, 0.22)
    rx = x * math.cos(ang) - y * math.sin(ang)
    ry = x * math.sin(ang) + y * math.cos(ang)
    out = np.stack([rx, ry], axis=1)
    if rng.random() < 0.35:
        out = out[::-1]  # drawn bottom-to-top
    return out.tolist()


def synthesize_v_funnel(rng):
    """Augment 'extract' with open V-funnel strokes alongside real Quick, Draw! triangles."""
    w = rng.uniform(45, 110)
    h = rng.uniform(45, 100)
    pts = np.array([
        [-w * 0.5 + rng.normal(0, 5), -h * 0.5 + rng.normal(0, 5)],
        [rng.normal(0, 6), h * 0.5 + rng.normal(0, 5)],
        [w * 0.5 + rng.normal(0, 5), -h * 0.5 + rng.normal(0, 5)],
    ], dtype=np.float32)
    t_orig = np.linspace(0, 1, 3)
    t_dense = np.linspace(0, 1, 22)
    x = np.interp(t_dense, t_orig, pts[:, 0]) + rng.normal(0, 2.2, 22)
    y = np.interp(t_dense, t_orig, pts[:, 1]) + rng.normal(0, 2.2, 22)
    ang = rng.uniform(-0.25, 0.25)
    rx = x * math.cos(ang) - y * math.sin(ang)
    ry = x * math.sin(ang) + y * math.cos(ang)
    out = np.stack([rx, ry], axis=1)
    if rng.random() < 0.4:
        out = out[::-1]
    return out.tolist()


class GestureMLP(nn.Module):
    def __init__(self, in_dim=68, num_classes=len(OP_CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    torch.set_num_threads(2)

    X_list, y_list = [], []

    def add_qd(cat, label_id, limit=2500):
        with open(os.path.join(DATA_DIR, f"qd_{cat}.json")) as f:
            raw = json.load(f)
        added = 0
        for item in raw:
            pts = flatten_drawing(item["drawing"])
            feat = extract_68_features(pts)
            if feat is not None:
                X_list.append(feat)
                y_list.append(label_id)
                added += 1
                if added >= limit:
                    break
        return added

    c0 = add_qd("circle", 0, 2500)
    c1a = add_qd("triangle", 1, 1500)
    c1b = 0
    for _ in range(1000):
        feat = extract_68_features(synthesize_v_funnel(rng))
        if feat is not None:
            X_list.append(feat); y_list.append(1); c1b += 1
    c2 = 0
    for _ in range(2500):
        feat = extract_68_features(synthesize_bracket(rng))
        if feat is not None:
            X_list.append(feat); y_list.append(2); c2 += 1
    c3 = add_qd("line", 3, 2500)
    c4 = add_qd("zigzag", 4, 2500)

    print(f"[data] emphasize(circle)={c0}, extract(tri+V)={c1a+c1b}, summarize(bracket)={c2}, compare(line)={c3}, itemize(zigzag)={c4}")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    n_tr = int(0.80 * len(X))
    n_va = int(0.10 * len(X))
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_va, y_va = X[n_tr:n_tr + n_va], y[n_tr:n_tr + n_va]
    X_te, y_te = X[n_tr + n_va:], y[n_tr + n_va:]

    tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=BATCH_SIZE, shuffle=True)
    model = GestureMLP()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for bx, by in tr_loader:
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(torch.from_numpy(X_va))
            va_acc = float((va_logits.argmax(1).numpy() == y_va).mean())
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 7 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:2d}/{EPOCHS} | val_acc={va_acc*100:.2f}% | {time.time()-t0:.1f}s")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_pred = model(torch.from_numpy(X_te)).argmax(1).numpy()
    te_acc = float((te_pred == y_te).mean())

    # Confusion matrix & per-class accuracy
    cm = np.zeros((len(OP_CLASSES), len(OP_CLASSES)), dtype=int)
    for t, p in zip(y_te, te_pred):
        cm[t, p] += 1
    per_class = {
        OP_CLASSES[i]: round(float(cm[i, i] / max(1, cm[i].sum())) * 100, 2)
        for i in range(len(OP_CLASSES))
    }
    print(f"[test] Held-out Test Accuracy = {te_acc*100:.2f}% ({n_params:,} params, {time.time()-t0:.1f}s)")
    print("[test] Per-class accuracy (%):", json.dumps(per_class))

    # Export clean JSON weights for pure-JS & ml5/TF.js inference
    def arr(t):
        return np.round(t.detach().cpu().numpy().astype(np.float32), 4).tolist()

    layers = [m for m in model.net if isinstance(m, nn.Linear)]
    payload = {
        "architecture": "GestureMLP-68-128-64-5",
        "classes": OP_CLASSES,
        "input_dim": 68,
        "metrics": {
            "params": n_params,
            "train_samples": int(len(X_tr)),
            "test_samples": int(len(X_te)),
            "test_accuracy": round(te_acc * 100, 2),
            "per_class_accuracy": per_class,
            "confusion_matrix": cm.tolist(),
            "train_seconds": round(time.time() - t0, 1),
        },
        "layers": [
            {"w": arr(l.weight), "b": arr(l.bias)} for l in layers
        ],
    }
    out_json = os.path.join(EXPORT_DIR, "gesture_mlp.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    torch.save({"state_dict": model.state_dict(), "classes": OP_CLASSES, "metrics": payload["metrics"]}, os.path.join(EXPORT_DIR, "gesture_mlp.pt"))
    print(f"[export] browser weights saved -> {out_json} ({os.path.getsize(out_json)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
