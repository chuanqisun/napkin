"""
Train Idea 1 Model: Conditional SketchRNN-lite (GRU + Bivariate Gaussian MDN)
on real human vector strokes (Quick, Draw! primitives: square, circle, diamond, triangle, line).

Stroke-5 format per step:
  [dx, dy, p_draw, p_move, p_end]
  - p_draw=1: pen down from previous point to (x+dx, y+dy)
  - p_move=1: pen up (move) to (x+dx, y+dy) without drawing
  - p_end=1:  end of sketch

Exports compact JSON weights for zero-dependency browser inference + PyTorch checkpoint.
"""
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "/home/antigravity/napkin-ml/data"
EXPORT_DIR = "/home/antigravity/napkin-ml/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

CLASSES = ["square", "circle", "diamond", "triangle", "line"]
MAX_SEQ_LEN = 38
HIDDEN_SIZE = 96
CLASS_EMB_DIM = 16
NUM_MIXTURES = 8
EPOCHS = 24
BATCH_SIZE = 128
LR = 3e-3
SEED = 42


def is_clean_primitive(cname, drawing):
    """Filter out non-flowchart doodles (e.g., faceted multi-stroke gemstone diamonds)."""
    n_strokes = len(drawing)
    total_pts = sum(len(s[0]) for s in drawing)
    if cname == "diamond":
        return n_strokes == 1 and 4 <= total_pts <= 26
    if cname == "circle":
        return n_strokes == 1 and 8 <= total_pts <= 36
    if cname == "line":
        return n_strokes == 1 and 2 <= total_pts <= 18
    return n_strokes <= 2 and 4 <= total_pts <= 34


def normalize_and_encode(drawing, max_len=MAX_SEQ_LEN):
    """Convert Quick, Draw! [[[x...],[y...]], ...] to normalized Stroke-5 array (L, 5)."""
    xs, ys = [], []
    for stroke in drawing:
        xs.extend(stroke[0])
        ys.extend(stroke[1])
    if len(xs) < 2:
        return None
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    scale = max(max_x - min_x, max_y - min_y, 1.0) / 2.0  # fit into [-1, 1]

    seq = []
    prev_x, prev_y = 0.0, 0.0
    for stroke in drawing:
        sx_list, sy_list = stroke[0], stroke[1]
        for idx in range(len(sx_list)):
            nx = (sx_list[idx] - cx) / scale
            ny = (sy_list[idx] - cy) / scale
            dx = nx - prev_x
            dy = ny - prev_y
            if idx == 0:
                seq.append([dx, dy, 0.0, 1.0, 0.0])  # pen move to stroke start
            else:
                if abs(dx) < 1e-4 and abs(dy) < 1e-4:
                    continue
                seq.append([dx, dy, 1.0, 0.0, 0.0])  # pen draw segment
            prev_x, prev_y = nx, ny
            if len(seq) >= max_len - 1:
                break
        if len(seq) >= max_len - 1:
            break

    seq.append([0.0, 0.0, 0.0, 0.0, 1.0])  # EOS
    return np.array(seq, dtype=np.float32)


class StrokeDataset(Dataset):
    def __init__(self, samples):
        # samples: list of (class_idx, seq ndarray (L, 5))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cls_id, seq = self.samples[idx]
        L = len(seq)
        # Input starts with SOS [0, 0, 0, 1, 0], shifted right by 1
        inp = np.zeros((MAX_SEQ_LEN, 5), dtype=np.float32)
        tgt = np.zeros((MAX_SEQ_LEN, 5), dtype=np.float32)
        mask = np.zeros((MAX_SEQ_LEN,), dtype=np.float32)

        inp[0] = [0.0, 0.0, 0.0, 1.0, 0.0]
        if L > 1:
            inp[1:L] = seq[:L - 1]
        tgt[:L] = seq
        mask[:L] = 1.0
        # Pad tail with EOS
        if L < MAX_SEQ_LEN:
            inp[L:, 4] = 1.0
            tgt[L:, 4] = 1.0
        return torch.tensor(cls_id, dtype=torch.long), torch.from_numpy(inp), torch.from_numpy(tgt), torch.from_numpy(mask)


class ConditionalStrokeGRU(nn.Module):
    def __init__(self, num_classes=len(CLASSES), emb_dim=CLASS_EMB_DIM, hidden_size=HIDDEN_SIZE, num_mixtures=NUM_MIXTURES):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_mixtures = num_mixtures

        self.class_emb = nn.Embedding(num_classes, emb_dim)
        self.gru = nn.GRU(5 + emb_dim, hidden_size, num_layers=1, batch_first=True)
        # 6 params per Gaussian mixture (pi, mu_x, mu_y, sigma_x, sigma_y, rho) + 3 pen logits
        self.head = nn.Linear(hidden_size, num_mixtures * 6 + 3)

    def forward(self, cls_id, x, h=None):
        # cls_id: (B,), x: (B, T, 5)
        emb = self.class_emb(cls_id).unsqueeze(1).expand(-1, x.size(1), -1)
        inp = torch.cat([x, emb], dim=-1)
        out, h_next = self.gru(inp, h)
        params = self.head(out)
        return params, h_next


def mdn_loss(params, tgt, mask, M=NUM_MIXTURES):
    # params: (B, T, 6*M + 3), tgt: (B, T, 5), mask: (B, T)
    pi_logits = params[..., :M]
    mu_x = params[..., M:2 * M]
    mu_y = params[..., 2 * M:3 * M]
    sig_x = torch.exp(params[..., 3 * M:4 * M].clamp(-6.0, 2.0)) + 1e-3
    sig_y = torch.exp(params[..., 4 * M:5 * M].clamp(-6.0, 2.0)) + 1e-3
    rho = torch.tanh(params[..., 5 * M:6 * M]) * 0.95
    pen_logits = params[..., 6 * M:]

    dx = tgt[..., 0:1]
    dy = tgt[..., 1:2]
    pen_target = tgt[..., 2:].argmax(dim=-1)  # 0=draw, 1=move, 2=end

    # Bivariate normal log-prob
    norm_x = (dx - mu_x) / sig_x
    norm_y = (dy - mu_y) / sig_y
    one_minus_rho2 = (1.0 - rho * rho).clamp(min=1e-4)
    z = norm_x * norm_x + norm_y * norm_y - 2.0 * rho * norm_x * norm_y
    log_norm = -math.log(2.0 * math.pi) - torch.log(sig_x) - torch.log(sig_y) - 0.5 * torch.log(one_minus_rho2)
    log_comp = log_norm - z / (2.0 * one_minus_rho2)
    log_pi = F.log_softmax(pi_logits, dim=-1)
    log_prob = torch.logsumexp(log_pi + log_comp, dim=-1)  # (B, T)

    # Only compute coordinate loss on non-EOS steps
    coord_mask = mask * (1.0 - tgt[..., 4])
    l_coord = -(log_prob * coord_mask).sum() / coord_mask.sum().clamp(min=1.0)

    # Pen-state loss across all valid steps
    l_pen = F.cross_entropy(pen_logits.reshape(-1, 3), pen_target.reshape(-1), reduction="none").reshape_as(mask)
    l_pen = (l_pen * mask).sum() / mask.sum().clamp(min=1.0)

    return l_coord + l_pen, l_coord.item(), l_pen.item()


@torch.no_grad()
def sample_sketch(model, cls_idx, temperature=0.35, max_steps=MAX_SEQ_LEN):
    model.eval()
    x = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)
    cls_t = torch.tensor([cls_idx], dtype=torch.long)
    h = None
    cur_x, cur_y = 0.0, 0.0
    strokes = []
    cur_stroke = []
    M = model.num_mixtures
    pen_temp = max(temperature, 0.85)  # Keep pen-end probability well-calibrated

    for _ in range(max_steps):
        params, h = model(cls_t, x, h)
        p = params[0, 0]
        pi = F.softmax(p[:M] / max(temperature, 0.05), dim=-1).numpy()
        mu_x = p[M:2 * M].numpy()
        mu_y = p[2 * M:3 * M].numpy()
        sig_x = (torch.exp(p[3 * M:4 * M].clamp(-6.0, 2.0)).numpy() + 1e-3) * math.sqrt(temperature)
        sig_y = (torch.exp(p[4 * M:5 * M].clamp(-6.0, 2.0)).numpy() + 1e-3) * math.sqrt(temperature)
        rho = (torch.tanh(p[5 * M:6 * M]).numpy()) * 0.95
        pen_probs = F.softmax(p[6 * M:] / pen_temp, dim=-1).numpy()

        pen_state = int(np.random.choice(3, p=pen_probs))
        if pen_state == 2 and len(strokes) + (1 if len(cur_stroke) >= 2 else 0) >= 1:
            break
        if pen_state == 2:
            pen_state = 0

        comp = int(np.random.choice(M, p=pi))
        cov = [
            [sig_x[comp] ** 2, rho[comp] * sig_x[comp] * sig_y[comp]],
            [rho[comp] * sig_x[comp] * sig_y[comp], sig_y[comp] ** 2],
        ]
        dx, dy = np.random.multivariate_normal([mu_x[comp], mu_y[comp]], cov)
        cur_x += float(dx)
        cur_y += float(dy)

        if pen_state == 1:  # pen up / new stroke
            if len(cur_stroke) >= 2:
                strokes.append(cur_stroke)
            cur_stroke = [[cur_x, cur_y]]
        else:
            if not cur_stroke:
                cur_stroke = [[cur_x - float(dx), cur_y - float(dy)]]
            cur_stroke.append([cur_x, cur_y])

        next_pen = [0.0, 0.0, 0.0]
        next_pen[pen_state] = 1.0
        x = torch.tensor([[[float(dx), float(dy), *next_pen]]], dtype=torch.float32)

    if len(cur_stroke) >= 2:
        strokes.append(cur_stroke)
    return strokes


def export_to_browser_json(model, metrics, path):
    """Export GRU + MDN weights rounded to 4 decimals for compact pure-JS execution."""
    def arr(t):
        return np.round(t.detach().cpu().numpy().astype(np.float32), 4).tolist()

    payload = {
        "architecture": "ConditionalStrokeGRU-MDN",
        "classes": CLASSES,
        "hidden_size": model.hidden_size,
        "emb_dim": model.emb_dim,
        "num_mixtures": model.num_mixtures,
        "max_seq_len": MAX_SEQ_LEN,
        "metrics": metrics,
        "weights": {
            "class_emb": arr(model.class_emb.weight),
            "gru_w_ih": arr(model.gru.weight_ih_l0),
            "gru_w_hh": arr(model.gru.weight_hh_l0),
            "gru_b_ih": arr(model.gru.bias_ih_l0),
            "gru_b_hh": arr(model.gru.bias_hh_l0),
            "head_w": arr(model.head.weight),
            "head_b": arr(model.head.bias),
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    size_kb = os.path.getsize(path) / 1024
    print(f"[export] browser weights saved -> {path} ({size_kb:.1f} KB)")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)

    all_samples = []
    for cid, cname in enumerate(CLASSES):
        p = os.path.join(DATA_DIR, f"qd_{cname}.json")
        with open(p) as f:
            raw = json.load(f)
        count = 0
        for item in raw:
            if not is_clean_primitive(cname, item["drawing"]):
                continue
            seq = normalize_and_encode(item["drawing"])
            if seq is not None and len(seq) >= 3:
                all_samples.append((cid, seq))
                count += 1
                if count >= 1800:
                    break
        print(f"[data] {cname:10s}: {count} clean sequences")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_samples))
    n_train = int(0.85 * len(all_samples))
    n_val = int(0.10 * len(all_samples))
    train_samples = [all_samples[i] for i in perm[:n_train]]
    val_samples = [all_samples[i] for i in perm[n_train:n_train + n_val]]
    test_samples = [all_samples[i] for i in perm[n_train + n_val:]]

    train_loader = DataLoader(StrokeDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StrokeDataset(val_samples), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(StrokeDataset(test_samples), batch_size=BATCH_SIZE, shuffle=False)

    model = ConditionalStrokeGRU()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] ConditionalStrokeGRU-MDN: {n_params:,} params | train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val = float("inf")
    best_state = None
    history = []
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_coord, tr_pen, n_b = 0.0, 0.0, 0.0, 0
        for cls_id, inp, tgt, mask in train_loader:
            opt.zero_grad()
            params, _ = model(cls_id, inp)
            loss, lc, lp = mdn_loss(params, tgt, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
            tr_coord += lc
            tr_pen += lp
            n_b += 1
        sched.step()

        model.eval()
        va_loss, va_coord, va_pen, vb = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for cls_id, inp, tgt, mask in val_loader:
                params, _ = model(cls_id, inp)
                loss, lc, lp = mdn_loss(params, tgt, mask)
                va_loss += loss.item()
                va_coord += lc
                va_pen += lp
                vb += 1

        tr_loss /= n_b
        va_loss /= vb
        va_coord /= vb
        va_pen /= vb
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4), "val_loss": round(va_loss, 4)})
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 4 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:2d}/{EPOCHS} | train={tr_loss:+.4f} | val={va_loss:+.4f} (coord={va_coord:+.4f}, pen={va_pen:.4f}) | {time.time()-t0:.1f}s")

    model.load_state_dict(best_state)
    model.eval()
    te_loss, te_coord, te_pen, tb = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for cls_id, inp, tgt, mask in test_loader:
            params, _ = model(cls_id, inp)
            loss, lc, lp = mdn_loss(params, tgt, mask)
            te_loss += loss.item()
            te_coord += lc
            te_pen += lp
            tb += 1
    te_loss /= tb
    te_coord /= tb
    te_pen /= tb
    print(f"[test] Held-out Test NLL={te_loss:+.4f} (coord={te_coord:+.4f}, pen={te_pen:.4f})")

    # Sampling sanity check
    sample_stats = {}
    for cid, cname in enumerate(CLASSES):
        lengths = []
        n_strokes = []
        for _ in range(10):
            s = sample_sketch(model, cid, temperature=0.35)
            n_strokes.append(len(s))
            lengths.append(sum(len(st) for st in s))
        sample_stats[cname] = {"avg_strokes": round(float(np.mean(n_strokes)), 2), "avg_pts": round(float(np.mean(lengths)), 1)}
    print("[samples @ T=0.35]", json.dumps(sample_stats))

    pt_path = os.path.join(EXPORT_DIR, "stroke_gru_mdn.pt")
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "history": history}, pt_path)

    metrics = {
        "params": n_params,
        "train_samples": len(train_samples),
        "test_nll": round(te_loss, 4),
        "test_coord_nll": round(te_coord, 4),
        "test_pen_ce": round(te_pen, 4),
        "train_seconds": round(time.time() - t0, 1),
        "sample_stats": sample_stats,
        "history": history,
    }
    export_to_browser_json(model, metrics, os.path.join(EXPORT_DIR, "stroke_gru_mdn.json"))


if __name__ == "__main__":
    main()
