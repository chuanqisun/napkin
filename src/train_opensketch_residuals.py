"""
Train DesignerStrokeResidualVAE on real professional industrial designer Wacom strokes
from OpenSketch (Inria, CC0).

Learns smooth human trajectory residuals (rx, ry) + real stylus pressure profile (p)
over K=32 arc-length steps along any canonical straight line or curved Bezier segment.

Conditioning: [is_curved, is_presentation, norm_log_len]
Output per stroke: 32 x (rx, ry, p) where (rx, ry) are in unit-chord coordinates and p in [0, 1].
Exports decoder-only weights to compact JSON for <0.2ms browser inference.
"""
import glob
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

OS_DIR = "/home/antigravity/napkin-ml/data/opensketch_json"
EXPORT_DIR = "/home/antigravity/napkin-ml/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

K = 32
LATENT_DIM = 8
RES_SCALE = 0.05  # typical designer residual magnitude (~1-4% of chord length)
EPOCHS = 32
BATCH_SIZE = 256
LR = 2.5e-3
SEED = 42


def extract_stroke_residual(points):
    """
    Given OpenSketch points [{x, y, p, t}, ...], align into canonical chord frame (0,0)->(1,0),
    resample to K=32 points, and compute residuals relative to ideal straight line or quadratic Bezier.
    Returns (seq (32, 3), is_curved (float), log_len (float)) or None.
    """
    if len(points) < 6:
        return None
    xs = np.array([pt["x"] for pt in points], dtype=np.float64)
    ys = np.array([pt["y"] for pt in points], dtype=np.float64)
    ps = np.array([pt.get("p", 0.5) for pt in points], dtype=np.float64)

    dx = np.diff(xs)
    dy = np.diff(ys)
    segs = np.hypot(dx, dy)
    arc = float(segs.sum())
    chord = float(math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    if arc < 35.0 or chord < 25.0:
        return None

    ratio = chord / max(arc, 1e-6)
    if ratio <= 0.58:
        return None  # skip tight loops / scribbles
    is_curved = 0.0 if ratio > 0.965 else 1.0

    # Resample uniformly by arc length to K points
    cum = np.concatenate([[0.0], np.cumsum(segs)])
    t_grid = np.linspace(0.0, arc, K)
    rx = np.interp(t_grid, cum, xs)
    ry = np.interp(t_grid, cum, ys)
    rp = np.clip(np.interp(t_grid, cum, ps), 0.04, 1.0)

    # Rotate & scale so start=(0,0) and end=(1,0)
    ux = (rx[-1] - rx[0]) / chord
    uy = (ry[-1] - ry[0]) / chord
    # Canonical coords: u along chord, v perpendicular
    du = rx - rx[0]
    dv = ry - ry[0]
    u = (du * ux + dv * uy) / chord
    v = (-du * uy + dv * ux) / chord

    t = np.linspace(0.0, 1.0, K)
    if is_curved < 0.5:
        ideal_u = t
        ideal_v = np.zeros_like(t)
    else:
        # Fit quadratic Bezier B(t) = (1-t)^2*(0,0) + 2(1-t)t*(p1u, p1v) + t^2*(1,0)
        b1 = 2.0 * (1.0 - t) * t
        denom = float((b1 * b1).sum()) + 1e-8
        p1u = float((b1 * (u - t * t)).sum()) / denom
        p1v = float((b1 * v).sum()) / denom
        ideal_u = b1 * p1u + t * t
        ideal_v = b1 * p1v

    err_u = u - ideal_u
    err_v = v - ideal_v

    # Reject extreme outliers (> 12% chord residual) to keep refined designer distribution clean
    if float(np.max(np.abs(err_u))) > 0.12 or float(np.max(np.abs(err_v))) > 0.12:
        return None

    # Normalize log_len roughly to [-1, 1] for chord lengths 30..600px
    norm_log_len = (math.log(max(chord, 25.0)) - math.log(140.0)) / 1.2
    norm_log_len = max(-1.5, min(1.5, norm_log_len))

    seq = np.stack([err_u / RES_SCALE, err_v / RES_SCALE, rp], axis=-1).astype(np.float32)
    return seq, is_curved, norm_log_len


class DesignerStrokeResidualVAE(nn.Module):
    def __init__(self, k=K, cond_dim=3, latent_dim=LATENT_DIM):
        super().__init__()
        self.k = k
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        in_dim = k * 3 + cond_dim

        self.enc1 = nn.Linear(in_dim, 128)
        self.enc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.dec1 = nn.Linear(latent_dim + cond_dim, 64)
        self.dec2 = nn.Linear(64, 128)
        self.dec3 = nn.Linear(128, k * 3)

    def encode(self, x_flat, cond):
        h = F.silu(self.enc1(torch.cat([x_flat, cond], dim=-1)))
        h = F.silu(self.enc2(h))
        return self.fc_mu(h), self.fc_logvar(h).clamp(-6.0, 2.0)

    def decode(self, z, cond):
        h = F.silu(self.dec1(torch.cat([z, cond], dim=-1)))
        h = F.silu(self.dec2(h))
        out = self.dec3(h).view(-1, self.k, 3)
        # Residual coordinates unconstrained (scaled), pressure in (0, 1)
        xy = out[..., :2]
        p = torch.sigmoid(out[..., 2:3])
        return torch.cat([xy, p], dim=-1)

    def forward(self, x, cond):
        mu, logvar = self.encode(x.view(x.size(0), -1), cond)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(z, cond)
        return recon, mu, logvar


def vae_loss(recon, target, mu, logvar, beta):
    xy_loss = F.mse_loss(recon[..., :2], target[..., :2])
    p_loss = F.mse_loss(recon[..., 2], target[..., 2])
    # Second-order smoothness regularizer so generated strokes stay silky & fluid
    d2_recon = recon[:, 2:, :2] - 2.0 * recon[:, 1:-1, :2] + recon[:, :-2, :2]
    d2_tgt = target[:, 2:, :2] - 2.0 * target[:, 1:-1, :2] + target[:, :-2, :2]
    smooth_loss = F.mse_loss(d2_recon, d2_tgt)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    total = xy_loss + 1.8 * p_loss + 0.5 * smooth_loss + beta * kl
    return total, xy_loss.item(), p_loss.item(), kl.item()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)

    files = sorted(glob.glob(os.path.join(OS_DIR, "**", "*.json"), recursive=True))
    print(f"[data] scanning {len(files)} OpenSketch professional designer JSONs...")

    seqs, conds, designer_ids = [], [], []
    designer_map = {}
    for fp in files:
        parts = fp.split(os.sep)
        designer = parts[-3]
        if designer not in designer_map:
            designer_map[designer] = len(designer_map)
        did = designer_map[designer]
        is_pres = 1.0 if "presentation" in os.path.basename(fp) else 0.0

        with open(fp) as f:
            data = json.load(f)
        for st in data.get("strokes", []):
            if st.get("is_removed"):
                continue
            res = extract_stroke_residual(st.get("points", []))
            if res is None:
                continue
            seq, is_curved, norm_log_len = res
            seqs.append(seq)
            conds.append([is_curved, is_pres, norm_log_len])
            designer_ids.append(did)

    X = np.stack(seqs, axis=0)
    C = np.array(conds, dtype=np.float32)
    D = np.array(designer_ids, dtype=np.int64)
    print(f"[data] extracted {len(X):,} clean designer strokes across {len(designer_map)} designers "
          f"(straight={(C[:,0]==0).sum():,}, curved={(C[:,0]==1).sum():,}, "
          f"concept={(C[:,1]==0).sum():,}, presentation={(C[:,1]==1).sum():,})")

    # Split strictly by designer ID (hold out last 2 designers for true generalization test!)
    n_des = len(designer_map)
    test_mask = D >= (n_des - 2)
    val_mask = D == (n_des - 3)
    train_mask = ~(test_mask | val_mask)

    X_tr, C_tr = X[train_mask], C[train_mask]
    X_va, C_va = X[val_mask], C[val_mask]
    X_te, C_te = X[test_mask], C[test_mask]
    print(f"[split] designer-disjoint: train={len(X_tr):,} | val={len(X_va):,} | test={len(X_te):,}")

    tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(C_tr)), batch_size=BATCH_SIZE, shuffle=True)
    model = DesignerStrokeResidualVAE()
    total_params = sum(p.numel() for p in model.parameters())
    dec_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("dec"))
    print(f"[model] DesignerStrokeResidualVAE: total={total_params:,} params | browser decoder={dec_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        beta = min(0.02, 0.02 * (epoch / 10.0))
        tr_tot, nb = 0.0, 0
        for bx, bc in tr_loader:
            opt.zero_grad()
            recon, mu, logvar = model(bx, bc)
            loss, _, _, _ = vae_loss(recon, bx, mu, logvar, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_tot += loss.item(); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            recon, mu, logvar = model(torch.from_numpy(X_va), torch.from_numpy(C_va))
            vl, vxy, vp, vkl = vae_loss(recon, torch.from_numpy(X_va), mu, logvar, beta=0.02)
        if vl.item() < best_val:
            best_val = vl.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 8 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:2d}/{EPOCHS} | train={tr_tot/nb:.4f} | val={vl.item():.4f} (xy={vxy:.4f}, p={vp:.4f}, kl={vkl:.3f}) | {time.time()-t0:.1f}s")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        recon, mu, logvar = model(torch.from_numpy(X_te), torch.from_numpy(C_te))
        tl, txy, tp, tkl = vae_loss(recon, torch.from_numpy(X_te), mu, logvar, beta=0.02)
        # Measure real RMS transverse error in % of stroke length & pressure MAE
        xy_rmse_pct = float(torch.sqrt(F.mse_loss(recon[..., :2], torch.from_numpy(X_te)[..., :2])) * RES_SCALE * 100)
        p_mae = float(F.l1_loss(recon[..., 2], torch.from_numpy(X_te)[..., 2]))

        # Inspect generated pressure profile at start / mid / end for straight presentation stroke
        z_test = torch.randn(32, LATENT_DIM) * 0.7
        c_pres = torch.tensor([[0.0, 1.0, 0.0]]).expand(32, -1)
        gen = model.decode(z_test, c_pres)
        p_start = float(gen[:, 0, 2].mean())
        p_mid = float(gen[:, K // 2, 2].mean())
        p_end = float(gen[:, -1, 2].mean())
        v_std_pct = float(gen[..., 1].std() * RES_SCALE * 100)

    print(f"[test] Unseen-Designer Test: xy_rmse={xy_rmse_pct:.2f}% chord | pressure_MAE={p_mae:.4f} | kl={tkl:.3f}")
    print(f"[sample] Generated Wacom Pressure Profile: start={p_start:.2f} -> mid={p_mid:.2f} -> end={p_end:.2f} | transverse_std={v_std_pct:.2f}%")

    # Export decoder-only weights for browser JS
    def arr(t):
        return np.round(t.detach().cpu().numpy().astype(np.float32), 4).tolist()

    payload = {
        "architecture": "DesignerStrokeResidualVAE-Decoder",
        "dataset": "OpenSketch (Inria, CC0)",
        "k": K,
        "latent_dim": LATENT_DIM,
        "cond_dim": 3,
        "res_scale": RES_SCALE,
        "metrics": {
            "total_strokes": int(len(X)),
            "num_designers": len(designer_map),
            "decoder_params": dec_params,
            "test_xy_rmse_pct": round(xy_rmse_pct, 3),
            "test_pressure_mae": round(p_mae, 4),
            "pressure_profile": {"start": round(p_start, 3), "mid": round(p_mid, 3), "end": round(p_end, 3)},
            "train_seconds": round(time.time() - t0, 1),
        },
        "layers": [
            {"w": arr(model.dec1.weight), "b": arr(model.dec1.bias)},
            {"w": arr(model.dec2.weight), "b": arr(model.dec2.bias)},
            {"w": arr(model.dec3.weight), "b": arr(model.dec3.bias)},
        ],
    }
    out_json = os.path.join(EXPORT_DIR, "opensketch_residual_vae.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    torch.save({"state_dict": model.state_dict(), "metrics": payload["metrics"]}, os.path.join(EXPORT_DIR, "opensketch_residual_vae.pt"))
    print(f"[export] browser decoder saved -> {out_json} ({os.path.getsize(out_json)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
