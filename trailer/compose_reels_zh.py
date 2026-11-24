import json
import os
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
VO_DIR = ROOT / "audio" / "vo_zh"
EDIT_DIR = ROOT / "edit"
EDIT_DIR.mkdir(parents=True, exist_ok=True)

OUT_W, OUT_H = 1080, 1920
FPS = 30

# Noto Sans CJK SC Bold (index=2 verified via PIL)
FONT_ZH_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_ZH_IDX = 2


def zh_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_ZH_TTC, size, index=FONT_ZH_IDX)


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def ease_in_out_quint(u):
    u = clamp(u, 0.0, 1.0)
    return 16 * pow(u, 5) if u < 0.5 else 1 - pow(-2 * u + 2, 5) / 2


def ease_out_expo(u):
    u = clamp(u, 0.0, 1.0)
    return 1.0 if u >= 1.0 else 1.0 - pow(2.0, -10.0 * u)


def ease_out_back(u):
    u = clamp(u, 0.0, 1.0)
    c1 = 1.55
    c3 = c1 + 1.0
    return 1.0 + c3 * pow(u - 1.0, 3) + c1 * pow(u - 1.0, 2)


def load_video_frames(mp4_path: Path):
    meta = json.loads(mp4_path.with_suffix(".json").read_text())
    w, h = meta["width"], meta["height"]
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(mp4_path),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_bytes = w * h * 3
    frames = []
    while True:
        chunk = proc.stdout.read(frame_bytes)
        if len(chunk) < frame_bytes:
            break
        frames.append(chunk)
    proc.wait()
    return frames, meta


def eval_camera(kfs, t_local):
    if t_local <= kfs[0][0]:
        return kfs[0][1], kfs[0][2], kfs[0][3], False
    for i in range(len(kfs) - 1):
        t0, z0, x0, y0, _ = kfs[i]
        t1, z1, x1, y1, mode = kfs[i + 1]
        if t_local <= t1:
            if mode == "hold":
                return z0, x0, y0, False
            span = max(1e-6, t1 - t0)
            p = (t_local - t0) / span
            if mode == "crash":
                e = ease_out_expo(p)
            elif mode == "back":
                e = ease_out_back(p)
            else:
                e = ease_in_out_quint(p)
            return (
                z0 + (z1 - z0) * e,
                x0 + (x1 - x0) * e,
                y0 + (y1 - y0) * e,
                True,
            )
    return kfs[-1][1], kfs[-1][2], kfs[-1][3], False


def render_camera_frame(raw_rgb: bytes, raw_w: int, raw_h: int, dpr: float, zoom: float, cx_log: float, cy_log: float, is_moving=False):
    img = Image.frombytes("RGB", (raw_w, raw_h), raw_rgb)
    zoom = max(1.0, zoom)
    if zoom <= 1.001:
        return img.resize((OUT_W, OUT_H), Image.Resampling.BICUBIC)
    crop_w = raw_w / zoom
    crop_h = raw_h / zoom
    left = clamp(cx_log * dpr - crop_w * 0.5, 0.0, raw_w - crop_w)
    top = clamp(cy_log * dpr - crop_h * 0.5, 0.0, raw_h - crop_h)
    if not is_moving:
        box = (int(round(left)), int(round(top)), int(round(left + crop_w)), int(round(top + crop_h)))
        return img.crop(box).resize((OUT_W, OUT_H), Image.Resampling.BICUBIC)
    sx = crop_w / OUT_W
    sy = crop_h / OUT_H
    return img.transform(
        (OUT_W, OUT_H),
        Image.Transform.AFFINE,
        (sx, 0.0, left, 0.0, sy, top),
        resample=Image.Resampling.BILINEAR,
    )


def draw_zh_caption(im: Image.Image, chars_with_state, alpha=1.0, y_center=1445, accent=False, scale=1.0):
    """Render crisp Simplified Chinese caption pill with character-synced amber highlight!"""
    if alpha <= 0.01 or not chars_with_state:
        return im
    overlay = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = int(round((54 if accent else 48) * scale))
    font = zh_font(font_size)

    # Measure total width & per-character advances
    full_text = "".join(ch for ch, _ in chars_with_state)
    full_bb = draw.textbbox((0, 0), full_text, font=font)
    total_w = full_bb[2] - full_bb[0]
    ref_bb = draw.textbbox((0, 0), "值一个亿！", font=font)
    line_h = ref_bb[3] - ref_bb[1]

    pad_x = 44 if accent else 38
    pad_y = 22 if accent else 20
    box_w = total_w + pad_x * 2
    box_h = line_h + pad_y * 2
    x0 = (OUT_W - box_w) // 2
    y0 = int(y_center - box_h // 2)
    x1 = x0 + box_w
    y1 = y0 + box_h

    draw.rounded_rectangle([x0 + 4, y0 + 10, x1 + 4, y1 + 10], radius=24, fill=(32, 22, 12, int(85 * alpha)))

    if accent:
        bg = (185, 28, 28, int(244 * alpha))
        border = (255, 245, 230, int(225 * alpha))
    else:
        bg = (29, 25, 21, int(236 * alpha))
        border = (240, 228, 208, int(95 * alpha))

    draw.rounded_rectangle([x0, y0, x1, y1], radius=22, fill=bg, outline=border, width=3)

    # Render character-by-character with accurate kerning via prefix widths
    start_x = (OUT_W - total_w) // 2 - full_bb[0]
    ty = y0 + pad_y - ref_bb[1]
    for idx, (ch, st) in enumerate(chars_with_state):
        prefix = full_text[:idx]
        px = draw.textlength(prefix, font=font) if prefix else 0.0
        if accent:
            col = (255, 250, 242, int(255 * alpha))
        elif st == "active":
            col = (252, 211, 77, int(255 * alpha))
        elif st == "past":
            col = (250, 244, 234, int(255 * alpha))
        else:
            col = (250, 244, 234, int(175 * alpha))
        draw.text((start_x + px, ty), ch, font=font, fill=col)

    return Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")


def draw_free_beta_cta_zh(im: Image.Image, progress: float):
    if progress <= 0.01:
        return im
    e = ease_out_back(min(1.0, progress))
    alpha = clamp(progress * 2.5, 0.0, 1.0)
    overlay = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_main = zh_font(46)
    font_sub = zh_font(28)
    label = "免费公测 · 立即开玩"
    bb = draw.textbbox((0, 0), label, font=font_main)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]

    icon_r = 26
    gap = 20
    pad_x = 46
    pad_y = 28
    box_w = icon_r * 2 + gap + tw + pad_x * 2
    box_h = max(icon_r * 2, th) + pad_y * 2

    y_slide = int((1.0 - e) * 32)
    y_center = 1175 + y_slide
    x0 = (OUT_W - box_w) // 2
    y0 = y_center - box_h // 2
    x1 = x0 + box_w
    y1 = y0 + box_h

    draw.rounded_rectangle([x0 + 5, y0 + 12, x1 + 5, y1 + 12], radius=34, fill=(40, 28, 16, int(95 * alpha)))
    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=32,
        fill=(28, 25, 22, int(246 * alpha)),
        outline=(245, 236, 220, int(175 * alpha)),
        width=3,
    )

    icx = x0 + pad_x + icon_r
    icy = y_center
    amber = (245, 158, 11, int(255 * alpha))
    dark = (28, 25, 22, int(255 * alpha))
    cream = (250, 244, 234, int(255 * alpha))
    draw.ellipse([icx - icon_r, icy - icon_r, icx + icon_r, icy + icon_r], fill=amber)
    tri = [(icx - 7, icy - 12), (icx - 7, icy + 12), (icx + 13, icy)]
    draw.polygon(tri, fill=dark)

    tx = icx + icon_r + gap - bb[0]
    ty = y_center - th // 2 - bb[1]
    draw.text((tx, ty), label, font=font_main, fill=cream)

    sub = "你的下一个好点子，趁热画下来。"
    sbb = draw.textbbox((0, 0), sub, font=font_sub)
    sw = sbb[2] - sbb[0]
    draw.text(
        ((OUT_W - sw) // 2 - sbb[0], y1 + 28 - sbb[1]),
        sub,
        font=font_sub,
        fill=(64, 50, 36, int(225 * alpha)),
    )

    return Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")


def marker_frame(meta, label, default=0):
    for m in meta["markers"]:
        if m["label"] == label:
            return m["frame"]
    return default


def marker_obj(meta, label):
    for m in meta["markers"]:
        if m["label"] == label:
            return m
    return None


def sample_segment(frames_list, meta, src_start, src_end, out_count):
    n_src = len(frames_list)
    out = []
    for i in range(out_count):
        u = i / max(1, out_count - 1) if out_count > 1 else 0.0
        sf = int(round(src_start + (src_end - src_start) * u))
        sf = clamp(sf, 0, n_src - 1)
        out.append((sf, frames_list[sf], meta["frames"][sf]))
    return out


def build_zh():
    print("Loading 5 raw 1440x2560 takes + Chinese ElevenLabs v4 VO manifest (EXACT v4 timeline)...")
    f1, m1 = load_video_frames(RAW_DIR / "01_pen_and_idea.mp4")
    f2, m2 = load_video_frames(RAW_DIR / "02_block_order_spill.mp4")
    f3, m3 = load_video_frames(RAW_DIR / "03_crumple_reset.mp4")
    f4, m4 = load_video_frames(RAW_DIR / "04_matcha_eureka.mp4")
    f5, m5 = load_video_frames(RAW_DIR / "05_endcard_napkin.mp4")

    vo_manifest = json.loads((VO_DIR / "vo_manifest_zh.json").read_text())
    vo_by_id = {line["id"]: line for line in vo_manifest["lines"]}
    raw_w, raw_h, dpr = m1["width"], m1["height"], m1["dpr"]

    out_specs = []
    audio_events = []
    vo_captions = []
    stamp_captions = []
    music_cues = {}

    def cur_t():
        return len(out_specs) / FPS

    def place_vo(vo_id, start_t, tempo=1.0, y_center=1445, show_caption=True, split_idx=None):
        line = vo_by_id[vo_id]
        dur = line["duration"] / tempo
        audio_events.append({"kind": "vo_zh", "id": vo_id, "t": round(start_t, 4), "tempo": tempo, "dur": round(dur, 4)})
        if not show_caption:
            return start_t + dur
        chars = [
            {"ch": c["ch"], "start": start_t + c["start"] / tempo, "end": start_t + c["end"] / tempo}
            for c in line["chars"]
        ]
        groups = [chars[:split_idx], chars[split_idx:]] if split_idx else [chars]
        for gi, grp in enumerate(groups):
            if not grp:
                continue
            p_start = grp[0]["start"]
            p_end = (groups[gi + 1][0]["start"] - 0.03) if gi < len(groups) - 1 else (grp[-1]["end"] + 0.16)
            vo_captions.append({"start": p_start, "end": p_end, "y": y_center, "chars": grp})
        return start_t + dur

    # =========================================================================
    # BEAT 1 (EXACT 5.83s = 175f): Big Idea -> Pen Menu -> 2.28x Fountain Pen -> ? box
    # =========================================================================
    b1_t = cur_t()
    place_vo("vo_01a", b1_t + 0.05, tempo=1.05, y_center=1445)

    big_idea_done = marker_frame(m1, "big_idea_done", 68)
    ftn_sel = marker_frame(m1, "fountain_selected", 112)
    pen_close = marker_frame(m1, "pen_menu_closed", 128)
    ftn_m = marker_obj(m1, "fountain_selected") or {"x": 190, "y": 300}

    # 1A (55f = 1.83s)
    seg1a = sample_segment(f1, m1, 0, big_idea_done, 55)
    for i, (sf, rgb, fr) in enumerate(seg1a):
        out_specs.append({"rgb": rgb, "zoom": 1.30, "cx": 336.0, "cy": 430.0, "moving": False})
    audio_events.append({"kind": "el_sfx", "id": "fountain_scratch", "t": b1_t + 0.08, "maxDur": 0.65, "gain": 0.58})
    audio_events.append({"kind": "el_sfx", "id": "fountain_scratch", "t": b1_t + 1.05, "maxDur": 0.55, "gain": 0.48})

    # 1B (90f = 3.00s): Pen Menu opens right on "第一步" -> 2.28x zoom on "有排面的笔"!
    b1b_t = cur_t()
    place_vo("vo_01b", b1b_t + 0.06, tempo=1.04, y_center=1445)
    seg1b = sample_segment(f1, m1, big_idea_done, ftn_sel, 36) + sample_segment(f1, m1, ftn_sel, pen_close, 54)
    cam1b = [
        (0.00, 1.30, 336, 430, "hold"),
        (0.20, 1.50, 220, 265, "crash"),
        (1.02, 1.50, 220, 265, "hold"),
        (1.22, 2.28, ftn_m["x"] - 10, ftn_m["y"] + 8, "crash"),
        (2.25, 2.28, ftn_m["x"] - 10, ftn_m["y"] + 8, "hold"),
        (2.48, 1.52, 220, 310, "snap"),
        (2.80, 1.52, 220, 310, "hold"),
        (3.00, 1.24, 340, 525, "snap"),
    ]
    for i, (sf, rgb, fr) in enumerate(seg1b):
        z, cx, cy, mv = eval_camera(cam1b, i / FPS)
        out_specs.append({"rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "moving": mv})
    audio_events.append({"kind": "el_sfx", "id": "zoom_whoosh", "t": b1b_t + 0.02, "gain": 0.30})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b1b_t + 0.24, "gain": 0.58})
    audio_events.append({"kind": "el_sfx", "id": "zoom_whoosh", "t": b1b_t + 1.02, "gain": 0.38})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b1b_t + 1.15, "gain": 0.72})

    # 1C (30f = 1.00s): Arrow + Box + freehand "?"
    b1c_t = cur_t()
    seg1c = sample_segment(f1, m1, pen_close, len(f1) - 1, 30)
    for i, (sf, rgb, fr) in enumerate(seg1c):
        out_specs.append({"rgb": rgb, "zoom": 1.24, "cx": 340.0, "cy": 525.0, "moving": False})
    audio_events.append({"kind": "el_sfx", "id": "fountain_scratch", "t": b1c_t + 0.08, "maxDur": 0.80, "gain": 0.60})

    # =========================================================================
    # BEAT 2 (EXACT 2.97s = 89f): Block -> Order Coffee Latte
    # =========================================================================
    b2_t = cur_t()
    undo_f = marker_frame(m2, "undo_block", 32)
    sip_click_f = marker_frame(m2, "sip_clicked", 97)
    splat_f = marker_frame(m2, "splat_hit", 142)
    mug_ret_f = marker_frame(m2, "mug_returned", 194)
    splat_m = marker_obj(m2, "splat_hit") or {"x": 336, "y": 649}
    ring_m = marker_obj(m2, "coffee_ring_revealed") or {"x": 585, "y": 330}

    seg2a = sample_segment(f2, m2, 0, undo_f + 12, 29)
    for i, (sf, rgb, fr) in enumerate(seg2a):
        out_specs.append({"rgb": rgb, "zoom": 1.16, "cx": 350.0, "cy": 510.0, "moving": False})
    audio_events.append({"kind": "el_sfx", "id": "fountain_scratch", "t": b2_t + 0.05, "maxDur": 0.30, "gain": 0.45})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b2_t + 0.50, "gain": 0.55})
    audio_events.append({"kind": "clip", "id": "sigh_block", "t": b2_t + 0.54, "maxDur": 0.90, "gain": 0.95})

    b2b_t = cur_t()
    place_vo("vo_02", b2b_t - 0.04, tempo=1.14, y_center=1445)
    seg2b = sample_segment(f2, m2, undo_f + 12, sip_click_f - 4, 60)
    cam2b = [
        (0.00, 1.16, 350, 510, "hold"),
        (0.22, 1.52, 240, 235, "crash"),
        (1.22, 1.52, 240, 235, "hold"),
        (1.48, 1.04, 360, 510, "snap"),
        (2.00, 1.04, 360, 510, "hold"),
    ]
    for i, (sf, rgb, fr) in enumerate(seg2b):
        z, cx, cy, mv = eval_camera(cam2b, i / FPS)
        out_specs.append({"rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "moving": mv})
    audio_events.append({"kind": "el_sfx", "id": "zoom_whoosh", "t": b2b_t + 0.02, "gain": 0.30})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b2b_t + 0.48, "gain": 0.58})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b2b_t + 0.90, "gain": 0.62})
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": b2b_t + 1.48, "gain": 1.10})

    # =========================================================================
    # BEAT 3 (EXACT 4.87s = 146f): Sip -> SPLAT ("糟了！") -> Coffee Ring ("第三步……完美翻车。")
    # =========================================================================
    b3_t = cur_t()
    seg3a = sample_segment(f2, m2, sip_click_f - 4, splat_f, 34)
    for i, (sf, rgb, fr) in enumerate(seg3a):
        out_specs.append({"rgb": rgb, "zoom": 1.04, "cx": 360.0, "cy": 510.0, "moving": False})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b3_t + 0.10, "gain": 0.50})
    audio_events.append({"kind": "clip", "id": "sip_ahhh", "t": b3_t + 0.24, "maxDur": 0.92, "gain": 1.20})

    splat_out_t = cur_t()
    music_cues["splatCutStart"] = splat_out_t
    seg3b = sample_segment(f2, m2, splat_f, mug_ret_f - 10, 38)
    cam3b = [
        (0.00, 1.04, 360, 510, "hold"),
        (0.17, 2.48, splat_m["x"] - 8, splat_m["y"] - 4, "crash"),
        (1.27, 2.48, splat_m["x"] - 8, splat_m["y"] - 4, "hold"),
    ]
    for i, (sf, rgb, fr) in enumerate(seg3b):
        z, cx, cy, mv = eval_camera(cam3b, i / FPS)
        out_specs.append({"rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "moving": mv})
    audio_events.append({"kind": "el_sfx", "id": "coffee_splat", "t": splat_out_t, "gain": 0.95})
    audio_events.append({"kind": "el_sfx", "id": "zoom_whoosh", "t": splat_out_t, "gain": 0.42})
    audio_events.append({"kind": "clip", "id": "oops_drip", "t": splat_out_t + 0.22, "gain": 1.10})
    stamp_captions.append({"start": splat_out_t + 0.22, "end": splat_out_t + 1.20, "text": "糟了！", "y": 1350, "accent": True})

    b3c_t = cur_t()
    # Split "第三步……" and "完美翻车。" cleanly around the coffee ring reveal!
    place_vo("vo_03", b3c_t + 0.03, tempo=1.06, y_center=1460, split_idx=5)
    seg3c = sample_segment(f2, m2, mug_ret_f - 10, len(f2) - 1, 74)
    cam3c = [
        (0.00, 1.14, 420, 460, "hold"),
        (1.40, 1.14, 420, 460, "hold"),
        (1.60, 1.82, ring_m["x"] - 30, ring_m["y"] + 25, "crash"),
        (2.47, 1.82, ring_m["x"] - 30, ring_m["y"] + 25, "hold"),
    ]
    for i, (sf, rgb, fr) in enumerate(seg3c):
        z, cx, cy, mv = eval_camera(cam3c, i / FPS)
        out_specs.append({"rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "moving": mv})
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": b3c_t + 0.14, "gain": 0.90})
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": b3c_t + 1.50, "gain": 1.05})
    audio_events.append({"kind": "el_sfx", "id": "zoom_whoosh", "t": b3c_t + 1.50, "gain": 0.34})

    # =========================================================================
    # BEAT 4 (EXACT 3.93s = 118f): Hero Crumple + 3x Rapid Stack ("还好，纸巾管够。")
    # =========================================================================
    b4_t = cur_t()
    trash_f = marker_frame(m3, "trash_clicked", 14)
    crumple_start_f = marker_frame(m3, "crumple_start", 27)
    crumple_done_f = marker_frame(m3, "crumple_done", 71)
    stack_start_f = marker_frame(m3, "stack_start", 85)
    trash_m = marker_obj(m3, "trash_clicked") or {"x": 490, "y": 75}

    seg4a = sample_segment(f3, m3, 0, min(len(f3) - 1, crumple_done_f + 4), 64)
    cam4a = [
        (0.00, 1.14, 420, 440, "hold"),
        (0.24, 1.46, trash_m["x"] - 10, trash_m["y"] + 180, "snap"),
        (0.50, 1.46, trash_m["x"] - 10, trash_m["y"] + 180, "hold"),
        (0.72, 1.00, 360, 640, "snap"),
        (2.13, 1.00, 360, 640, "hold"),
    ]
    for i, (sf, rgb, fr) in enumerate(seg4a):
        z, cx, cy, mv = eval_camera(cam4a, i / FPS)
        out_specs.append({"rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "moving": mv})
    ratio4a = 64 / max(1, (crumple_done_f + 4))
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b4_t + (trash_f * ratio4a) / FPS, "gain": 0.65})
    audio_events.append({"kind": "clip", "id": "crumble", "t": b4_t + (crumple_start_f * ratio4a) / FPS, "gain": 1.32})
    audio_events.append({"kind": "clip", "id": "argh_stain", "t": b4_t + ((crumple_start_f + 4) * ratio4a) / FPS, "gain": 0.96})
    audio_events.append({"kind": "el_sfx", "id": "zoom_whoosh", "t": b4_t + ((crumple_start_f + 26) * ratio4a) / FPS, "gain": 0.34})

    b4b_t = cur_t()
    music_cues["musicResume"] = b4b_t
    place_vo("vo_04", b4b_t + 0.02, tempo=1.0, y_center=1445)

    seg4b = sample_segment(f3, m3, stack_start_f, len(f3) - 1, 54)
    for i, (sf, rgb, fr) in enumerate(seg4b):
        out_specs.append({"rgb": rgb, "zoom": 1.00, "cx": 360.0, "cy": 640.0, "moving": False})
    stack_span = max(1, (len(f3) - 1) - stack_start_f)
    for k in [2, 3, 4]:
        sf_k = marker_frame(m3, f"stack_crumple_{k}_start", stack_start_f + int(stack_span * ((k - 2) / 3.0 + 0.12)))
        t_k = b4b_t + ((sf_k - stack_start_f) / stack_span) * (54 / FPS)
        audio_events.append({"kind": "crumble_fast", "t": round(t_k, 4), "rate": 2.4, "gain": 1.10})
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": b4b_t + 1.66, "gain": 1.05})

    # =========================================================================
    # BEAT 5 (EXACT 4.43s = 133f): Matcha + Flow + EUREKA ("有了！！" -> "对，就是它！")
    # =========================================================================
    b5_t = cur_t()
    matcha_land_f = marker_frame(m4, "matcha_landed", 44)
    aha_f = marker_frame(m4, "aha_spark", 176)
    eureka_f = marker_frame(m4, "eureka_hit", 247)
    eureka_m = marker_obj(m4, "eureka_hit") or {"x": 320, "y": 845}

    seg5a = sample_segment(f4, m4, 0, matcha_land_f, 24)
    cam5a = [
        (0.00, 1.00, 360, 640, "hold"),
        (0.18, 1.48, 240, 235, "crash"),
        (0.50, 1.48, 240, 235, "hold"),
        (0.76, 1.10, 345, 580, "snap"),
        (0.80, 1.10, 345, 580, "hold"),
    ]
    for i, (sf, rgb, fr) in enumerate(seg5a):
        z, cx, cy, mv = eval_camera(cam5a, i / FPS)
        out_specs.append({"rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "moving": mv})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b5_t + 0.12, "gain": 0.52})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b5_t + 0.32, "gain": 0.58})
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": b5_t + 0.72, "gain": 1.02})

    b5b_t = cur_t()
    seg5b = sample_segment(f4, m4, matcha_land_f, eureka_f, 57)
    for i, (sf, rgb, fr) in enumerate(seg5b):
        out_specs.append({"rgb": rgb, "zoom": 1.10, "cx": 345.0, "cy": 580.0, "moving": False})
    audio_events.append({"kind": "el_sfx", "id": "fountain_scratch", "t": b5b_t + 0.08, "maxDur": 0.42, "gain": 0.45})
    audio_events.append({"kind": "el_sfx", "id": "fountain_scratch", "t": b5b_t + 0.65, "maxDur": 0.40, "gain": 0.45})
    aha_out_t = b5b_t + ((aha_f - matcha_land_f) / max(1, eureka_f - matcha_land_f)) * (57 / FPS)
    audio_events.append({"kind": "clip", "id": "aha_spark", "t": aha_out_t - 0.10, "maxDur": 0.70, "gain": 0.86})
    audio_events.append({"kind": "el_sfx", "id": "marker_sketch", "t": b5b_t + 1.25, "maxDur": 0.64, "gain": 0.72})

    eureka_out_t = cur_t()
    music_cues["eurekaHit"] = eureka_out_t
    vo5_start = eureka_out_t + 0.66
    place_vo("vo_05", vo5_start, tempo=1.18, y_center=1445)

    seg5c = sample_segment(f4, m4, eureka_f, min(len(f4) - 1, eureka_f + 52), 52)
    cam5c = [
        (0.00, 1.10, 345, 580, "hold"),
        (0.17, 1.92, eureka_m["x"], eureka_m["y"] + 8, "crash"),
        (0.80, 1.92, eureka_m["x"], eureka_m["y"] + 8, "hold"),
        (1.06, 1.00, 360, 620, "snap"),
        (1.73, 1.00, 360, 620, "hold"),
    ]
    for i, (sf, rgb, fr) in enumerate(seg5c):
        z, cx, cy, mv = eval_camera(cam5c, i / FPS)
        out_specs.append({"rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "moving": mv})
    audio_events.append({"kind": "el_sfx", "id": "zoom_whoosh", "t": eureka_out_t, "gain": 0.38})
    audio_events.append({"kind": "clip", "id": "eureka_win", "t": eureka_out_t + 0.02, "maxDur": 1.00, "gain": 1.08})
    audio_events.append({"kind": "el_sfx", "id": "eureka_chime", "t": eureka_out_t + 0.06, "gain": 0.70})
    stamp_captions.append({"start": eureka_out_t + 0.03, "end": vo5_start - 0.06, "text": "有了！！", "y": 1390, "accent": True})

    # =========================================================================
    # BEAT 6 (EXACT 2.80s = 84f): End Card + "免费公测 · 立即开玩"
    # =========================================================================
    b6_t = cur_t()
    music_cues["endCardStart"] = b6_t
    vo6_start = b6_t + 0.04
    vo6_tempo = 1.15  # Brings spoken end (3.04s) cleanly to 2.64s -> fits 2.80s exact timeline!
    place_vo("vo_06", vo6_start, tempo=vo6_tempo, show_caption=False)

    swoosh_f = marker_frame(m5, "swoosh_done", 37)
    seg6 = sample_segment(f5, m5, 3, swoosh_f, 25) + sample_segment(f5, m5, swoosh_f, min(len(f5) - 1, 68), 59)

    # Pop CTA badge cleanly on the exact millisecond "免" begins in vo_06!
    vo6_chars = vo_by_id["vo_06"]["chars"]
    mian_char = next((c for c in vo6_chars if c["ch"] == "免"), vo6_chars[min(7, len(vo6_chars) - 1)])
    cta_trigger_t = vo6_start + mian_char["start"] / vo6_tempo

    for i, (sf, rgb, fr) in enumerate(seg6):
        t_abs = b6_t + i / FPS
        cta_p = clamp((t_abs - cta_trigger_t) / 0.28, 0.0, 1.0)
        out_specs.append({
            "rgb": rgb, "zoom": 1.00, "cx": 360.0, "cy": 640.0, "moving": False,
            "free_beta_cta": cta_p,
        })
    audio_events.append({"kind": "el_sfx", "id": "fountain_scratch", "t": b6_t + 0.06, "maxDur": 0.70, "gain": 0.50})
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": b6_t + 2.52, "gain": 0.85})

    total_frames = len(out_specs)
    total_sec = total_frames / FPS
    print(f"Composing Chinese v4: {total_frames} frames ({total_sec:.2f}s @ {FPS}fps, {OUT_W}x{OUT_H})...")

    timeline = {
        "fps": FPS,
        "width": OUT_W,
        "height": OUT_H,
        "totalFrames": total_frames,
        "totalDurationSec": round(total_sec, 4),
        "musicCues": music_cues,
        "audioEvents": audio_events,
    }
    (EDIT_DIR / "timeline_zh.json").write_text(json.dumps(timeline, indent=2, ensure_ascii=False))

    silent_mp4 = EDIT_DIR / "video_zh_silent.mp4"
    ff = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{OUT_W}x{OUT_H}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(silent_mp4),
    ], stdin=subprocess.PIPE)

    for idx, sp in enumerate(out_specs):
        t_now = idx / FPS
        im = render_camera_frame(sp["rgb"], raw_w, raw_h, dpr, sp["zoom"], sp["cx"], sp["cy"], sp.get("moving", False))

        rendered_vo_cap = False
        for ph in vo_captions:
            if ph["start"] <= t_now <= ph["end"]:
                fade_in = clamp((t_now - ph["start"]) / 0.08, 0.0, 1.0)
                fade_out = clamp((ph["end"] - t_now) / 0.10, 0.0, 1.0)
                alpha = min(fade_in, fade_out)
                c_states = []
                for c in ph["chars"]:
                    if t_now < c["start"]:
                        st = "future"
                    elif t_now <= c["end"] + 0.07:
                        st = "active"
                    else:
                        st = "past"
                    c_states.append((c["ch"], st))
                im = draw_zh_caption(im, c_states, alpha=alpha, y_center=ph["y"], accent=False)
                rendered_vo_cap = True
                break

        if not rendered_vo_cap:
            for stp in stamp_captions:
                if stp["start"] <= t_now <= stp["end"]:
                    p_in = clamp((t_now - stp["start"]) / 0.11, 0.0, 1.0)
                    p_out = clamp((stp["end"] - t_now) / 0.09, 0.0, 1.0)
                    sc = 0.86 + 0.16 * ease_out_back(p_in)
                    im = draw_zh_caption(
                        im,
                        [(ch, "active") for ch in stp["text"]],
                        alpha=min(clamp(p_in * 2.0, 0.0, 1.0), p_out),
                        y_center=stp["y"],
                        accent=stp["accent"],
                        scale=sc,
                    )
                    break

        if sp.get("free_beta_cta", 0.0) > 0.01:
            im = draw_free_beta_cta_zh(im, sp["free_beta_cta"])

        ff.stdin.write(im.tobytes())
        if (idx + 1) % 150 == 0 or idx == total_frames - 1:
            print(f"  rendered {idx + 1}/{total_frames} frames...")

    ff.stdin.close()
    ret = ff.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {ret}")
    print(f"[VIDEO ZH] Wrote {silent_mp4} ({total_sec:.2f}s)")


if __name__ == "__main__":
    build_zh()
