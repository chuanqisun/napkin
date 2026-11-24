import json
import math
import os
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
VO_DIR = ROOT / "audio" / "vo"
EDIT_DIR = ROOT / "edit"
EDIT_DIR.mkdir(parents=True, exist_ok=True)

OUT_W, OUT_H = 1080, 1920
FPS = 30

FONT_BOLD = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
if not os.path.exists(FONT_BOLD):
    FONT_BOLD = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"


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
    """Evaluate decisive locked/snapped camera keyframes: (t, zoom, cx, cy, mode).
    mode='hold' locks dead still; 'snap' = decisive quintic move; 'crash' = fast expo punch.
    """
    if t_local <= kfs[0][0]:
        return kfs[0][1], kfs[0][2], kfs[0][3]
    for i in range(len(kfs) - 1):
        t0, z0, x0, y0, _ = kfs[i]
        t1, z1, x1, y1, mode = kfs[i + 1]
        if t_local <= t1:
            if mode == "hold":
                return z0, x0, y0
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
            )
    return kfs[-1][1], kfs[-1][2], kfs[-1][3]


def subpixel_crop_and_scale(raw_rgb: bytes, raw_w: int, raw_h: int, dpr: float, zoom: float, cx_log: float, cy_log: float):
    """Subpixel-accurate affine crop & resize — ZERO integer-pixel jitter!"""
    img = Image.frombytes("RGB", (raw_w, raw_h), raw_rgb)
    zoom = max(1.0, zoom)
    crop_w = raw_w / zoom
    crop_h = raw_h / zoom
    cx = cx_log * dpr
    cy = cy_log * dpr
    left = clamp(cx - crop_w * 0.5, 0.0, raw_w - crop_w)
    top = clamp(cy - crop_h * 0.5, 0.0, raw_h - crop_h)
    sx = crop_w / OUT_W
    sy = crop_h / OUT_H
    return img.transform(
        (OUT_W, OUT_H),
        Image.Transform.AFFINE,
        (sx, 0.0, left, 0.0, sy, top),
        resample=Image.Resampling.BICUBIC,
    )


def draw_word_synced_caption(im: Image.Image, words_with_state, alpha=1.0, y_center=1440, accent=False, scale=1.0):
    """Render clean pill caption with active spoken-word amber highlight!
    words_with_state: list of (word_str, state) where state in ('past', 'active', 'future')
    """
    if alpha <= 0.01 or not words_with_state:
        return im
    overlay = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = int(round((54 if accent else 46) * scale))
    font = ImageFont.truetype(FONT_BOLD, font_size)

    # Wrap cleanly into <= 2 lines if total width exceeds 840px
    space_w = draw.textbbox((0, 0), " ", font=font)[2]
    measured = []
    for w, st in words_with_state:
        bb = draw.textbbox((0, 0), w, font=font)
        measured.append((w, st, bb[2] - bb[0], bb[3] - bb[1], bb[0], bb[1]))

    lines = []
    cur_line = []
    cur_w = 0
    max_line_w = 820
    for item in measured:
        add_w = item[2] + (space_w if cur_line else 0)
        if cur_line and cur_w + add_w > max_line_w:
            lines.append((cur_line, cur_w))
            cur_line = [item]
            cur_w = item[2]
        else:
            cur_line.append(item)
            cur_w += add_w
    if cur_line:
        lines.append((cur_line, cur_w))

    # Measure consistent line height
    ref_bb = draw.textbbox((0, 0), "Ag", font=font)
    line_h = ref_bb[3] - ref_bb[1]
    line_gap = 14
    max_w = max(w for _, w in lines)
    total_h = len(lines) * line_h + (len(lines) - 1) * line_gap

    pad_x = 42 if accent else 36
    pad_y = 22 if accent else 20
    box_w = max_w + pad_x * 2
    box_h = total_h + pad_y * 2
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

    cur_y = y0 + pad_y
    for line_items, lw in lines:
        cur_x = (OUT_W - lw) // 2
        for w, st, ww, wh, ox, oy in line_items:
            if accent:
                col = (255, 250, 242, int(255 * alpha))
            elif st == "active":
                col = (252, 211, 77, int(255 * alpha))   # Warm golden highlight on current spoken word!
            elif st == "past":
                col = (250, 244, 234, int(255 * alpha))
            else:
                col = (250, 244, 234, int(175 * alpha))
            draw.text((cur_x - ox, cur_y - ref_bb[1]), w, font=font, fill=col)
            cur_x += ww + space_w
        cur_y += line_h + line_gap

    return Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")


def draw_free_beta_cta(im: Image.Image, progress: float):
    """Clean, high-craft 'PLAY FREE BETA NOW' pill badge (zero Steam mention, zero URLs!)."""
    if progress <= 0.01:
        return im
    e = ease_out_back(min(1.0, progress))
    alpha = clamp(progress * 2.5, 0.0, 1.0)
    overlay = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_main = ImageFont.truetype(FONT_BOLD, 46)
    font_sub = ImageFont.truetype(FONT_BOLD, 28)
    label = "PLAY FREE BETA NOW"
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

    # Warm shadow + deep espresso pill
    draw.rounded_rectangle([x0 + 5, y0 + 12, x1 + 5, y1 + 12], radius=34, fill=(40, 28, 16, int(95 * alpha)))
    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=32,
        fill=(28, 25, 22, int(246 * alpha)),
        outline=(245, 236, 220, int(175 * alpha)),
        width=3,
    )

    # Crisp amber play-circle icon
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

    # Subtle charming subtitle cleanly above Reels safe zone
    sub = "Your next big idea, served hot."
    sbb = draw.textbbox((0, 0), sub, font=font_sub)
    sw = sbb[2] - sbb[0]
    draw.text(
        ((OUT_W - sw) // 2 - sbb[0], y1 + 28 - sbb[1]),
        sub,
        font=font_sub,
        fill=(64, 50, 36, int(220 * alpha)),
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


def build_v2():
    print("Loading 5 raw 1440x2560 takes + ElevenLabs v4 VO manifest...")
    f1, m1 = load_video_frames(RAW_DIR / "01_pen_and_idea.mp4")
    f2, m2 = load_video_frames(RAW_DIR / "02_block_order_spill.mp4")
    f3, m3 = load_video_frames(RAW_DIR / "03_crumple_reset.mp4")
    f4, m4 = load_video_frames(RAW_DIR / "04_matcha_eureka.mp4")
    f5, m5 = load_video_frames(RAW_DIR / "05_endcard_napkin.mp4")

    vo_manifest = json.loads((VO_DIR / "vo_manifest.json").read_text())
    vo_by_id = {line["id"]: line for line in vo_manifest["lines"]}

    raw_w, raw_h, dpr = m1["width"], m1["height"], m1["dpr"]

    out_specs = []
    audio_events = []
    vo_placements = []   # { id, start_t, end_t, words, phrases }
    stamp_captions = []  # { start_t, end_t, text, y, accent }
    music_cues = {}

    def cur_t():
        return len(out_specs) / FPS

    def place_vo(vo_id, start_t, phrases_split=None, y_center=1445, show_caption=True):
        """Schedule a VO clip and auto-build phrase-grouped word-synced captions from ElevenLabs timestamps."""
        line = vo_by_id[vo_id]
        words = line["words"]
        end_t = start_t + line["duration"]
        audio_events.append({"kind": "vo", "id": vo_id, "t": round(start_t, 4), "dur": line["duration"]})
        if not show_caption:
            return end_t
        # Group words into phrases (either explicit index slices or single phrase)
        if phrases_split:
            groups = [words[a:b] for a, b in phrases_split]
        else:
            groups = [words]
        phrases = []
        for gi, grp in enumerate(groups):
            if not grp:
                continue
            p_start = start_t + grp[0]["start"]
            # Hold phrase cleanly until next phrase starts or +0.18s after final word
            if gi < len(groups) - 1:
                p_end = start_t + groups[gi + 1][0]["start"] - 0.03
            else:
                p_end = start_t + grp[-1]["end"] + 0.22
            phrases.append({
                "start": p_start,
                "end": p_end,
                "y": y_center,
                "words": [
                    {"word": w["word"], "start": start_t + w["start"], "end": start_t + w["end"]}
                    for w in grp
                ],
            })
        vo_placements.append({"id": vo_id, "start": start_t, "end": end_t, "phrases": phrases})
        return end_t

    # =========================================================================
    # BEAT 1: SELECTING PEN + "Big Idea ↓ Magic" (5.10s = 153f)
    # Timed cleanly to vo_01 (4.96s): "Billion-dollar idea." (0..1.64s) | "Step one: ridiculously fancy pen." (1.92..4.96s)
    # =========================================================================
    b1_t = cur_t()
    place_vo("vo_01", b1_t + 0.06, phrases_split=[(0, 2), (2, 7)], y_center=1440)

    pen_open = marker_frame(m1, "pen_menu_open", 10)
    pen_close = marker_frame(m1, "pen_menu_closed", 47)

    # 1A: Pen popover (54f = 1.80s) -> LOCKED on Pen menu, crisp 5-frame punch on Fountain click, LOCK!
    seg1a = sample_segment(f1, m1, max(0, pen_open - 3), pen_close, 54)
    cam1a = [
        (0.00, 1.50, 222, 265, "hold"),
        (0.95, 1.50, 222, 265, "hold"),   # Dead steady while browsing
        (1.14, 1.66, 200, 290, "crash"),  # Decisive 6-frame punch onto Fountain Pen click!
        (1.55, 1.66, 200, 290, "hold"),   # Hold steady on Designer Cursive
        (1.80, 1.24, 340, 525, "snap"),   # Decisive snap down to center napkin composition!
    ]
    for i, (sf, rgb, fr) in enumerate(seg1a):
        z, cx, cy = eval_camera(cam1a, i / FPS)
        out_specs.append({"rgb": rgb, "zoom": z, "cx": cx, "cy": cy})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b1_t + 0.12, "gain": 0.55})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b1_t + 1.08, "gain": 0.65})
    audio_events.append({"kind": "el_sfx", "id": "zoom_whoosh", "t": b1_t + 1.54, "gain": 0.30})

    # 1B: Drawing Big Idea ↓ Magic (99f = 3.30s) -> 100% LOCKED steady frame! Zero drift!
    b1b_t = cur_t()
    seg1b = sample_segment(f1, m1, pen_close, len(f1) - 1, 99)
    for i, (sf, rgb, fr) in enumerate(seg1b):
        # Rock-steady 1.24x framing capturing BOTH Big Idea and Magic cleanly!
        out_specs.append({"rgb": rgb, "zoom": 1.24, "cx": 340.0, "cy": 525.0})
    audio_events.append({"kind": "el_sfx", "id": "fountain_scratch", "t": b1b_t + 0.08, "maxDur": 0.65, "gain": 0.62})
    audio_events.append({"kind": "el_sfx", "id": "fountain_scratch", "t": b1b_t + 1.00, "maxDur": 0.55, "gain": 0.48})
    audio_events.append({"kind": "el_sfx", "id": "fountain_scratch", "t": b1b_t + 2.02, "maxDur": 0.65, "gain": 0.60})

    # =========================================================================
    # BEAT 2: WRITER'S BLOCK -> ORDER COFFEE LATTE (3.10s = 93f)
    # =========================================================================
    b2_t = cur_t()
    undo_f = marker_frame(m2, "undo_block", 32)
    sip_click_f = marker_frame(m2, "sip_clicked", 97)
    splat_f = marker_frame(m2, "splat_hit", 142)
    mug_ret_f = marker_frame(m2, "mug_returned", 194)

    splat_m = marker_obj(m2, "splat_hit") or {"x": 336, "y": 649}
    ring_m = marker_obj(m2, "coffee_ring_revealed") or {"x": 585, "y": 330}

    # 2A: Messy squiggle -> Undo -> weary sigh (33f = 1.10s, LOCKED medium frame!)
    seg2a = sample_segment(f2, m2, 0, undo_f + 12, 33)
    for i, (sf, rgb, fr) in enumerate(seg2a):
        out_specs.append({"rgb": rgb, "zoom": 1.16, "cx": 350.0, "cy": 510.0})
    audio_events.append({"kind": "el_sfx", "id": "fountain_scratch", "t": b2_t + 0.06, "maxDur": 0.35, "gain": 0.45})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b2_t + 0.58, "gain": 0.55})
    audio_events.append({"kind": "clip", "id": "sigh_block", "t": b2_t + 0.62, "maxDur": 1.05, "gain": 0.95})

    # 2B: Decisive whip-snap to LOCKED Café Menu -> Coffee Latte -> pull to LOCKED wide (60f = 2.00s)
    b2b_t = cur_t()
    place_vo("vo_02", b2b_t + 0.05, y_center=1445)
    seg2b = sample_segment(f2, m2, undo_f + 12, sip_click_f - 4, 60)
    cam2b = [
        (0.00, 1.16, 350, 510, "hold"),
        (0.22, 1.52, 240, 235, "crash"),  # Decisive 6-frame whip onto all 4 Café Menu drinks!
        (1.22, 1.52, 240, 235, "hold"),   # Dead steady lock on Café Menu
        (1.48, 1.04, 360, 510, "snap"),   # Decisive pull-out to LOCKED frame as Latte glides in!
        (2.00, 1.04, 360, 510, "hold"),
    ]
    for i, (sf, rgb, fr) in enumerate(seg2b):
        z, cx, cy = eval_camera(cam2b, i / FPS)
        out_specs.append({"rgb": rgb, "zoom": z, "cx": cx, "cy": cy})
    audio_events.append({"kind": "el_sfx", "id": "zoom_whoosh", "t": b2b_t + 0.02, "gain": 0.30})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b2b_t + 0.48, "gain": 0.58})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b2b_t + 0.90, "gain": 0.62})
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": b2b_t + 1.48, "gain": 1.10})

    # =========================================================================
    # BEAT 3: SPILL / DRIP — DEAD-STILL CRASH ZOOM + COFFEE RING (5.20s = 156f)
    # =========================================================================
    b3_t = cur_t()
    # 3A: Click Latte -> mug lifts -> loud slurp "Aaaahhh..." (38f = 1.27s, LOCKED steady!)
    seg3a = sample_segment(f2, m2, sip_click_f - 4, splat_f, 38)
    for i, (sf, rgb, fr) in enumerate(seg3a):
        out_specs.append({"rgb": rgb, "zoom": 1.04, "cx": 360.0, "cy": 510.0})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b3_t + 0.12, "gain": 0.50})
    audio_events.append({"kind": "clip", "id": "sip_ahhh", "t": b3_t + 0.28, "maxDur": 1.02, "gain": 1.20})

    # 3B: SPLAT!! 5-FRAME 2.48x CRASH ZOOM -> DEAD-STILL LOCK + MUSIC SILENCE + "Oops!" (42f = 1.40s)
    splat_out_t = cur_t()
    music_cues["splatCutStart"] = splat_out_t
    seg3b = sample_segment(f2, m2, splat_f, mug_ret_f - 10, 42)
    cam3b = [
        (0.00, 1.04, 360, 510, "hold"),
        (0.17, 2.48, splat_m["x"] - 8, splat_m["y"] - 4, "crash"),  # Decisive 5-frame crash zoom!
        (1.40, 2.48, splat_m["x"] - 8, splat_m["y"] - 4, "hold"),   # DEAD STILL comedy lock!
    ]
    for i, (sf, rgb, fr) in enumerate(seg3b):
        z, cx, cy = eval_camera(cam3b, i / FPS)
        out_specs.append({"rgb": rgb, "zoom": z, "cx": cx, "cy": cy})
    audio_events.append({"kind": "el_sfx", "id": "coffee_splat", "t": splat_out_t, "gain": 0.95})
    audio_events.append({"kind": "el_sfx", "id": "zoom_whoosh", "t": splat_out_t, "gain": 0.42})
    audio_events.append({"kind": "clip", "id": "oops_drip", "t": splat_out_t + 0.25, "gain": 1.10})
    stamp_captions.append({"start": splat_out_t + 0.24, "end": splat_out_t + 1.32, "text": "OOPS.", "y": 1350, "accent": True})

    # 3C: Mug returns -> drag reveals dark COFFEE RING -> decisive snap to LOCKED ring! (76f = 2.53s)
    # Timed cleanly to vo_03 ("Step three... ruin it." = 2.48s!)
    b3c_t = cur_t()
    place_vo("vo_03", b3c_t + 0.04, phrases_split=[(0, 2), (2, 4)], y_center=1460)
    seg3c = sample_segment(f2, m2, mug_ret_f - 10, len(f2) - 1, 76)
    cam3c = [
        (0.00, 1.14, 420, 460, "hold"),
        (1.45, 1.14, 420, 460, "hold"),   # Steady while mug lands & drags down
        (1.65, 1.82, ring_m["x"] - 30, ring_m["y"] + 25, "crash"),  # Decisive 6-frame snap onto dark coffee ring!
        (2.53, 1.82, ring_m["x"] - 30, ring_m["y"] + 25, "hold"),   # Dead steady lock on "ruin it."!
    ]
    for i, (sf, rgb, fr) in enumerate(seg3c):
        z, cx, cy = eval_camera(cam3c, i / FPS)
        out_specs.append({"rgb": rgb, "zoom": z, "cx": cx, "cy": cy})
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": b3c_t + 0.14, "gain": 0.90})
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": b3c_t + 1.56, "gain": 1.05})
    audio_events.append({"kind": "el_sfx", "id": "zoom_whoosh", "t": b3c_t + 1.56, "gain": 0.34})

    # =========================================================================
    # BEAT 4: TRASH AN IDEA — MUG LIFTS -> 3D CRUMPLE + TOSS (3.30s = 99f)
    # =========================================================================
    b4_t = cur_t()
    trash_f = marker_frame(m3, "trash_clicked", 16)
    crumple_start_f = marker_frame(m3, "crumple_start", 29)
    crumple_done_f = marker_frame(m3, "crumple_done", 73)
    trash_m = marker_obj(m3, "trash_clicked") or {"x": 490, "y": 75}

    # VO-04 ("Good thing there's a whole stack." = 1.92s) starts right as fresh napkin appears!
    vo4_start = b4_t + (crumple_done_f - 4) / FPS
    place_vo("vo_04", vo4_start, y_center=1440)

    seg4 = sample_segment(f3, m3, 0, len(f3) - 1, 99)
    cam4 = [
        (0.00, 1.14, 420, 440, "hold"),
        (0.28, 1.46, trash_m["x"] - 10, trash_m["y"] + 180, "snap"),   # Decisive snap to Trash icon!
        (0.58, 1.46, trash_m["x"] - 10, trash_m["y"] + 180, "hold"),
        (0.82, 1.00, 360, 640, "snap"),   # Decisive pull to LOCKED FULL WIDE before crumple starts!
        (3.30, 1.00, 360, 640, "hold"),   # Rock-solid full frame so 3D crumple mesh shines with ZERO jitter!
    ]
    for i, (sf, rgb, fr) in enumerate(seg4):
        z, cx, cy = eval_camera(cam4, i / FPS)
        out_specs.append({"rgb": rgb, "zoom": z, "cx": cx, "cy": cy})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b4_t + trash_f / FPS, "gain": 0.65})
    audio_events.append({"kind": "clip", "id": "crumble", "t": b4_t + crumple_start_f / FPS, "gain": 1.30})
    audio_events.append({"kind": "clip", "id": "argh_stain", "t": b4_t + (crumple_start_f + 4) / FPS, "gain": 0.96})
    audio_events.append({"kind": "el_sfx", "id": "zoom_whoosh", "t": b4_t + (crumple_start_f + 26) / FPS, "gain": 0.35})
    clink_fresh_t = b4_t + 2.74
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": clink_fresh_t, "gain": 1.08})
    music_cues["musicResume"] = clink_fresh_t - 0.05

    # =========================================================================
    # BEAT 5: MATCHA POP + VERTICAL BUILD + HERO EUREKA! (4.80s = 144f)
    # =========================================================================
    b5_t = cur_t()
    matcha_land_f = marker_frame(m4, "matcha_landed", 44)
    aha_f = marker_frame(m4, "aha_spark", 176)
    eureka_f = marker_frame(m4, "eureka_hit", 247)
    eureka_m = marker_obj(m4, "eureka_hit") or {"x": 320, "y": 845}

    # 5A: Matcha swap (27f = 0.90s) -> decisive snap to Menu -> pull to LOCKED vertical canvas!
    seg5a = sample_segment(f4, m4, 0, matcha_land_f, 27)
    cam5a = [
        (0.00, 1.00, 360, 640, "hold"),
        (0.20, 1.48, 240, 235, "crash"),
        (0.56, 1.48, 240, 235, "hold"),
        (0.84, 1.10, 345, 580, "snap"),
        (0.90, 1.10, 345, 580, "hold"),
    ]
    for i, (sf, rgb, fr) in enumerate(seg5a):
        z, cx, cy = eval_camera(cam5a, i / FPS)
        out_specs.append({"rgb": rgb, "zoom": z, "cx": cx, "cy": cy})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b5_t + 0.14, "gain": 0.52})
    audio_events.append({"kind": "el_sfx", "id": "ui_click", "t": b5_t + 0.36, "gain": 0.58})
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": b5_t + 0.80, "gain": 1.02})

    # 5B: Kinetic flow-state vertical sketch (63f = 2.10s) -> ROCK-SOLID LOCKED COMPOSITION!
    # Shows Weird Thought ↓ Napkin ↓ Ship It! + green Matcha simultaneously with ZERO restless drift!
    b5b_t = cur_t()
    seg5b = sample_segment(f4, m4, matcha_land_f, eureka_f, 63)
    for i, (sf, rgb, fr) in enumerate(seg5b):
        out_specs.append({"rgb": rgb, "zoom": 1.10, "cx": 345.0, "cy": 580.0})
    audio_events.append({"kind": "el_sfx", "id": "fountain_scratch", "t": b5b_t + 0.10, "maxDur": 0.45, "gain": 0.45})
    audio_events.append({"kind": "el_sfx", "id": "fountain_scratch", "t": b5b_t + 0.72, "maxDur": 0.42, "gain": 0.45})
    aha_out_t = b5b_t + ((aha_f - matcha_land_f) / max(1, eureka_f - matcha_land_f)) * (len(seg5b) / FPS)
    audio_events.append({"kind": "clip", "id": "aha_spark", "t": aha_out_t - 0.10, "maxDur": 0.72, "gain": 0.86})
    audio_events.append({"kind": "el_sfx", "id": "marker_sketch", "t": b5b_t + 1.38, "maxDur": 0.68, "gain": 0.72})

    # 5C: HERO EUREKA CLIMAX! (54f = 1.80s)
    # 5-frame crash zoom -> LOCK on "Ship It!" -> decisive pull-out as VO-05 ("There it is." = 1.04s) lands!
    eureka_out_t = cur_t()
    music_cues["eurekaHit"] = eureka_out_t
    vo5_start = eureka_out_t + 0.78
    place_vo("vo_05", vo5_start, y_center=1440)

    seg5c = sample_segment(f4, m4, eureka_f, min(len(f4) - 1, eureka_f + 54), 54)
    cam5c = [
        (0.00, 1.10, 345, 580, "hold"),
        (0.17, 1.92, eureka_m["x"], eureka_m["y"] + 8, "crash"),  # 5-frame hero crash zoom!
        (0.84, 1.92, eureka_m["x"], eureka_m["y"] + 8, "hold"),   # Dead-steady triumphant lock!
        (1.12, 1.00, 360, 620, "snap"),   # Decisive pull-out to full masterpiece + Matcha!
        (1.80, 1.00, 360, 620, "hold"),
    ]
    for i, (sf, rgb, fr) in enumerate(seg5c):
        z, cx, cy = eval_camera(cam5c, i / FPS)
        out_specs.append({"rgb": rgb, "zoom": z, "cx": cx, "cy": cy})
    audio_events.append({"kind": "el_sfx", "id": "zoom_whoosh", "t": eureka_out_t, "gain": 0.38})
    audio_events.append({"kind": "clip", "id": "eureka_win", "t": eureka_out_t + 0.02, "maxDur": 1.15, "gain": 1.08})
    audio_events.append({"kind": "el_sfx", "id": "eureka_chime", "t": eureka_out_t + 0.06, "gain": 0.70})
    stamp_captions.append({"start": eureka_out_t + 0.03, "end": vo5_start - 0.06, "text": "EUREKA!!", "y": 1390, "accent": True})

    # =========================================================================
    # BEAT 6: FULL-BLEED SUNLIT NAPKIN END CARD + "PLAY FREE BETA NOW" (2.87s = 86f)
    # Timed cleanly to vo_06 ("Napkin." 0..0.76s | "Play the free beta now." 0.96..2.48s)
    # =========================================================================
    b6_t = cur_t()
    music_cues["endCardStart"] = b6_t
    vo6_start = b6_t + 0.08
    # Keep visual End Card clean (Napkin writes on "Napkin.", PLAY FREE BETA NOW badge springs in right on "Play"!)
    place_vo("vo_06", vo6_start, show_caption=False)

    swoosh_f = marker_frame(m5, "swoosh_done", 37)
    seg6a = sample_segment(f5, m5, 3, swoosh_f, 26)
    seg6b = sample_segment(f5, m5, swoosh_f, min(len(f5) - 1, 68), 60)
    seg6 = seg6a + seg6b

    cta_trigger_t = vo6_start + vo_by_id["vo_06"]["words"][1]["start"]  # Exact millisecond "Play" begins!
    for i, (sf, rgb, fr) in enumerate(seg6):
        t_abs = b6_t + i / FPS
        # Clean, locked full-bleed napkin end card (zero jitter/drift!)
        cta_p = clamp((t_abs - cta_trigger_t) / 0.30, 0.0, 1.0)
        out_specs.append({
            "rgb": rgb, "zoom": 1.00, "cx": 360.0, "cy": 640.0,
            "free_beta_cta": cta_p,
        })
    audio_events.append({"kind": "el_sfx", "id": "fountain_scratch", "t": b6_t + 0.06, "maxDur": 0.72, "gain": 0.50})
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": b6_t + 2.45, "gain": 0.85})

    total_frames = len(out_specs)
    total_sec = total_frames / FPS
    print(f"Composing v2: {total_frames} frames ({total_sec:.2f}s @ {FPS}fps, {OUT_W}x{OUT_H}) with subpixel locked camera...")

    # Save exact timeline_v2.json for build_audio_v2.mjs!
    timeline = {
        "fps": FPS,
        "width": OUT_W,
        "height": OUT_H,
        "totalFrames": total_frames,
        "totalDurationSec": round(total_sec, 4),
        "musicCues": music_cues,
        "audioEvents": audio_events,
    }
    (EDIT_DIR / "timeline_v2.json").write_text(json.dumps(timeline, indent=2))

    silent_mp4 = EDIT_DIR / "video_v2_silent.mp4"
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

    # Flatten all active VO phrases for O(1)-ish frame lookup
    all_phrases = []
    for vp in vo_placements:
        all_phrases.extend(vp["phrases"])

    for idx, sp in enumerate(out_specs):
        t_now = idx / FPS
        im = subpixel_crop_and_scale(sp["rgb"], raw_w, raw_h, dpr, sp["zoom"], sp["cx"], sp["cy"])

        # 1. Render VO-synced word-highlighted caption if active
        rendered_vo_cap = False
        for ph in all_phrases:
            if ph["start"] <= t_now <= ph["end"]:
                fade_in = clamp((t_now - ph["start"]) / 0.10, 0.0, 1.0)
                fade_out = clamp((ph["end"] - t_now) / 0.12, 0.0, 1.0)
                alpha = min(fade_in, fade_out)
                w_states = []
                for w in ph["words"]:
                    if t_now < w["start"]:
                        st = "future"
                    elif t_now <= w["end"] + 0.06:
                        st = "active"
                    else:
                        st = "past"
                    w_states.append((w["word"], st))
                im = draw_word_synced_caption(im, w_states, alpha=alpha, y_center=ph["y"], accent=False)
                rendered_vo_cap = True
                break

        # 2. Render red SFX stamp captions (OOPS., EUREKA!!) when no VO is active
        if not rendered_vo_cap:
            for stp in stamp_captions:
                if stp["start"] <= t_now <= stp["end"]:
                    p_in = clamp((t_now - stp["start"]) / 0.12, 0.0, 1.0)
                    p_out = clamp((stp["end"] - t_now) / 0.10, 0.0, 1.0)
                    sc = 0.86 + 0.16 * ease_out_back(p_in)
                    im = draw_word_synced_caption(
                        im,
                        [(stp["text"], "active")],
                        alpha=min(clamp(p_in * 2.0, 0.0, 1.0), p_out),
                        y_center=stp["y"],
                        accent=stp["accent"],
                        scale=sc,
                    )
                    break

        # 3. Render PLAY FREE BETA NOW badge synced to "Play" in vo_06!
        if sp.get("free_beta_cta", 0.0) > 0.01:
            im = draw_free_beta_cta(im, sp["free_beta_cta"])

        ff.stdin.write(im.tobytes())
        if (idx + 1) % 120 == 0 or idx == total_frames - 1:
            print(f"  rendered {idx + 1}/{total_frames} frames...")

    ff.stdin.close()
    ret = ff.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {ret}")
    print(f"[VIDEO V2] Wrote {silent_mp4} ({total_sec:.2f}s)")


if __name__ == "__main__":
    build_v2()
