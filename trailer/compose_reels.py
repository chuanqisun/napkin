import json
import math
import os
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
EDIT_DIR = ROOT / "edit"
EDIT_DIR.mkdir(parents=True, exist_ok=True)

OUT_W, OUT_H = 1080, 1920
FPS = 30

FONT_BOLD = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
if not os.path.exists(FONT_BOLD):
    FONT_BOLD = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def ease_in_out(u):
    u = clamp(u, 0.0, 1.0)
    return 4 * u * u * u if u < 0.5 else 1 - pow(-2 * u + 2, 3) / 2


def ease_out_expo(u):
    u = clamp(u, 0.0, 1.0)
    return 1.0 if u >= 1.0 else 1.0 - pow(2.0, -10.0 * u)


def ease_out_back(u):
    u = clamp(u, 0.0, 1.0)
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * pow(u - 1.0, 3) + c1 * pow(u - 1.0, 2)


def load_video_frames(mp4_path: Path):
    """Decode raw 1440x2560 video into RGB bytes via ffmpeg pipe."""
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


def eval_keyframes(kfs, u_t):
    """Interpolate (t, zoom, cx, cy, ease_mode) keyframes in logical 720x1280 space."""
    if u_t <= kfs[0][0]:
        return kfs[0][1], kfs[0][2], kfs[0][3]
    for i in range(len(kfs) - 1):
        t0, z0, x0, y0, _ = kfs[i]
        t1, z1, x1, y1, mode = kfs[i + 1]
        if u_t <= t1:
            span = max(1e-6, t1 - t0)
            p = (u_t - t0) / span
            if mode == "crash":
                e = ease_out_expo(p)
            elif mode == "back":
                e = ease_out_back(p)
            elif mode == "linear":
                e = clamp(p, 0.0, 1.0)
            else:
                e = ease_in_out(p)
            return (
                z0 + (z1 - z0) * e,
                x0 + (x1 - x0) * e,
                y0 + (y1 - y0) * e,
            )
    return kfs[-1][1], kfs[-1][2], kfs[-1][3]


def crop_and_scale(raw_rgb: bytes, raw_w: int, raw_h: int, dpr: float, zoom: float, cx_log: float, cy_log: float, shake_x=0.0, shake_y=0.0):
    img = Image.frombytes("RGB", (raw_w, raw_h), raw_rgb)
    zoom = max(1.0, zoom)
    crop_w = raw_w / zoom
    crop_h = raw_h / zoom
    cx = cx_log * dpr + shake_x * dpr
    cy = cy_log * dpr + shake_y * dpr
    left = clamp(cx - crop_w * 0.5, 0.0, raw_w - crop_w)
    top = clamp(cy - crop_h * 0.5, 0.0, raw_h - crop_h)
    box = (int(round(left)), int(round(top)), int(round(left + crop_w)), int(round(top + crop_h)))
    cropped = img.crop(box)
    return cropped.resize((OUT_W, OUT_H), Image.Resampling.BICUBIC)


def draw_caption_pill(im: Image.Image, lines, alpha=1.0, y_center=1380, accent=False, scale=1.0):
    if alpha <= 0.01 or not lines:
        return im
    overlay = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = int(round((54 if accent else 46) * scale))
    font = ImageFont.truetype(FONT_BOLD, font_size)

    line_sizes = []
    max_w = 0
    total_h = 0
    line_gap = 12
    for ln in lines:
        bb = draw.textbbox((0, 0), ln, font=font)
        lw = bb[2] - bb[0]
        lh = bb[3] - bb[1]
        line_sizes.append((ln, lw, lh, bb[0], bb[1]))
        max_w = max(max_w, lw)
        total_h += lh
    total_h += line_gap * (len(lines) - 1)

    pad_x = 44 if accent else 36
    pad_y = 24 if accent else 20
    box_w = max_w + pad_x * 2
    box_h = total_h + pad_y * 2
    x0 = (OUT_W - box_w) // 2
    y0 = int(y_center - box_h // 2)
    x1 = x0 + box_w
    y1 = y0 + box_h

    # Warm espresso shadow + crisp rounded pill
    sh_a = int(90 * alpha)
    draw.rounded_rectangle([x0 + 4, y0 + 10, x1 + 4, y1 + 10], radius=24, fill=(32, 22, 12, sh_a))

    if accent:
        bg = (185, 28, 28, int(242 * alpha))
        border = (255, 245, 230, int(220 * alpha))
        fg = (255, 250, 242, int(255 * alpha))
    else:
        bg = (31, 27, 23, int(232 * alpha))
        border = (238, 226, 206, int(90 * alpha))
        fg = (250, 244, 234, int(255 * alpha))

    draw.rounded_rectangle([x0, y0, x1, y1], radius=22, fill=bg, outline=border, width=3)

    cur_y = y0 + pad_y
    for ln, lw, lh, ox, oy in line_sizes:
        tx = (OUT_W - lw) // 2 - ox
        ty = cur_y - oy
        draw.text((tx, ty), ln, font=font, fill=fg)
        cur_y += lh + line_gap

    return Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")


def draw_steam_cta_badge(im: Image.Image, progress: float):
    if progress <= 0.01:
        return im
    e = ease_out_back(min(1.0, progress))
    alpha = clamp(progress * 2.2, 0.0, 1.0)
    overlay = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_main = ImageFont.truetype(FONT_BOLD, 46)
    font_sub = ImageFont.truetype(FONT_BOLD, 26)
    label = "WISHLIST ON STEAM"
    bb = draw.textbbox((0, 0), label, font=font_main)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]

    icon_r = 28
    gap = 20
    pad_x = 46
    pad_y = 28
    box_w = icon_r * 2 + gap + tw + pad_x * 2
    box_h = max(icon_r * 2, th) + pad_y * 2

    y_slide = int((1.0 - e) * 34)
    y_center = 1170 + y_slide
    x0 = (OUT_W - box_w) // 2
    y0 = y_center - box_h // 2
    x1 = x0 + box_w
    y1 = y0 + box_h

    # Soft shadow + deep warm charcoal Steam pill
    draw.rounded_rectangle([x0 + 5, y0 + 12, x1 + 5, y1 + 12], radius=34, fill=(40, 28, 16, int(95 * alpha)))
    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=32,
        fill=(28, 25, 22, int(245 * alpha)),
        outline=(245, 236, 220, int(170 * alpha)),
        width=3,
    )

    # Clean geometric Steam-inspired crank emblem inside circle
    icx = x0 + pad_x + icon_r
    icy = y_center
    cream = (248, 243, 234, int(255 * alpha))
    dark = (28, 25, 22, int(255 * alpha))
    draw.ellipse([icx - icon_r, icy - icon_r, icx + icon_r, icy + icon_r], fill=cream)
    # Minimal steam linkage dots & bar
    draw.line([icx - 16, icy + 8, icx + 6, icy - 6], fill=dark, width=7)
    draw.ellipse([icx + 0, icy - 16, icx + 18, icy + 2], fill=dark)
    draw.ellipse([icx + 4, icy - 12, icx + 14, icy - 2], fill=cream)
    draw.ellipse([icx - 15, icy + 3, icx - 1, icy + 17], fill=dark)

    tx = icx + icon_r + gap - bb[0]
    ty = y_center - th // 2 - bb[1]
    draw.text((tx, ty), label, font=font_main, fill=cream)

    # Subtle charming subtitle beneath pill (still cleanly above Reels 1500px safe line!)
    sub = "Your next big idea, served hot."
    sbb = draw.textbbox((0, 0), sub, font=font_sub)
    sw = sbb[2] - sbb[0]
    draw.text(
        ((OUT_W - sw) // 2 - sbb[0], y1 + 26 - sbb[1]),
        sub,
        font=font_sub,
        fill=(62, 48, 34, int(215 * alpha)),
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
    """Map out_count output frames smoothly across [src_start, src_end] of raw take."""
    n_src = len(frames_list)
    out = []
    for i in range(out_count):
        u = i / max(1, out_count - 1) if out_count > 1 else 0.0
        sf = int(round(src_start + (src_end - src_start) * u))
        sf = clamp(sf, 0, n_src - 1)
        out.append((sf, frames_list[sf], meta["frames"][sf]))
    return out


def build_reels():
    print("Loading 5 raw 1440x2560 takes...")
    f1, m1 = load_video_frames(RAW_DIR / "01_pen_and_idea.mp4")
    f2, m2 = load_video_frames(RAW_DIR / "02_block_order_spill.mp4")
    f3, m3 = load_video_frames(RAW_DIR / "03_crumple_reset.mp4")
    f4, m4 = load_video_frames(RAW_DIR / "04_matcha_eureka.mp4")
    f5, m5 = load_video_frames(RAW_DIR / "05_endcard_napkin.mp4")

    raw_w, raw_h, dpr = m1["width"], m1["height"], m1["dpr"]
    print(f"Loaded takes: {[len(x) for x in [f1, f2, f3, f4, f5]]} frames @ {raw_w}x{raw_h}")

    # Timeline output collector: list of dicts describing each rendered frame
    out_specs = []
    audio_events = []
    music_cues = {}

    def cur_t():
        return len(out_specs) / FPS

    # =========================================================================
    # BEAT 1: SELECTING PEN + AUTO-BEAUTIFY "Big Idea ↓ Magic" (~3.9s = 117f)
    # =========================================================================
    b1_start_t = cur_t()
    pen_open = marker_frame(m1, "pen_menu_open", 10)
    ftn_sel = marker_frame(m1, "fountain_selected", 31)
    pen_close = marker_frame(m1, "pen_menu_closed", 47)

    # Sub-seg 1A: Pen menu opens -> skips Ballpoint -> clicks Fountain Pen (38f = 1.27s)
    seg1a = sample_segment(f1, m1, max(0, pen_open - 4), pen_close, 38)
    for i, (sf, rgb, fr) in enumerate(seg1a):
        u = i / max(1, len(seg1a) - 1)
        # Tight zoom on Pen & Writing Style popover with micro punch on Fountain Pen click!
        z, cx, cy = eval_keyframes([
            (0.00, 1.48, 220, 265, "ease"),
            (0.50, 1.56, 200, 270, "ease"),
            (0.62, 1.72, 195, 295, "crash"),
            (0.85, 1.52, 220, 320, "ease"),
            (1.00, 1.38, 290, 390, "ease"),
        ], u)
        out_specs.append({
            "rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "sx": 0.0, "sy": 0.0,
            "caption": ["Billion-dollar idea.", "Step 1: ridiculously fancy pen."],
            "cap_alpha": min(1.0, (i + 4) / 8.0),
            "cap_y": 1400,
        })
    audio_events.append({"kind": "click", "t": b1_start_t + 0.14, "gain": 0.28})
    audio_events.append({"kind": "click", "t": b1_start_t + 0.72, "gain": 0.32})
    audio_events.append({"kind": "whoosh", "t": b1_start_t + 1.15, "dur": 0.16, "gain": 0.18})

    # Sub-seg 1B: Kinetic sketch oval -> SNAP -> Big Idea -> arrow -> box -> SNAP -> Magic (78f = 2.60s)
    b1b_t = cur_t()
    seg1b = sample_segment(f1, m1, pen_close, len(f1) - 1, 78)
    smooth_x, smooth_y = 335.0, 420.0
    for i, (sf, rgb, fr) in enumerate(seg1b):
        u = i / max(1, len(seg1b) - 1)
        # Smooth spring-damped cursor-follow blended with vertical composition anchor!
        target_x = 340.0 * 0.55 + fr["cx"] * 0.45
        target_y = (420.0 + 230.0 * ease_in_out(u)) * 0.65 + fr["cy"] * 0.35
        smooth_x += (target_x - smooth_x) * 0.18
        smooth_y += (target_y - smooth_y) * 0.18
        z = 1.35 + 0.10 * ease_in_out(u)
        out_specs.append({
            "rgb": rgb, "zoom": z, "cx": smooth_x, "cy": smooth_y, "sx": 0.0, "sy": 0.0,
            "caption": ["Billion-dollar idea.", "Step 1: ridiculously fancy pen."],
            "cap_alpha": 1.0 if i < 70 else max(0.0, (78 - i) / 8.0),
            "cap_y": 1400,
        })
    # Fountain-pen ASMR scratches synced cleanly to oval, typing Big Idea, arrow, box, Magic!
    audio_events.append({"kind": "scratch", "t": b1b_t + 0.08, "dur": 0.34, "style": "fountain", "gain": 0.28})
    audio_events.append({"kind": "scratch", "t": b1b_t + 0.75, "dur": 0.36, "style": "fountain", "gain": 0.22})
    audio_events.append({"kind": "scratch", "t": b1b_t + 1.25, "dur": 0.20, "style": "fountain", "gain": 0.24})
    audio_events.append({"kind": "scratch", "t": b1b_t + 1.58, "dur": 0.34, "style": "fountain", "gain": 0.28})
    audio_events.append({"kind": "scratch", "t": b1b_t + 2.22, "dur": 0.28, "style": "fountain", "gain": 0.22})

    # =========================================================================
    # BEAT 2: WRITER'S BLOCK -> ORDER COFFEE LATTE (~2.9s = 87f)
    # =========================================================================
    b2_t = cur_t()
    undo_f = marker_frame(m2, "undo_block", 32)
    menu_open_f = marker_frame(m2, "drink_menu_open", 58)
    latte_ord_f = marker_frame(m2, "latte_ordered", 68)
    latte_land_f = marker_frame(m2, "latte_landed", 88)
    sip_click_f = marker_frame(m2, "sip_clicked", 97)
    splat_f = marker_frame(m2, "splat_hit", 142)
    mug_ret_f = marker_frame(m2, "mug_returned", 194)
    ring_f = marker_frame(m2, "coffee_ring_revealed", 224)

    splat_m = marker_obj(m2, "splat_hit") or {"x": 336, "y": 649}
    ring_m = marker_obj(m2, "coffee_ring_revealed") or {"x": 585, "y": 330}

    # Sub-seg 2A: Messy squiggle -> Undo -> weary sigh (35f = 1.17s)
    seg2a = sample_segment(f2, m2, 0, undo_f + 12, 35)
    for i, (sf, rgb, fr) in enumerate(seg2a):
        u = i / max(1, len(seg2a) - 1)
        z, cx, cy = eval_keyframes([
            (0.0, 1.32, 340, 680, "ease"),
            (0.6, 1.22, 360, 560, "ease"),
            (1.0, 1.15, 360, 480, "ease"),
        ], u)
        out_specs.append({
            "rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "sx": 0.0, "sy": 0.0,
            "caption": None, "cap_alpha": 0.0, "cap_y": 1400,
        })
    audio_events.append({"kind": "scratch", "t": b2_t + 0.08, "dur": 0.30, "style": "fountain", "gain": 0.22})
    audio_events.append({"kind": "click", "t": b2_t + 0.65, "gain": 0.26})
    audio_events.append({"kind": "clip", "id": "sigh_block", "t": b2_t + 0.72, "maxDur": 1.15, "gain": 0.85})

    # Sub-seg 2B: Whip-pan to pulsing amber Menu -> Coffee Latte glides in & clinks (52f = 1.73s)
    b2b_t = cur_t()
    seg2b = sample_segment(f2, m2, undo_f + 12, sip_click_f - 4, 52)
    for i, (sf, rgb, fr) in enumerate(seg2b):
        u = i / max(1, len(seg2b) - 1)
        z, cx, cy = eval_keyframes([
            (0.00, 1.15, 360, 460, "ease"),
            (0.28, 1.52, 240, 235, "crash"),  # Frame ALL 4 drinks in Café Menu grid cleanly!
            (0.62, 1.52, 235, 240, "linear"),
            (0.92, 1.06, 390, 440, "ease"),   # Smooth pull-out as Latte glides onto corner!
            (1.00, 1.06, 390, 440, "linear"),
        ], u)
        out_specs.append({
            "rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "sx": 0.0, "sy": 0.0,
            "caption": ["Step 2: caffeine."],
            "cap_alpha": clamp(i / 6.0, 0.0, 1.0),
            "cap_y": 1445,
        })
    audio_events.append({"kind": "whoosh", "t": b2b_t + 0.06, "dur": 0.14, "gain": 0.20})
    audio_events.append({"kind": "click", "t": b2b_t + 0.42, "gain": 0.30})
    audio_events.append({"kind": "click", "t": b2b_t + 0.78, "gain": 0.30})
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": b2b_t + 1.25, "gain": 1.05})

    # =========================================================================
    # BEAT 3: SPILL / DRIP — HERO COMEDY CRASH ZOOM (~4.5s = 135f)
    # =========================================================================
    b3_t = cur_t()
    # Sub-seg 3A: Click Latte -> mug lifts -> loud slurp "Aaaahhh..." right up to SPLAT! (42f = 1.40s)
    seg3a = sample_segment(f2, m2, sip_click_f - 4, splat_f, 42)
    for i, (sf, rgb, fr) in enumerate(seg3a):
        u = i / max(1, len(seg3a) - 1)
        z, cx, cy = eval_keyframes([
            (0.0, 1.06, 390, 440, "ease"),
            (1.0, 1.18, 360, 560, "ease"),
        ], u)
        out_specs.append({
            "rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "sx": 0.0, "sy": 0.0,
            "caption": ["Step 2: caffeine."],
            "cap_alpha": max(0.0, 1.0 - i / 14.0),
            "cap_y": 1400,
        })
    audio_events.append({"kind": "click", "t": b3_t + 0.12, "gain": 0.25})
    audio_events.append({"kind": "clip", "id": "sip_ahhh", "t": b3_t + 0.35, "maxDur": 1.08, "gain": 1.15})

    # Sub-seg 3B: SPLAT!! 5-FRAME 2.55x CRASH ZOOM onto Magic + DEAD SILENCE + "Oops!" (40f = 1.33s)
    splat_out_t = cur_t()
    music_cues["splatCutStart"] = splat_out_t
    seg3b = sample_segment(f2, m2, splat_f, mug_ret_f - 10, 40)
    for i, (sf, rgb, fr) in enumerate(seg3b):
        u = i / max(1, len(seg3b) - 1)
        z, cx, cy = eval_keyframes([
            (0.00, 1.18, 360, 560, "linear"),
            (0.14, 2.55, splat_m["x"] - 10, splat_m["y"] - 4, "crash"),  # 5-frame violent crash zoom!
            (1.00, 2.68, splat_m["x"] - 10, splat_m["y"] - 4, "linear"),  # slow comedic creep!
        ], u)
        # Tiny 4px jolt on splat impact frames
        sx = math.sin(i * 2.5) * max(0.0, 5.0 - i * 0.9)
        sy = math.cos(i * 3.1) * max(0.0, 4.0 - i * 0.8)
        oops_alpha = clamp((i - 6) / 4.0, 0.0, 1.0)
        oops_scale = 0.85 + 0.15 * ease_out_back(clamp((i - 6) / 7.0, 0.0, 1.0))
        out_specs.append({
            "rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "sx": sx, "sy": sy,
            "caption": ["OOPS."],
            "cap_alpha": oops_alpha,
            "cap_y": 1350,
            "cap_accent": True,
            "cap_scale": oops_scale,
        })
    audio_events.append({"kind": "splat", "t": splat_out_t, "gain": 0.52})
    audio_events.append({"kind": "whoosh", "t": splat_out_t, "dur": 0.13, "gain": 0.28})
    audio_events.append({"kind": "clip", "id": "oops_drip", "t": splat_out_t + 0.24, "gain": 1.05})

    # Sub-seg 3C: Mug returns -> slow defeated drag reveals DARK COFFEE RING -> second snap zoom! (53f = 1.77s)
    b3c_t = cur_t()
    seg3c = sample_segment(f2, m2, mug_ret_f - 10, len(f2) - 1, 53)
    for i, (sf, rgb, fr) in enumerate(seg3c):
        u = i / max(1, len(seg3c) - 1)
        z, cx, cy = eval_keyframes([
            (0.00, 1.25, 440, 460, "ease"),
            (0.55, 1.20, 470, 430, "ease"),
            (0.68, 1.92, ring_m["x"] - 20, ring_m["y"] + 25, "crash"),  # Snap punch-in on fresh coffee ring!
            (1.00, 1.98, ring_m["x"] - 20, ring_m["y"] + 25, "linear"),
        ], u)
        out_specs.append({
            "rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "sx": 0.0, "sy": 0.0,
            "caption": ["…step 3: ruin it."],
            "cap_alpha": clamp((i - 6) / 6.0, 0.0, 1.0),
            "cap_y": 1455,
        })
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": b3c_t + 0.10, "gain": 0.85})
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": b3c_t + 1.08, "gain": 0.98})
    audio_events.append({"kind": "whoosh", "t": b3c_t + 1.08, "dur": 0.12, "gain": 0.22})

    # =========================================================================
    # BEAT 4: TRASH AN IDEA — MUG LIFTS -> 3D MESH CRUMPLE + TOSS (~3.3s = 98f)
    # =========================================================================
    b4_t = cur_t()
    trash_f = marker_frame(m3, "trash_clicked", 16)
    crumple_start_f = marker_frame(m3, "crumple_start", 29)
    crumple_done_f = marker_frame(m3, "crumple_done", 73)
    trash_m = marker_obj(m3, "trash_clicked") or {"x": 490, "y": 75}

    seg4 = sample_segment(f3, m3, 0, len(f3) - 1, 98)
    for i, (sf, rgb, fr) in enumerate(seg4):
        u = i / max(1, len(seg4) - 1)
        z, cx, cy = eval_keyframes([
            (0.00, 1.15, 420, 440, "ease"),
            (0.16, 1.50, trash_m["x"] - 10, trash_m["y"] + 160, "crash"),  # Cursor dart to Trash!
            (0.28, 1.05, 360, 620, "ease"),  # Pull wide so full 3D crumple mesh & mug lift shine!
            (0.75, 1.05, 395, 620, "linear"),  # Subtle follow-pan right as ball yeets off!
            (0.90, 1.02, 360, 620, "ease"),
            (1.00, 1.02, 360, 620, "linear"),
        ], u)
        # Camera impact micro-shake synced to exact 3D crumple crunch peaks!
        sx, sy = 0.0, 0.0
        if crumple_start_f <= sf <= crumple_start_f + 32:
            cf = sf - crumple_start_f
            # Crunch peaks at ~80ms (f2), 230ms (f7), 730ms (f22), 890ms (f27)
            amp = (
                math.exp(-pow((cf - 2.5) / 2.0, 2)) * 7.5
                + math.exp(-pow((cf - 7.5) / 2.2, 2)) * 5.5
                + math.exp(-pow((cf - 22.0) / 2.4, 2)) * 6.0
                + math.exp(-pow((cf - 27.0) / 2.0, 2)) * 4.5
            )
            sx = math.sin(cf * 2.7) * amp
            sy = math.cos(cf * 3.4) * amp

        cap = ["Good thing there's", "a whole stack."] if sf >= crumple_done_f - 4 else None
        cap_a = clamp((sf - (crumple_done_f - 4)) / 6.0, 0.0, 1.0) if cap else 0.0
        out_specs.append({
            "rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "sx": sx, "sy": sy,
            "caption": cap, "cap_alpha": cap_a, "cap_y": 1400,
        })
    audio_events.append({"kind": "click", "t": b4_t + trash_f / FPS, "gain": 0.32})
    audio_events.append({"kind": "clip", "id": "crumble", "t": b4_t + crumple_start_f / FPS, "gain": 1.25})
    audio_events.append({"kind": "clip", "id": "argh_stain", "t": b4_t + (crumple_start_f + 4) / FPS, "gain": 0.92})
    audio_events.append({"kind": "whoosh", "t": b4_t + (crumple_start_f + 27) / FPS, "dur": 0.22, "gain": 0.26})
    clink_fresh_t = b4_t + 2.73
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": clink_fresh_t, "gain": 1.05})
    music_cues["musicResume"] = clink_fresh_t - 0.04  # Upbeat groove kicks back in on the fresh-napkin clink!

    # =========================================================================
    # BEAT 5: MATCHA POP + VERTICAL "Weird Thought ↓ Napkin ↓ Ship It!" + EUREKA! (~4.4s = 132f)
    # =========================================================================
    b5_t = cur_t()
    matcha_land_f = marker_frame(m4, "matcha_landed", 44)
    aha_f = marker_frame(m4, "aha_spark", 176)
    eureka_f = marker_frame(m4, "eureka_hit", 247)
    eureka_m = marker_obj(m4, "eureka_hit") or {"x": 320, "y": 845}

    # Sub-seg 5A: Quick Matcha swap flash (26f = 0.87s)
    seg5a = sample_segment(f4, m4, 0, matcha_land_f, 26)
    for i, (sf, rgb, fr) in enumerate(seg5a):
        u = i / max(1, len(seg5a) - 1)
        z, cx, cy = eval_keyframes([
            (0.00, 1.05, 360, 480, "ease"),
            (0.35, 1.55, 360, 235, "crash"),
            (0.95, 1.15, 440, 320, "ease"),
            (1.00, 1.15, 440, 320, "linear"),
        ], u)
        out_specs.append({
            "rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "sx": 0.0, "sy": 0.0,
            "caption": ["Good thing there's", "a whole stack."],
            "cap_alpha": max(0.0, 1.0 - i / 10.0),
            "cap_y": 1400,
        })
    audio_events.append({"kind": "click", "t": b5_t + 0.14, "gain": 0.25})
    audio_events.append({"kind": "click", "t": b5_t + 0.36, "gain": 0.28})
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": b5_t + 0.78, "gain": 0.98})

    # Sub-seg 5B: Kinetic flow-state vertical sketch (64f = 2.13s, crisp ~3.1x energy ramp!)
    b5b_t = cur_t()
    seg5b = sample_segment(f4, m4, matcha_land_f, eureka_f, 64)
    sx_f, sy_f = 300.0, 420.0
    for i, (sf, rgb, fr) in enumerate(seg5b):
        u = i / max(1, len(seg5b) - 1)
        # Smooth vertical tilt down tracking Weird Thought -> Napkin -> Ship It!
        tx = 305.0 * 0.65 + fr["cx"] * 0.35
        ty = (400.0 + 465.0 * ease_in_out(u)) * 0.70 + fr["cy"] * 0.30
        sx_f += (tx - sx_f) * 0.20
        sy_f += (ty - sy_f) * 0.20
        z = 1.28 + 0.12 * u
        out_specs.append({
            "rgb": rgb, "zoom": z, "cx": sx_f, "cy": sy_f, "sx": 0.0, "sy": 0.0,
            "caption": None, "cap_alpha": 0.0, "cap_y": 1400,
        })
    # Accelerating ballpoint -> marker squeaks + "Aha!" on red arrow!
    audio_events.append({"kind": "scratch", "t": b5b_t + 0.10, "dur": 0.24, "style": "fountain", "gain": 0.22})
    audio_events.append({"kind": "scratch", "t": b5b_t + 0.55, "dur": 0.24, "style": "fountain", "gain": 0.22})
    audio_events.append({"kind": "scratch", "t": b5b_t + 1.00, "dur": 0.22, "style": "marker", "gain": 0.26})
    aha_out_t = b5b_t + ((aha_f - matcha_land_f) / max(1, eureka_f - matcha_land_f)) * (len(seg5b) / FPS)
    audio_events.append({"kind": "clip", "id": "aha_spark", "t": aha_out_t - 0.12, "maxDur": 0.75, "gain": 0.80})
    audio_events.append({"kind": "scratch", "t": b5b_t + 1.55, "dur": 0.28, "style": "marker", "gain": 0.30})
    audio_events.append({"kind": "scratch", "t": b5b_t + 1.88, "dur": 0.22, "style": "marker", "gain": 0.32})

    # Sub-seg 5C: HERO EUREKA CLIMAX! 5-frame Crash Zoom on "Ship It!" -> smooth pull-out (42f = 1.40s)
    eureka_out_t = cur_t()
    music_cues["eurekaHit"] = eureka_out_t
    seg5c = sample_segment(f4, m4, eureka_f, min(len(f4) - 1, eureka_f + 46), 42)
    for i, (sf, rgb, fr) in enumerate(seg5c):
        u = i / max(1, len(seg5c) - 1)
        z, cx, cy = eval_keyframes([
            (0.00, 1.38, 320, 845, "linear"),
            (0.14, 1.98, eureka_m["x"], eureka_m["y"] + 8, "back"),  # Hero punch-bounce crash zoom!
            (0.55, 1.92, eureka_m["x"], eureka_m["y"] + 8, "linear"),
            (0.92, 1.02, 360, 620, "ease"),   # Majestic pull-out showing whole napkin + green Matcha!
            (1.00, 1.02, 360, 620, "linear"),
        ], u)
        sx = math.sin(i * 2.9) * max(0.0, 4.5 - i * 0.9)
        sy = math.cos(i * 3.2) * max(0.0, 3.5 - i * 0.8)
        if i < 24:
            cap = ["EUREKA!!"]
            cap_a = clamp(i / 3.0, 0.0, 1.0)
            cap_accent = True
            cap_scale = 0.86 + 0.16 * ease_out_back(clamp(i / 6.0, 0.0, 1.0))
        else:
            cap = ["There it is."]
            cap_a = clamp((i - 24) / 5.0, 0.0, 1.0)
            cap_accent = False
            cap_scale = 1.0
        out_specs.append({
            "rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "sx": sx, "sy": sy,
            "caption": cap, "cap_alpha": cap_a, "cap_y": 1390,
            "cap_accent": cap_accent, "cap_scale": cap_scale,
        })
    audio_events.append({"kind": "whoosh", "t": eureka_out_t, "dur": 0.14, "gain": 0.25})
    audio_events.append({"kind": "clip", "id": "eureka_win", "t": eureka_out_t + 0.02, "maxDur": 1.35, "gain": 1.05})

    # =========================================================================
    # BEAT 6: FULL-BLEED SUNLIT NAPKIN END CARD + STEAM CTA (~2.4s = 72f)
    # =========================================================================
    b6_t = cur_t()
    music_cues["endCardStart"] = b6_t
    swoosh_f = marker_frame(m5, "swoosh_done", 37)

    # First 26f: live-writing "Napkin" + red swoosh; Next 46f: crisp Steam CTA badge + rock-solid Latte hold!
    seg6a = sample_segment(f5, m5, 3, swoosh_f, 26)
    seg6b = sample_segment(f5, m5, swoosh_f, min(len(f5) - 1, 68), 46)
    seg6 = seg6a + seg6b
    for i, (sf, rgb, fr) in enumerate(seg6):
        u = i / max(1, len(seg6) - 1)
        z, cx, cy = eval_keyframes([
            (0.00, 1.09, 360, 580, "ease"),
            (0.55, 1.00, 360, 640, "ease"),
            (1.00, 1.00, 360, 640, "linear"),
        ], u)
        cta_p = clamp((i - 22) / 10.0, 0.0, 1.0)
        out_specs.append({
            "rgb": rgb, "zoom": z, "cx": cx, "cy": cy, "sx": 0.0, "sy": 0.0,
            "caption": None, "cap_alpha": 0.0, "cap_y": 1400,
            "steam_cta": cta_p,
        })
    audio_events.append({"kind": "scratch", "t": b6_t + 0.08, "dur": 0.50, "style": "fountain", "gain": 0.26})
    audio_events.append({"kind": "scratch", "t": b6_t + 0.62, "dur": 0.22, "style": "fountain", "gain": 0.28})
    audio_events.append({"kind": "clip", "id": "cup_setdown", "t": b6_t + 1.85, "gain": 0.78})

    total_frames = len(out_specs)
    total_sec = total_frames / FPS
    print(f"Composing {total_frames} frames ({total_sec:.2f}s @ {FPS}fps, {OUT_W}x{OUT_H})...")

    # Save exact timeline.json for build_audio.mjs!
    timeline = {
        "fps": FPS,
        "width": OUT_W,
        "height": OUT_H,
        "totalFrames": total_frames,
        "totalDurationSec": round(total_sec, 4),
        "musicCues": music_cues,
        "audioEvents": audio_events,
    }
    (EDIT_DIR / "timeline.json").write_text(json.dumps(timeline, indent=2))

    # Stream composed 1080x1920 frames directly into ffmpeg libx264!
    silent_mp4 = EDIT_DIR / "video_silent.mp4"
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
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(silent_mp4),
    ], stdin=subprocess.PIPE)

    for idx, sp in enumerate(out_specs):
        im = crop_and_scale(sp["rgb"], raw_w, raw_h, dpr, sp["zoom"], sp["cx"], sp["cy"], sp["sx"], sp["sy"])
        if sp.get("caption") and sp.get("cap_alpha", 0.0) > 0.01:
            im = draw_caption_pill(
                im,
                sp["caption"],
                alpha=sp["cap_alpha"],
                y_center=sp.get("cap_y", 1400),
                accent=sp.get("cap_accent", False),
                scale=sp.get("cap_scale", 1.0),
            )
        if sp.get("steam_cta", 0.0) > 0.01:
            im = draw_steam_cta_badge(im, sp["steam_cta"])
        ff.stdin.write(im.tobytes())
        if (idx + 1) % 100 == 0 or idx == total_frames - 1:
            print(f"  rendered {idx + 1}/{total_frames} frames...")

    ff.stdin.close()
    ret = ff.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {ret}")
    print(f"[VIDEO] Wrote {silent_mp4} ({total_sec:.2f}s)")


if __name__ == "__main__":
    build_reels()
