import math
import os
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RAW_DIR = ROOT / "raw"
EDIT_DIR = ROOT / "edit"
EDIT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1080, 1920
FONT_BOLD = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
if not os.path.exists(FONT_BOLD):
    FONT_BOLD = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"


def extract_frame(mp4_path: Path, frame_idx: int) -> Image.Image:
    tmp = EDIT_DIR / f"_tmp_{mp4_path.stem}_{frame_idx}.png"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-i", str(mp4_path),
            "-vf", f"select='eq(n,{frame_idx})'",
            "-frames:v", "1", "-vsync", "vfr",
            str(tmp),
        ],
        check=True,
    )
    im = Image.open(tmp).convert("RGBA")
    tmp.unlink(missing_ok=True)
    return im


def add_drop_shadow(sprite: Image.Image, offset=(16, 22), blur=22, opacity=0.38, pad=60):
    sw, sh = sprite.size
    canvas = Image.new("RGBA", (sw + pad * 2, sh + pad * 2), (0, 0, 0, 0))
    alpha = sprite.split()[3]
    sh_alpha = alpha.point(lambda a: int(a * opacity))
    shadow = Image.new("RGBA", (sw, sh), (44, 28, 14, 255))
    shadow.putalpha(sh_alpha)
    canvas.paste(shadow, (pad + offset[0], pad + offset[1]))
    canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
    canvas.alpha_composite(sprite, (pad, pad))
    return canvas


def paste_multiply_region(dst_rgb: Image.Image, src_rgba: Image.Image, box, dst_pos, target_w, boost_ink=1.18):
    """Extract a clean ink/stain region from a napkin capture and multiply-blend it onto our cover napkin."""
    crop = src_rgba.crop(box).convert("RGB")
    scale = target_w / crop.width
    th = int(round(crop.height * scale))
    crop = crop.resize((target_w, th), Image.Resampling.LANCZOS)

    # Normalize local paper background to pure white (255,255,255) so ONLY ink & coffee stains multiply cleanly!
    px = crop.load()
    # Sample corner paper median as background reference
    corners = [px[4, 4], px[target_w - 5, 4], px[4, th - 5], px[target_w - 5, th - 5]]
    bg_r = sum(c[0] for c in corners) / 4.0
    bg_g = sum(c[1] for c in corners) / 4.0
    bg_b = sum(c[2] for c in corners) / 4.0

    # Smooth feather mask around edges so borders blend invisibly
    norm = Image.new("RGB", (target_w, th), (255, 255, 255))
    npx = norm.load()
    edge = 18
    for y in range(th):
        fy = min(1.0, y / edge, (th - 1 - y) / edge)
        for x in range(target_w):
            fx = min(1.0, x / edge, (target_w - 1 - x) / edge)
            f = fx * fy
            r, g, b = px[x, y]
            # Divide out paper color -> paper becomes ~1.0, ink & stains stay rich!
            nr = min(1.0, (r / max(1.0, bg_r)) ** boost_ink)
            ng = min(1.0, (g / max(1.0, bg_g)) ** boost_ink)
            nb = min(1.0, (b / max(1.0, bg_b)) ** boost_ink)
            # Lift near-white paper grain cleanly to 1.0
            if nr > 0.93 and ng > 0.93 and nb > 0.93:
                nr = ng = nb = 1.0
            rr = int(round((1.0 - f * (1.0 - nr)) * 255))
            gg = int(round((1.0 - f * (1.0 - ng)) * 255))
            bb = int(round((1.0 - f * (1.0 - nb)) * 255))
            npx[x, y] = (rr, gg, bb)

    # Multiply onto destination region
    dx, dy = dst_pos
    sub = dst_rgb.crop((dx, dy, dx + target_w, dy + th))
    blended = ImageChops.multiply(sub, norm)
    dst_rgb.paste(blended, (dx, dy))


def draw_stamp(base_rgba: Image.Image, text: str, center_xy, angle_deg: float, bg_col, border_col, txt_col, font_size=36):
    font = ImageFont.truetype(FONT_BOLD, font_size)
    tmp = Image.new("RGBA", (420, 140), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bb = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    px, py = 26, 14
    bw, bh = tw + px * 2, th + py * 2
    x0, y0 = (420 - bw) // 2, (140 - bh) // 2
    d.rounded_rectangle([x0 + 3, y0 + 6, x0 + bw + 3, y0 + bh + 6], radius=16, fill=(35, 22, 10, 75))
    d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=15, fill=bg_col, outline=border_col, width=3)
    d.text((x0 + px - bb[0], y0 + py - bb[1]), text, font=font, fill=txt_col)
    rot = tmp.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=True)
    rw, rh = rot.size
    base_rgba.alpha_composite(rot, (int(center_xy[0] - rw // 2), int(center_xy[1] - rh // 2)))


def build_cover():
    print("Building 1080x1920 Instagram Reels Cover from real Napkin game assets...")

    # 1. Warm sunlit café tabletop background (gradient matching Napkin's authentic palette)
    bg = Image.new("RGB", (W, H), (206, 187, 159))
    bpx = bg.load()
    for y in range(H):
        for x in range(W):
            dx = (x - W * 0.48) / (W * 0.85)
            dy = (y - H * 0.42) / (H * 0.65)
            dist = math.hypot(dx, dy)
            t = min(1.0, dist)
            # Warm sunlight center -> rich café oak vignette edges
            r = int(223 - t * 38)
            g = int(206 - t * 42)
            b = int(179 - t * 46)
            bpx[x, y] = (r, g, b)

    # 2. Extract real high-res kraft napkin sheet from Take 5 (clean paper grain + emboss + crease!)
    t5_clean = extract_frame(RAW_DIR / "05_endcard_napkin.mp4", 60)  # 1440x2560
    t2_splat = extract_frame(RAW_DIR / "02_block_order_spill.mp4", 160)
    t4_eureka = extract_frame(RAW_DIR / "04_matcha_eureka.mp4", 275)

    # Build a crisp upright kraft paper napkin card (900x1320) from the clean bottom paper region (zero swoosh sliver!)
    paper_base = t5_clean.crop((40, 1360, 1380, 2500)).convert("RGB")
    nap_w, nap_h = 900, 1330
    napkin = paper_base.resize((nap_w, nap_h), Image.Resampling.LANCZOS)

    # Subtle embossed stitched double-border + center fold crease on the hero napkin!
    nd = ImageDraw.Draw(napkin, "RGBA")
    nd.rectangle([24, 24, nap_w - 24, nap_h - 24], outline=(112, 86, 56, 42), width=2)
    nd.rectangle([32, 32, nap_w - 32, nap_h - 32], outline=(112, 86, 56, 28), width=1)
    # Soft vertical & horizontal napkin fold crease
    for k in range(-18, 19):
        a = int(14 * math.exp(-((k / 8.0) ** 2)))
        col = (90, 66, 38, a) if k < 0 else (255, 250, 240, int(a * 1.15))
        nd.line([(nap_w // 2 + k, 0), (nap_w // 2 + k, nap_h)], fill=col, width=1)
        nd.line([(0, nap_h // 2 + k), (nap_w, nap_h // 2 + k)], fill=col, width=1)

    # 3. Multiply-blend REAL game artwork directly onto our hero napkin (zero overlap with CTA pill!)
    # 3A. Big iconic live-written "Napkin" + red swoosh from Take 5!
    paste_multiply_region(
        napkin,
        t5_clean,
        box=(260, 950, 1180, 1345),
        dst_pos=(115, 118),
        target_w=660,
        boost_ink=1.22,
    )

    # 3B. Real "Big Idea ↓ [ ? + coffee splat! ]" from Take 2!
    paste_multiply_region(
        napkin,
        t2_splat,
        box=(320, 710, 1020, 1445),
        dst_pos=(210, 405),
        target_w=440,
        boost_ink=1.22,
    )

    # 3C. Bold Red Felt-Tip Marker arrow + "Ship It!" + double underlines from Take 4!
    # Cleanly remove the small crosshair cursor at bottom-right of the second underline before multiply-blending
    t4_clean = t4_eureka.copy()
    t4_bg = t4_clean.getpixel((860, 1950))
    ImageDraw.Draw(t4_clean).rectangle([885, 1870, 975, 1960], fill=t4_bg)
    paste_multiply_region(
        napkin,
        t4_clean,
        box=(250, 1345, 1040, 1965),
        dst_pos=(205, 845),
        target_w=460,
        boost_ink=1.22,
    )

    # 4. Tilt the hero napkin gently (-1.8 deg) with a soft realistic café-table shadow!
    nap_rgba = napkin.convert("RGBA")
    # Subtle rounded corners on the napkin sheet
    mask = Image.new("L", (nap_w, nap_h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, nap_w, nap_h], radius=14, fill=255)
    nap_rgba.putalpha(mask)
    nap_rot = nap_rgba.rotate(1.8, resample=Image.Resampling.BICUBIC, expand=True)
    nap_shadowed = add_drop_shadow(nap_rot, offset=(14, 24), blur=28, opacity=0.30, pad=55)

    comp = bg.convert("RGBA")
    nx = (W - nap_shadowed.width) // 2
    ny = (H - nap_shadowed.height) // 2 - 40
    comp.alpha_composite(nap_shadowed, (nx, ny))

    # 5. Layer authentic high-res Game Props (Coffee Latte, Ceremonial Matcha, Gold Fountain Pen)!
    # 5A. Coffee Latte overhanging upper-right corner!
    latte = Image.open(REPO / "docs/vfx/cups/coffee_latte_100.webp").convert("RGBA")
    latte = latte.resize((420, 420), Image.Resampling.LANCZOS).rotate(-22, resample=Image.Resampling.BICUBIC, expand=True)
    latte_sh = add_drop_shadow(latte, offset=(18, 26), blur=24, opacity=0.38, pad=50)
    comp.alpha_composite(latte_sh, (715, 70))

    # 5B. Ceremonial Matcha overhanging lower-left corner!
    matcha = Image.open(REPO / "docs/vfx/cups/matcha_pure_100.webp").convert("RGBA")
    matcha = matcha.resize((390, 390), Image.Resampling.LANCZOS).rotate(152, resample=Image.Resampling.BICUBIC, expand=True)
    matcha_sh = add_drop_shadow(matcha, offset=(16, 22), blur=24, opacity=0.36, pad=50)
    comp.alpha_composite(matcha_sh, (-145, 1345))

    # 5C. Gold-nibbed Fountain Pen resting diagonally across right edge of napkin!
    fountain = Image.open(REPO / "docs/vfx/pens/fountain.webp").convert("RGBA")
    fw, fh = fountain.size
    target_pw = 500
    fountain = fountain.resize((target_pw, int(round(fh * target_pw / fw))), Image.Resampling.LANCZOS)
    fountain = fountain.rotate(142, resample=Image.Resampling.BICUBIC, expand=True)
    fountain_sh = add_drop_shadow(fountain, offset=(10, 16), blur=14, opacity=0.42, pad=36)
    comp.alpha_composite(fountain_sh, (640, 1085))

    # 6. Graphic Trailer Stamps & Editorial Typography (100% Grid-Safe inside 1:1 & 3:4 center!)
    # Top kicker tag above "Napkin"
    draw_stamp(
        comp,
        "BILLION-DOLLAR IDEA?",
        center_xy=(430, 355),
        angle_deg=-2.5,
        bg_col=(28, 25, 22, 238),
        border_col=(245, 236, 220, 160),
        txt_col=(252, 211, 77, 255),
        font_size=30,
    )

    # Cheeky crimson "OOPS." stamp right next to the coffee-splattered "?" box!
    draw_stamp(
        comp,
        "OOPS.",
        center_xy=(765, 935),
        angle_deg=7.5,
        bg_col=(185, 28, 28, 245),
        border_col=(255, 245, 230, 220),
        txt_col=(255, 250, 242, 255),
        font_size=34,
    )

    # Crisp bottom CTA pill ("PLAY FREE BETA NOW") cleanly below Ship It! & inside 3:4 + Reels safe zone!
    d = ImageDraw.Draw(comp)
    font_cta = ImageFont.truetype(FONT_BOLD, 44)
    label = "PLAY FREE BETA NOW"
    bb = d.textbbox((0, 0), label, font=font_cta)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    ir = 24
    gap = 18
    px, py = 44, 24
    bw = ir * 2 + gap + tw + px * 2
    bh = max(ir * 2, th) + py * 2
    yc = 1555
    x0, y0 = (W - bw) // 2, yc - bh // 2
    x1, y1 = x0 + bw, y0 + bh

    d.rounded_rectangle([x0 + 5, y0 + 12, x1 + 5, y1 + 12], radius=32, fill=(36, 24, 12, 100))
    d.rounded_rectangle([x0, y0, x1, y1], radius=30, fill=(28, 25, 22, 246), outline=(245, 236, 220, 180), width=3)
    icx, icy = x0 + px + ir, yc
    d.ellipse([icx - ir, icy - ir, icx + ir, icy + ir], fill=(245, 158, 11, 255))
    d.polygon([(icx - 6, icy - 11), (icx - 6, icy + 11), (icx + 12, icy)], fill=(28, 25, 22, 255))
    d.text((icx + ir + gap - bb[0], yc - th // 2 - bb[1]), label, font=font_cta, fill=(250, 244, 234, 255))

    out_png = EDIT_DIR / "instagram_cover.png"
    comp.convert("RGB").save(out_png, quality=95)
    print(f"[COVER] Wrote {out_png} ({W}x{H})")

    # Also save a side-by-side 9:16 + 3:4 Profile Grid crop check image!
    grid_crop = comp.convert("RGB").crop((0, (H - 1440) // 2, W, (H + 1440) // 2))
    preview = Image.new("RGB", (540 + 40 + 540, 960), (24, 22, 20))
    preview.paste(comp.convert("RGB").resize((540, 960), Image.Resampling.LANCZOS), (0, 0))
    g_small = grid_crop.resize((540, 720), Image.Resampling.LANCZOS)
    preview.paste(g_small, (580, 120))
    preview.save(EDIT_DIR / "instagram_cover_preview.jpg", quality=92)
    print(f"[COVER] Wrote preview check (9:16 Reels + 3:4 Profile Grid)")


if __name__ == "__main__":
    build_cover()
