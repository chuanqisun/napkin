import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops
from make_instagram_cover import extract_frame, add_drop_shadow, paste_multiply_region

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RAW_DIR = ROOT / "raw"
EDIT_DIR = ROOT / "edit"

W, H = 1080, 1920
FONT_ZH_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_ZH_IDX = 2


def draw_zh_stamp(base_rgba: Image.Image, text: str, center_xy, angle_deg: float, bg_col, border_col, txt_col, font_size=34):
    font = ImageFont.truetype(FONT_ZH_TTC, font_size, index=FONT_ZH_IDX)
    tmp = Image.new("RGBA", (480, 140), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bb = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    px, py = 28, 15
    bw, bh = tw + px * 2, th + py * 2
    x0, y0 = (480 - bw) // 2, (140 - bh) // 2
    d.rounded_rectangle([x0 + 3, y0 + 6, x0 + bw + 3, y0 + bh + 6], radius=16, fill=(35, 22, 10, 75))
    d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=15, fill=bg_col, outline=border_col, width=3)
    d.text((x0 + px - bb[0], y0 + py - bb[1]), text, font=font, fill=txt_col)
    rot = tmp.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=True)
    rw, rh = rot.size
    base_rgba.alpha_composite(rot, (int(center_xy[0] - rw // 2), int(center_xy[1] - rh // 2)))


def build_zh_cover():
    print("Building 1080x1920 Chinese Instagram Reels Cover...")
    bg = Image.new("RGB", (W, H), (206, 187, 159))
    bpx = bg.load()
    for y in range(H):
        for x in range(W):
            dx = (x - W * 0.48) / (W * 0.85)
            dy = (y - H * 0.42) / (H * 0.65)
            t = min(1.0, math.hypot(dx, dy))
            bpx[x, y] = (int(223 - t * 38), int(206 - t * 42), int(179 - t * 46))

    t5_clean = extract_frame(RAW_DIR / "05_endcard_napkin.mp4", 60)
    t2_splat = extract_frame(RAW_DIR / "02_block_order_spill.mp4", 160)
    t4_eureka = extract_frame(RAW_DIR / "04_matcha_eureka.mp4", 275)

    paper_base = t5_clean.crop((40, 1360, 1380, 2500)).convert("RGB")
    nap_w, nap_h = 900, 1330
    napkin = paper_base.resize((nap_w, nap_h), Image.Resampling.LANCZOS)

    nd = ImageDraw.Draw(napkin, "RGBA")
    nd.rectangle([24, 24, nap_w - 24, nap_h - 24], outline=(112, 86, 56, 42), width=2)
    nd.rectangle([32, 32, nap_w - 32, nap_h - 32], outline=(112, 86, 56, 28), width=1)
    for k in range(-18, 19):
        a = int(14 * math.exp(-((k / 8.0) ** 2)))
        col = (90, 66, 38, a) if k < 0 else (255, 250, 240, int(a * 1.15))
        nd.line([(nap_w // 2 + k, 0), (nap_w // 2 + k, nap_h)], fill=col, width=1)
        nd.line([(0, nap_h // 2 + k), (nap_w, nap_h // 2 + k)], fill=col, width=1)

    paste_multiply_region(napkin, t5_clean, box=(260, 950, 1180, 1345), dst_pos=(115, 118), target_w=660, boost_ink=1.22)
    paste_multiply_region(napkin, t2_splat, box=(320, 710, 1020, 1445), dst_pos=(210, 405), target_w=440, boost_ink=1.22)

    t4_clean = t4_eureka.copy()
    t4_bg = t4_clean.getpixel((860, 1950))
    ImageDraw.Draw(t4_clean).rectangle([885, 1870, 975, 1960], fill=t4_bg)
    paste_multiply_region(napkin, t4_clean, box=(250, 1345, 1040, 1965), dst_pos=(205, 845), target_w=460, boost_ink=1.22)

    nap_rgba = napkin.convert("RGBA")
    mask = Image.new("L", (nap_w, nap_h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, nap_w, nap_h], radius=14, fill=255)
    nap_rgba.putalpha(mask)
    nap_rot = nap_rgba.rotate(1.8, resample=Image.Resampling.BICUBIC, expand=True)
    nap_shadowed = add_drop_shadow(nap_rot, offset=(14, 24), blur=28, opacity=0.30, pad=55)

    comp = bg.convert("RGBA")
    comp.alpha_composite(nap_shadowed, ((W - nap_shadowed.width) // 2, (H - nap_shadowed.height) // 2 - 40))

    latte = Image.open(REPO / "docs/vfx/cups/coffee_latte_100.webp").convert("RGBA").resize((420, 420), Image.Resampling.LANCZOS).rotate(-22, resample=Image.Resampling.BICUBIC, expand=True)
    comp.alpha_composite(add_drop_shadow(latte, offset=(18, 26), blur=24, opacity=0.38, pad=50), (715, 70))

    matcha = Image.open(REPO / "docs/vfx/cups/matcha_pure_100.webp").convert("RGBA").resize((390, 390), Image.Resampling.LANCZOS).rotate(152, resample=Image.Resampling.BICUBIC, expand=True)
    comp.alpha_composite(add_drop_shadow(matcha, offset=(16, 22), blur=24, opacity=0.36, pad=50), (-145, 1345))

    fountain = Image.open(REPO / "docs/vfx/pens/fountain.webp").convert("RGBA")
    fw, fh = fountain.size
    fountain = fountain.resize((500, int(round(fh * 500 / fw))), Image.Resampling.LANCZOS).rotate(142, resample=Image.Resampling.BICUBIC, expand=True)
    comp.alpha_composite(add_drop_shadow(fountain, offset=(10, 16), blur=14, opacity=0.42, pad=36), (640, 1085))

    draw_zh_stamp(
        comp,
        "值一个亿的灵感？",
        center_xy=(430, 355),
        angle_deg=-2.5,
        bg_col=(28, 25, 22, 238),
        border_col=(245, 236, 220, 160),
        txt_col=(252, 211, 77, 255),
        font_size=32,
    )
    draw_zh_stamp(
        comp,
        "糟了！",
        center_xy=(765, 935),
        angle_deg=7.5,
        bg_col=(185, 28, 28, 245),
        border_col=(255, 245, 230, 220),
        txt_col=(255, 250, 242, 255),
        font_size=36,
    )

    d = ImageDraw.Draw(comp)
    font_cta = ImageFont.truetype(FONT_ZH_TTC, 44, index=FONT_ZH_IDX)
    label = "免费公测 · 立即开玩"
    bb = d.textbbox((0, 0), label, font=font_cta)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    ir, gap, px, py = 24, 18, 44, 24
    bw, bh = ir * 2 + gap + tw + px * 2, max(ir * 2, th) + py * 2
    yc = 1555
    x0, y0 = (W - bw) // 2, yc - bh // 2
    x1, y1 = x0 + bw, y0 + bh

    d.rounded_rectangle([x0 + 5, y0 + 12, x1 + 5, y1 + 12], radius=32, fill=(36, 24, 12, 100))
    d.rounded_rectangle([x0, y0, x1, y1], radius=30, fill=(28, 25, 22, 246), outline=(245, 236, 220, 180), width=3)
    icx, icy = x0 + px + ir, yc
    d.ellipse([icx - ir, icy - ir, icx + ir, icy + ir], fill=(245, 158, 11, 255))
    d.polygon([(icx - 6, icy - 11), (icx - 6, icy + 11), (icx + 12, icy)], fill=(28, 25, 22, 255))
    d.text((icx + ir + gap - bb[0], yc - th // 2 - bb[1]), label, font=font_cta, fill=(250, 244, 234, 255))

    out_png = EDIT_DIR / "instagram_cover_zh.png"
    comp.convert("RGB").save(out_png, quality=95)
    print(f"[COVER ZH] Wrote {out_png} ({W}x{H})")


if __name__ == "__main__":
    build_zh_cover()
