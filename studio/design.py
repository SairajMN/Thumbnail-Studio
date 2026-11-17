"""Drawing a reel cover that makes people stop scrolling, at 1080x1920, from a spec in one of the designs
(designs.py): the words are always exact.

Back to front:
1. the design's background (a studio, a colour flood, a spotlight, a sunset...) with its decorations;
2. the subject as the design shows it: cut out with a white outline and a glow, standing on a podium, as a
   silhouette, in a tilted card, or the frame itself; zoomed in on the subject ("punch-in") and colour graded;
3. a ring round the product the video is about and a hand-drawn arrow pointing at it;
4. the headline (a small line with one word highlighted, a huge line with a thick outline, an optional line under it
   for a date), the pill, the PEGI label and the brand wave, all with a drop shadow, over a wash that never reaches
   a face.
Then the cover is checked the way the profile grid shows it: the middle 3:4 (y 240-1680), about 240 px wide.
Nothing is ever drawn over a head: not a word, the pill, the wash, the ring, the arrow, the label or the wave.
"""

import io
from functools import cache
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import segno
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from pydantic import BaseModel, Field, model_validator

from studio import designs
from studio.designs import Recipe
from studio.palette import BLUE, INK, WHITE, YELLOW, Colors, Pal, contrast, ink_on, lum, resolve, rgb  # noqa: F401  (re-exported)

W, H = 1080, 1920
SAFE = (60, 240, 1020, 1680)  # the 3:4 profile-grid crop, less a side margin: every word and mark stays inside
TOP = 290  # where the headline block starts
GRID_W = 240  # a cover's width in the profile grid on a phone
GAP = 60  # clear space between the headline block and the top of a head
FADE = 240  # how far below the words the wash fades out, when no head is nearer
WAVE_TOP = 1610  # the wave's highest point; a head reaching below it turns the wave off
MIN_CONTRAST = 3.0  # WCAG ratio for large text
PILL_H = 70
SUBLINE_PX = 46
CUT_MODES = {"cutout", "pedestal", "silhouette"}

Template = Literal["staff_hook", "pedestal", "key_art", "group", "winner", "flyer"]
Box = tuple[float, float, float, float]  # x, y, w, h as fractions of the frame
PxBox = tuple[int, int, int, int]  # x, y, w, h in canvas pixels


class Spec(BaseModel):
    template: Template = "staff_hook"
    design: str = "staff_hook"  # a key of designs.DESIGNS
    line1: str = ""
    line2: str
    highlight: str | None = None  # a word of the small line drawn in the highlight colour, the one the eye should land on
    subline: str | None = None  # a small line under the big one: a date, a time, a place
    pill_left: str | None = None
    pill_right: str | None = None
    name_tag: str | None = None
    game: str | None = None
    colors: Colors = Colors()
    colors_from: str = "auto"  # "custom" when picked role by role (kept exactly); a preset, "game" or "design" otherwise
    punch_in: bool = True  # zoom in on the subject, and grade the colours
    arrow: bool = True  # ring and arrow on the product, when the analysis found one
    wave: bool | None = None  # None: the design's choice
    offer: str | None = Field(None, max_length=24)  # a price or offer from the context ("20% OFF"), shown as a tag on the Offer design
    qr_url: str | None = Field(None, pattern=r"^https?://\S{3,300}$")  # a link shown as a QR code card, on any design
    qr_label: str = Field("SCAN ME", max_length=20)
    pan_x: float = Field(0.0, ge=-1.0, le=1.0)
    pan_y: float = Field(0.0, ge=-1.0, le=1.0)
    zoom: float = Field(1.0, ge=0.5, le=2.5)

    @model_validator(mode="before")
    @classmethod
    def _older_covers(cls, v):
        """Covers saved before designs kept the game's colours as bg and accent."""
        if isinstance(v, dict) and ("bg" in v or "accent" in v):
            v = dict(v)
            bg, accent = v.pop("bg", None), v.pop("accent", None)
            v.setdefault("colors", {"bg": bg, "big": accent, "highlight": accent, "pill": accent, "glow": accent})
            v.pop("pop", None)
        return v

    @property
    def recipe(self) -> Recipe:
        return designs.DESIGNS.get(self.design, designs.DESIGNS["staff_hook"])


# --- type ---------------------------------------------------------------------------------------

FONTS = {  # (file, style inside a .ttc) in order of preference; macOS first, then any machine's fallback
    "heavy": [("/System/Library/Fonts/Avenir Next.ttc", "Heavy"), ("/System/Library/Fonts/Supplemental/Arial Black.ttf", None)],
    "bold": [("/System/Library/Fonts/Avenir Next.ttc", "Bold"), ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", None)],
    # for text with a character the faces above lack (₹ is missing from Avenir Next and Arial)
    "heavy_fallback": [("/System/Library/Fonts/SFCompact.ttf", None), ("/System/Library/Fonts/HelveticaNeue.ttc", "Condensed Black")],
    "bold_fallback": [("/System/Library/Fonts/HelveticaNeue.ttc", "Bold"), ("/System/Library/Fonts/Helvetica.ttc", "Bold")],
}


@cache
def _face_file(weight: str) -> tuple[str, int] | None:
    for path, style in FONTS[weight]:
        if not Path(path).exists():
            continue
        if style is None:
            return path, 0
        for i in range(24):
            try:
                if ImageFont.truetype(path, 10, index=i).getname()[1] == style:
                    return path, i
            except OSError:
                break
    return None


def font(weight: str, px: int) -> ImageFont.FreeTypeFont:
    f = _face_file(weight)
    return ImageFont.truetype(f[0], px, index=f[1]) if f else ImageFont.load_default(size=px)


@cache
def _has(weight: str, ch: str) -> bool:
    """Whether the weight's face draws `ch` (a missing character draws the same box as a private-use one)."""
    def raster(c):
        im = Image.new("L", (90, 90), 0)
        ImageDraw.Draw(im).text((8, 8), c, font=font(weight, 48), fill=255)
        return im.tobytes()
    return raster(ch) != raster("\ue000")


def font_for(weight: str, px: int, text: str) -> ImageFont.FreeTypeFont:
    """The weight's face, or its fallback when the text has a character the face lacks."""
    if all(_has(weight, ch) for ch in set(text)) or not _face_file(weight + "_fallback"):
        return font(weight, px)
    f = _face_file(weight + "_fallback")
    return ImageFont.truetype(f[0], px, index=f[1])


def _width(draw, text: str, f, tracking: float = 0) -> float:
    return draw.textlength(text, font=f) + tracking * max(0, len(text) - 1)


def _fit(draw, text: str, weight: str, start: int, low: int, max_w: int, spaced: bool = False) -> tuple[ImageFont.FreeTypeFont, int]:
    px = start
    while px > low and _width(draw, text, font_for(weight, px, text), 0.32 * px if spaced else 0) > max_w:
        px -= 4
    return font_for(weight, px, text), px


def _line2(draw, text: str) -> tuple[list[str], ImageFont.FreeTypeFont, int]:
    """The big line, as large as fits 900 px (its outline takes the rest of the safe width); a long one goes onto two
    lines when that lets it stay bigger."""
    f, px = _fit(draw, text, "heavy", 240, 80, 900)
    words = text.split()
    if px < 150 and len(words) > 1:
        cut = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
        two = [" ".join(words[:cut]), " ".join(words[cut:])]
        f2, px2 = _fit(draw, max(two, key=lambda t: draw.textlength(t, font=font_for("heavy", 100, t))), "heavy", 210, 80, 900)
        if px2 > px:
            return two, f2, px2
    return [text], f, px


def headline_bottom(spec: Spec) -> int:
    """How far down the headline block reaches (the subject is placed below it)."""
    d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    y = TOP + (70 if spec.name_tag else 0)
    if spec.line1:
        y += round(_fit(d, spec.line1.upper(), "bold", 76, 40, 920, spec.recipe.spaced)[1] * 1.05)
    lines, _, px2 = _line2(d, spec.line2.upper())
    y += round(px2 * 0.95) * len(lines)
    y += round(SUBLINE_PX * 1.4) if spec.subline else 0
    return y + (30 + PILL_H if spec.pill_left or spec.pill_right else 0)


# --- the picture ------------------------------------------------------------------------------

def head(box: PxBox) -> PxBox:
    """A face grown to the whole head: hair above, ears at the sides. Nothing may be drawn over it."""
    x, y, w, h = box
    return round(x - 0.15 * w), round(y - 0.45 * h), round(w * 1.3), round(h * 1.55)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) < 0.01 * mask.size:  # nothing that stands out
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())


def transform(pw: int, ph: int, bottom: int, heads: list[PxBox], subject: tuple[int, int, int, int] | None,
              punch_in: bool, product: PxBox | None = None, fill: float = 0.85, max_bottom: float | None = None,
              edge_off: bool = True) -> tuple[float, float, float]:
    """Scale and offset (s, ox, oy) that put the frame on the canvas: the subject under the headline, every head at
    least GAP below the words; with punch-in, zoomed so the subject fills `fill` of the width, but never so far that
    the product (the frame's pixels) drops out of the profile grid's crop, or the subject's bottom passes
    `max_bottom` (room for a podium)."""
    cover = max(W / pw, H / ph)
    top = bottom + GAP
    if subject is None and heads:
        xs, ys = [x for x, _, w, _ in heads] + [x + w for x, _, w, _ in heads], [y for _, y, _, h in heads] + [y + h for _, y, _, h in heads]
        subject = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    s = cover
    if punch_in and subject:
        sx, sy, sw, sh = subject
        anchor = min((y for _, y, _, _ in heads), default=sy)
        # fill the width, and let the subject run off the bottom by at most a quarter of its height
        s = min(fill * W / max(sw, 1), (H - top) / max(0.75 * (sh + (sy - anchor)), 1))
        if product and product[1] + product[3] > anchor:  # the product's bottom stays 80 px inside the crop
            s = min(s, (SAFE[3] - 80 - top) / (product[1] + product[3] - anchor))
        if max_bottom:
            s = min(s, (max_bottom - top) / max(sy + sh - anchor, 1))
        s = min(max(s, 0.3 * cover), 2.2 * cover)
    cx = (subject[0] + subject[2] / 2) if subject else pw / 2
    ox = W / 2 - cx * s
    if pw * s >= W:
        ox = min(0, max(W - pw * s, ox))
    anchor = min((y for _, y, _, _ in heads), default=subject[1] if subject else 0)
    oy = top - anchor * s if (heads or subject) else 0
    if not punch_in:
        oy = min(max(0, oy), 0.6 * H)  # the original framing only ever moves down
    if edge_off and subject and subject[1] + subject[3] >= 0.97 * ph:  # cut by the frame's bottom: keep that edge off-canvas
        oy = max(oy, H - ph * s)
    return s, ox, oy


def _warp(a: np.ndarray, s: float, ox: float, oy: float, interp=cv2.INTER_CUBIC) -> np.ndarray:
    return cv2.warpAffine(a, np.float32([[s, 0, ox], [0, s, oy]]), (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT)


def _graded(img: Image.Image) -> Image.Image:
    return ImageEnhance.Sharpness(ImageEnhance.Color(ImageEnhance.Contrast(img).enhance(1.12)).enhance(1.25)).enhance(1.4)


def _vignette(a: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((xx - W / 2) / (W / 2)) ** 2 * 0.6 + ((yy - H * 0.55) / (H / 2)) ** 2 * 0.6
    return a * np.clip(1 - 0.32 * r2, 0.55, 1)[..., None]


def _ellipse_k(n: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))


def _card(src: Image.Image, bottom: int, heads_src: list[PxBox], focus: tuple[float, float], panel: bool = False,
          pan: tuple[float, float] = (0.0, 0.0), zoom: float = 1.0,
          ) -> tuple[Image.Image, tuple[int, int], list[PxBox]]:
    """The frame in a white-bordered card tilted 4° (or, as a `panel`, straight with rounded corners and no border),
    placed under the headline: the card, where it goes, and the heads in canvas pixels."""
    top = bottom + GAP
    cw = round(W * (0.86 if panel else 0.80))
    ch = round(max(420, min(cw * 1.25, SAFE[3] - 40 - top)))
    pad, angle = (0, 0.0) if panel else (18, 4.0)
    iw, ih = cw - 2 * pad, ch - 2 * pad
    pw, ph = src.size
    sc = max(iw / pw, ih / ph) * max(1.0, zoom)
    max_cx, max_cy = max(0.0, pw * sc - iw), max(0.0, ph * sc - ih)
    cx0 = min(max(0.0, focus[0] * sc - iw / 2 - pan[0] * max(max_cx * 0.5, iw * 0.25)), max_cx)
    cy0 = min(max(0.0, focus[1] * sc - ih / 2 - pan[1] * max(max_cy * 0.5, ih * 0.25)), max_cy)
    if heads_src and pan == (0.0, 0.0):  # never crop through a head unless the user manually panned
        cy0 = max(0, min(cy0, min(y for _, y, _, _ in heads_src) * sc - 20))
    inner = src.resize((round(pw * sc), round(ph * sc)), Image.LANCZOS).crop((round(cx0), round(cy0), round(cx0) + iw, round(cy0) + ih))
    card = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))
    card.paste(inner, (pad, pad))
    if panel:
        rounded = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(rounded).rounded_rectangle((0, 0, cw - 1, ch - 1), radius=40, fill=255)
        card.putalpha(rounded)
    rot = card.rotate(angle, resample=Image.BICUBIC, expand=True)
    x0, y0 = (W - rot.width) // 2, top
    th = np.deg2rad(angle)

    def to_canvas(px: float, py: float) -> tuple[float, float]:
        dx, dy = px - cw / 2, py - ch / 2
        return x0 + rot.width / 2 + dx * np.cos(th) + dy * np.sin(th), y0 + rot.height / 2 - dx * np.sin(th) + dy * np.cos(th)

    heads = []
    for x, y, w, h in heads_src:
        corners = [to_canvas(pad + cx * sc - cx0, pad + cy * sc - cy0) for cx, cy in ((x, y), (x + w, y), (x, y + h), (x + w, y + h))]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        l, t = max(min(xs), x0), max(min(ys), y0)  # only what shows inside the card
        r, b = min(max(xs), x0 + rot.width), min(max(ys), y0 + rot.height)
        if r > l and b > t:
            heads.append((round(l), round(t), round(r - l), round(b - t)))
    return rot, (x0, y0), heads


def _tile(img: Image.Image, faces: list[Box], tw: int, th: int) -> tuple[Image.Image, list[PxBox]]:
    """An image cropped to tw x th around its faces (never through a head), and its heads in the tile's pixels."""
    src = img.convert("RGB")
    pw, ph = src.size
    heads = [head((round(x * pw), round(y * ph), round(w * pw), round(h * ph))) for x, y, w, h in faces]
    sc = max(tw / pw, th / ph)
    fx = sum(x + w / 2 for x, _, w, _ in heads) / len(heads) if heads else pw / 2
    fy = sum(y + h / 2 for _, y, _, h in heads) / len(heads) if heads else ph / 2
    x0 = min(max(0, fx * sc - tw / 2), pw * sc - tw)
    y0 = min(max(0, fy * sc - th / 2), ph * sc - th)
    if heads:
        y0 = max(0, min(y0, min(y for _, y, _, _ in heads) * sc - 16))
    tile = src.resize((round(pw * sc), round(ph * sc)), Image.LANCZOS).crop((round(x0), round(y0), round(x0) + tw, round(y0) + th))
    out = []
    for x, y, w, h in heads:
        l, t = max(0, x * sc - x0), max(0, y * sc - y0)
        r, b = min(tw, (x + w) * sc - x0), min(th, (y + h) * sc - y0)
        if r > l and b > t:
            out.append((round(l), round(t), round(r - l), round(b - t)))
    return tile, out


def collage(items: list[tuple[Image.Image, list[Box]]], bottom: int) -> tuple[Image.Image, list[PxBox]]:
    """Up to 4 images in white-bordered tiles under the headline (2 side by side; 3 as two over one; 4 as 2 x 2), on a
    transparent layer, with the heads in canvas pixels."""
    items = items[:4]
    rows = {1: [1], 2: [2], 3: [2, 1], 4: [2, 2]}[len(items)]
    top, gap, pad = bottom + GAP, 24, 10
    area_w, area_h = round(W * 0.88), SAFE[3] - 30 - (bottom + GAP)
    tw = (area_w - gap) // 2 if max(rows) == 2 else area_w
    th = min((area_h - gap * (len(rows) - 1)) // len(rows), round(tw * (1.6 if len(rows) == 1 else 1.0)))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    heads: list[PxBox] = []
    k = 0
    for r, n in enumerate(rows):
        y = top + r * (th + gap)
        x = (W - (n * tw + (n - 1) * gap)) // 2
        for _ in range(n):
            img, faces = items[k]
            tile, th_heads = _tile(img, faces, tw - 2 * pad, th - 2 * pad)
            card = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
            ImageDraw.Draw(card).rounded_rectangle((0, 0, tw - 1, th - 1), radius=20, fill=(255, 255, 255, 255))
            inner = Image.new("L", tile.size, 0)
            ImageDraw.Draw(inner).rounded_rectangle((0, 0, tile.width - 1, tile.height - 1), radius=12, fill=255)
            card.paste(tile, (pad, pad), inner)
            shadow = card.getchannel("A").filter(ImageFilter.GaussianBlur(16)).point(lambda v: int(v * 0.35))
            layer.alpha_composite(Image.merge("RGBA", (*Image.new("RGB", card.size, (0, 0, 0)).split(), shadow)), (x + 6, y + 12))
            layer.alpha_composite(card, (x, y))
            heads += [(x + pad + hx, y + pad + hy, hw, hh) for hx, hy, hw, hh in th_heads]
            x += tw + gap
            k += 1
    return layer, heads


def picture(spec: Spec, pal: Pal, frame: Image.Image, bottom: int, faces: list[Box], mask: np.ndarray | None,
            product: Box | None = None, cut: bool = True, extras: list[tuple[Image.Image, list[Box]]] = ()) -> dict:
    """The picture layers (1 and 2) as a float RGB array, with the heads in canvas pixels, the transform used (None
    for a card) and the subject treatment drawn. `cut` False draws the design without its cut-out: the frame itself.
    The subject is cut out only when it is a person or the product the video is about: on key art or a busy scene
    the most salient blob is often a glow or a logo, and a sticker of that looks broken."""
    recipe = spec.recipe
    src = frame.convert("RGB")
    pw, ph = src.size
    heads_src = [head((round(x * pw), round(y * ph), round(w * pw), round(h * ph))) for x, y, w, h in faces]
    subject = _bbox(mask) if mask is not None else None
    if subject and not heads_src:
        pb = product and (product[0] * pw, product[1] * ph, (product[0] + product[2]) * pw, (product[1] + product[3]) * ph)
        if not (pb and overlap(pb, _xyxy(subject))):
            subject = None  # no person, and not the product: nothing to cut out or zoom in on
    mode = recipe.subject
    if mode == "collage":
        mode = "collage" if extras else "panel"  # a collage needs the images the user adds
    if mode == "pedestal" and heads_src:
        mode = "cutout"  # people don't stand on a podium
    if not cut and mode in CUT_MODES or mode in CUT_MODES and subject is None:
        # without a cut-out: the frame itself; on a design with a background of its own, as a panel on it
        mode = "frame" if recipe.background in ("blur", "frame") else "panel"
    cover = max(W / pw, H / ph)
    frame_cover = _warp(np.asarray(src, np.float32), cover, (W - pw * cover) / 2, (H - ph * cover) / 2)
    back = designs.background(recipe, pal, frame_cover, spec.line2)
    graded = _graded(src) if spec.punch_in else src

    if mode == "collage":
        back = designs.decorate(back, recipe, pal, seed=7)
        img = Image.fromarray(np.clip(back, 0, 255).astype(np.uint8)).convert("RGBA")
        layer, heads = collage([(graded, list(faces)), *[((_graded(i) if spec.punch_in else i.convert("RGB")), f) for i, f in extras]], bottom)
        img.alpha_composite(layer)
        return {"base": np.asarray(img.convert("RGB"), np.float32), "heads": heads, "t": None, "mode": mode}

    if mode in ("card", "panel"):
        focus = ((subject[0] + subject[2] / 2) / pw, (subject[1] + subject[3] / 2) / ph) if subject else (0.5, 0.5)
        back = designs.decorate(back, recipe, pal, seed=7)
        img = Image.fromarray(np.clip(back, 0, 255).astype(np.uint8)).convert("RGBA")
        card, (x0, y0), heads = _card(graded, bottom, heads_src, (focus[0] * pw, focus[1] * ph), panel=mode == "panel",
                                      pan=(spec.pan_x, spec.pan_y), zoom=spec.zoom)
        shadow = card.getchannel("A").filter(ImageFilter.GaussianBlur(22)).point(lambda v: int(v * 0.45))
        img.alpha_composite(Image.merge("RGBA", (*Image.new("RGB", card.size, (0, 0, 0)).split(), shadow)), (x0 + 10, y0 + 18))
        img.alpha_composite(card, (x0, y0))
        return {"base": np.asarray(img.convert("RGB"), np.float32), "heads": heads, "t": None, "mode": mode}

    pedestal = mode == "pedestal"
    prod_src = product and (round(product[0] * pw), round(product[1] * ph), round(product[2] * pw), round(product[3] * ph))
    s, ox, oy = transform(pw, ph, bottom, heads_src, subject, spec.punch_in, prod_src or None,
                          fill=0.62 if pedestal else 0.72 if recipe.badge else 0.85,  # room for the podium, or the price tag
                          max_bottom=H * 0.78 if pedestal else None, edge_off=mode not in CUT_MODES)
    if spec.zoom != 1.0:
        ox = W / 2 - (W / 2 - ox) * spec.zoom
        oy = H / 2 - (H / 2 - oy) * spec.zoom
        s *= spec.zoom
    ox += spec.pan_x * W * 0.45
    oy += spec.pan_y * H * 0.45
    sharp = _warp(np.asarray(graded, np.float32), s, ox, oy)
    valid = _warp(np.ones((ph, pw), np.float32), s, ox, oy, cv2.INTER_LINEAR)
    sub_c = subject and (subject[0] * s + ox, subject[1] * s + oy, subject[2] * s, subject[3] * s)
    sun = sub_c and (sub_c[0] + sub_c[2] / 2, max(bottom + W * 0.3, sub_c[1] + sub_c[3] * 0.25))
    back = designs.decorate(back, recipe, pal, seed=7, sun_at=sun or (W / 2, H * 0.55))
    if pedestal and sub_c and sub_c[1] + sub_c[3] < H - 60:
        back = designs.podium(back, recipe.podium or "glossy", sub_c[0] + sub_c[2] / 2, sub_c[1] + sub_c[3] - sub_c[3] * 0.04,
                              min(max(sub_c[2] * 1.35, 380), W * 0.92))
    if mode in CUT_MODES:
        m = _warp(mask, s, ox, oy, cv2.INTER_LINEAR) * valid
        solid = cv2.morphologyEx((m > 0.5).astype(np.uint8), cv2.MORPH_OPEN, _ellipse_k(9))
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, _ellipse_k(11))
        n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
        if n_cc > 1:
            areas = cc_stats[1:, cv2.CC_STAT_AREA]
            keep_min = max(900, int(areas.max() * 0.05))
            cleaned = np.zeros_like(solid)
            for idx, area in enumerate(areas, start=1):
                if area >= keep_min:
                    cleaned[cc_labels == idx] = 1
            solid = cleaned
        alpha = cv2.GaussianBlur(solid.astype(np.float32), (0, 0), 1.8)[..., None]
        if mode == "silhouette":  # near-black figure with a thin rim of light
            rim = cv2.GaussianBlur(cv2.dilate(solid, _ellipse_k(17)).astype(np.float32), (0, 0), 8)[..., None]
            base = back * (1 - 0.7 * rim) + np.float32(rgb(pal.glow)) * 0.7 * rim
            base = base * (1 - alpha) + np.float32((12, 12, 14)) * alpha
        else:
            out = cv2.GaussianBlur(cv2.dilate(solid, _ellipse_k(29 if mode == "cutout" else 15)).astype(np.float32), (0, 0), 2.0)[..., None]
            glow = cv2.GaussianBlur(cv2.dilate(solid, _ellipse_k(61)).astype(np.float32), (0, 0), 22)[..., None] * (0.85 if mode == "cutout" else 0.4)
            base = back * (1 - glow) + np.float32(rgb(pal.glow)) * glow
            base = base * (1 - out) + np.float32(rgb(WHITE)) * out
            base = base * (1 - alpha) + sharp * alpha
    else:  # the frame itself over the background, its edges feathered
        v = cv2.GaussianBlur(cv2.erode(valid, np.ones((25, 25), np.uint8)), (0, 0), 12)[..., None]
        base = back * (1 - v) + sharp * v
    if spec.punch_in and recipe.background in ("blur", "frame"):
        base = _vignette(base)
    heads = [(round(x * s + ox), round(y * s + oy), round(w * s), round(h * s)) for x, y, w, h in heads_src]
    return {"base": base, "heads": heads, "t": (s, ox, oy), "mode": mode}


def wash_end(bottom: int, heads: list[PxBox]) -> int:
    """Where the headline's wash is gone: FADE below the words, or sooner, so it never reaches a head."""
    return max(bottom, min(H, bottom + FADE, min((t for _, t, _, _ in heads), default=H) - 10))


def _wash(a: np.ndarray, bottom: int, ground: str, end: int) -> np.ndarray:
    al = np.zeros(H, np.float32)
    al[:bottom] = 0.92
    al[bottom:end] = np.linspace(0.92, 0, end - bottom, dtype=np.float32)
    return a * (1 - al[:, None, None]) + np.float32(rgb(ground)) * al[:, None, None]


# --- marks --------------------------------------------------------------------------------------

def _xyxy(b: PxBox) -> tuple[int, int, int, int]:
    return b[0], b[1], b[0] + b[2], b[1] + b[3]


def overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _bezier(p0, p1, p2, n=48) -> list[tuple[float, float]]:
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0], (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
            for t in np.linspace(0, 1, n)]


def plan_marks(product: PxBox, heads: list[PxBox], keep_out: list[tuple[int, int, int, int]], bottom: int) -> dict:
    """A ring round the product and an arrow towards it from the side with more room, neither touching a head or the
    words. Either may be left out when no clear route exists."""
    x, y, w, h = product
    pad_x, pad_y = 0.18 * w + 14, 0.18 * h + 14
    ring = (round(x - pad_x), round(y - pad_y), round(x + w + pad_x), round(y + h + pad_y))
    blocked = [_xyxy(hd) for hd in heads] + keep_out
    marks: dict = {}
    if not any(overlap(ring, b) for b in blocked) and ring[1] > bottom and SAFE[0] <= ring[0] and ring[2] <= SAFE[2] and ring[3] <= SAFE[3]:
        marks["ring"] = ring
    cx, cy, rx, ry = (ring[0] + ring[2]) / 2, (ring[1] + ring[3]) / 2, (ring[2] - ring[0]) / 2, (ring[3] - ring[1]) / 2
    head_x = sum(hx + hw / 2 for hx, _, hw, _ in heads) / len(heads) if heads else W / 2
    sides = sorted([110, W - 110], key=lambda sx: -abs(sx - head_x))  # away from the faces first
    for sx in sides:
        start = (sx, min(max(cy + 280, bottom + 120), SAFE[3] - 70))
        ang = np.arctan2(start[1] - cy, start[0] - cx)
        end = (cx + (rx + 26) * np.cos(ang), cy + (ry + 26) * np.sin(ang))
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        nx, ny = -(end[1] - start[1]), end[0] - start[0]
        k = 0.25 * (1 if sx < W / 2 else -1)
        pts = _bezier(start, (mid[0] + k * nx, mid[1] + k * ny), end)
        if np.hypot(end[0] - start[0], end[1] - start[1]) < 240:
            continue  # too close to the edge for an arrow that reads as one
        if not any(blocked_b[0] - 30 < px < blocked_b[2] + 30 and blocked_b[1] - 30 < py < blocked_b[3] + 30
                   for px, py in pts for blocked_b in blocked) and all(SAFE[0] <= px <= SAFE[2] and SAFE[1] <= py <= SAFE[3] for px, py in pts):
            marks["arrow"] = pts
            break
    return marks


def _draw_marks(d: ImageDraw.ImageDraw, marks: dict, colour: str) -> dict[str, tuple[int, int, int, int]]:
    boxes = {}
    edge = ink_on(colour)
    if "ring" in marks:
        r = marks["ring"]
        d.ellipse(r, outline=edge, width=22)
        d.ellipse(r, outline=colour, width=12)
        boxes["ring"] = r
    if "arrow" in marks:
        pts = marks["arrow"]
        d.line(pts, fill=edge, width=34, joint="curve")
        d.line(pts, fill=colour, width=20, joint="curve")
        (x1, y1), (x2, y2) = pts[-4], pts[-1]
        ang, size = np.arctan2(y2 - y1, x2 - x1), 64
        tip = (x2 + 14 * np.cos(ang), y2 + 14 * np.sin(ang))
        wing = lambda a: (tip[0] - size * np.cos(ang - a), tip[1] - size * np.sin(ang - a))  # noqa: E731
        d.polygon([tip, wing(0.5), wing(-0.5)], fill=colour, outline=edge, width=5)
        xs, ys = [p[0] for p in pts] + [tip[0]], [p[1] for p in pts] + [tip[1]]
        boxes["arrow"] = (round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys)))
    return boxes


def place(size: tuple[int, int], spots: list[tuple[int, int]], taken: list[tuple[int, int, int, int]]) -> tuple[int, int] | None:
    """The first spot where a badge of `size` touches nothing taken (heads, marks, words, other badges), or None."""
    w, h = size
    return next(((x, y) for x, y in spots if SAFE[0] <= x and x + w <= SAFE[2] and SAFE[1] <= y and y + h <= SAFE[3]
                 and not any(overlap((x, y, x + w, y + h), t) for t in taken)), None)


def corners(size: tuple[int, int], bottom: int) -> list[tuple[int, int]]:
    """Places for a badge: the lower corners first, then up the sides, never above the headline."""
    w, h = size
    left, right = SAFE[0] + 10, SAFE[2] - 10 - w
    ys = [y for y in (SAFE[3] - 80 - h, SAFE[3] - 420 - h, SAFE[3] - 760 - h) if y > bottom + 20]
    return [(x, y) for y in ys for x in (left, right)]


def price_tag(text: str, fill: str) -> Image.Image:
    """The offer on a tilted price tag, with a hole like a shop tag."""
    f, _ = _fit(ImageDraw.Draw(Image.new("RGB", (8, 8))), text.upper(), "heavy", 76, 36, 250)
    d0 = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    tw = round(d0.textlength(text.upper(), font=f))
    w, h = tw + 110, 136
    tag = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(tag)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=26, fill=fill, outline=ink_on(fill), width=6)
    d.ellipse((24, h / 2 - 14, 52, h / 2 + 14), fill=(0, 0, 0, 0), outline=ink_on(fill), width=5)  # the hole
    d.text((70 + tw / 2, h / 2), text.upper(), font=f, fill=ink_on(fill), anchor="mm")
    return tag.rotate(-8, resample=Image.BICUBIC, expand=True)


def qr_card(url: str, label: str) -> Image.Image:
    """A white card with the link's QR code and a label under it."""
    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="png", scale=6, border=2)
    qr = Image.open(io.BytesIO(buf.getvalue())).convert("RGBA").resize((190, 190), Image.NEAREST)
    card = Image.new("RGBA", (220, 262), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    d.rounded_rectangle((0, 0, 219, 261), radius=18, fill=(255, 255, 255, 255))
    card.alpha_composite(qr, (15, 15))
    f, _ = _fit(d, label.upper(), "bold", 30, 16, 190)
    d.text((110, 232), label.upper(), font=f, fill=INK, anchor="mm")
    return card


def _wave(d: ImageDraw.ImageDraw, colour: str) -> None:
    xs = np.linspace(0, W, 60)
    for base, amp, phase, c in ((1640, 26, 0.0, colour), (1676, 30, 1.3, BLUE)):
        d.polygon([(float(x), base + amp * np.sin(x / W * 2 * np.pi + phase)) for x in xs] + [(W, H), (0, H)], fill=c)


# --- the words ----------------------------------------------------------------------------------

def _pill(d: ImageDraw.ImageDraw, y: int, left: str | None, right: str | None, fill: str, x_left: int | None = None) -> tuple[int, int, int, int]:
    f = font_for("bold", 34, f"{left or ''}{right or ''}")
    pad, h = 32, PILL_H
    parts = [t for t in (left, right) if t]
    widths = [round(d.textlength(t, font=f)) + 2 * pad for t in parts]
    x = x_left if x_left is not None else (W - sum(widths)) // 2
    if len(parts) == 2:  # white half with a blue outline, then the filled half
        d.rounded_rectangle([x, y, x + sum(widths), y + h], radius=h // 2, fill=WHITE, outline=BLUE, width=3)
        d.rounded_rectangle([x + widths[0], y, x + sum(widths), y + h], radius=h // 2, fill=fill)
        d.rectangle([x + widths[0], y, x + widths[0] + h // 2, y + h], fill=fill)
        d.text((x + widths[0] / 2, y + h / 2), parts[0], font=f, fill=BLUE, anchor="mm")
        d.text((x + widths[0] + widths[1] / 2, y + h / 2), parts[1], font=f, fill=ink_on(fill), anchor="mm")
    else:
        d.rounded_rectangle([x, y, x + widths[0], y + h], radius=h // 2, fill=fill)
        d.text((x + widths[0] / 2, y + h / 2), parts[0], font=f, fill=ink_on(fill), anchor="mm")
    return x, y, x + sum(widths), y + h


def _words(spec: Spec, pal: Pal) -> tuple[Image.Image, dict, dict]:
    """The words on their own transparent layer, with a drop shadow; their boxes and sizes."""
    recipe = spec.recipe
    left = recipe.align == "left"
    x_left = SAFE[0] + 20
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    boxes: dict[str, tuple[int, int, int, int]] = {}
    stroke2 = INK if lum(pal.big) > 0.3 else WHITE  # the big line always gets a thick outline: the sticker look
    y, px1 = TOP, None
    if spec.name_tag:
        f = font_for("bold", 34, spec.name_tag.upper())
        tw = round(d.textlength(spec.name_tag.upper(), font=f))
        x0 = x_left if left else (W - tw) // 2 - 26
        boxes["name_tag"] = (x0, y, x0 + tw + 52, y + 54)
        d.rounded_rectangle(boxes["name_tag"], radius=27, outline=pal.small, width=3)
        d.text((x0 + 26 + tw / 2, y + 27), spec.name_tag.upper(), font=f, fill=pal.small, anchor="mm")
        y += 70
    if spec.line1:
        text = spec.line1.upper()
        f1, px1 = _fit(d, text, "bold", 76, 40, 920, recipe.spaced)
        track = 0.32 * px1 if recipe.spaced else 0
        width = _width(d, text, f1, track)
        x = x_left if left else (W - width) / 2
        hi = (spec.highlight or "").upper().strip(" ,.!?")
        if recipe.spaced:
            for ch in text:
                d.text((x, y), ch, font=f1, fill=pal.small)
                x += d.textlength(ch, font=f1) + track
        else:
            for word in text.split(" "):
                d.text((x, y), word, font=f1, fill=pal.highlight if hi and word.strip(",.!?") == hi else pal.small)
                x += d.textlength(word + " ", font=f1)
        b = d.textbbox((0, y), text, font=f1)
        x0 = x_left if left else round((W - width) / 2)
        boxes["line1"] = (x0, y, round(x0 + width), b[3])
        y += round(px1 * 1.05)
    lines, f2, px2 = _line2(d, spec.line2.upper())
    sw = max(4, round(px2 * (0.06 if recipe.hollow else 0.05)))
    for i, line in enumerate(lines):
        yy = y + i * round(px2 * 0.95)
        xy, anchor = ((x_left, yy), "lt") if left else ((W / 2, yy), "mt")
        boxes[f"line2_{i}"] = d.textbbox(xy, line, font=f2, anchor=anchor, stroke_width=sw)
        if recipe.hollow:  # the outline only: drawing the fill transparent clears the inside
            d.text(xy, line, font=f2, fill=(0, 0, 0, 0), anchor=anchor, stroke_width=sw, stroke_fill=pal.big)
        else:
            d.text(xy, line, font=f2, fill=pal.big, anchor=anchor, stroke_width=sw, stroke_fill=stroke2)
    y += round(px2 * 0.95) * len(lines)
    if spec.subline:
        fs, _ = _fit(d, spec.subline.upper(), "bold", SUBLINE_PX, 28, 900)
        xy, anchor = ((x_left, y + 8), "lt") if left else ((W / 2, y + 8), "mt")
        boxes["subline"] = d.textbbox(xy, spec.subline.upper(), font=fs, anchor=anchor)
        d.text(xy, spec.subline.upper(), font=fs, fill=pal.small, anchor=anchor)
        y += round(SUBLINE_PX * 1.4)
    if spec.pill_left or spec.pill_right:
        boxes["pill"] = _pill(d, y + 30, spec.pill_left, spec.pill_right, pal.pill, x_left if left else None)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(layer.getchannel("A").point(lambda v: int(v * 0.6)).filter(ImageFilter.GaussianBlur(12)))
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.alpha_composite(shadow, (0, 10))
    out.alpha_composite(layer)
    return out, boxes, {"line1_px": px1, "line2_px": px2, "stroke": pal.big if recipe.hollow else stroke2}


# --- the cover ----------------------------------------------------------------------------------

def palette_for(spec: Spec, frame: Image.Image) -> Pal:
    light = float(np.asarray(frame.convert("L").resize((64, 64))).mean()) > 118
    return resolve(spec.colors, spec.recipe.colors, designs.wash_of(spec.recipe), light, keep=spec.colors_from == "custom" or spec.colors_from.startswith("mine:"))


def draw(spec: Spec, frame: Image.Image, faces: list[Box] = (), mask: np.ndarray | None = None, product: Box | None = None,
         cut: bool = True, extras: list[tuple[Image.Image, list[Box]]] = ()) -> tuple[Image.Image, dict]:
    """The cover and its report: the box of every word and mark, the heads, the checks, and the subject treatment
    used. `faces` and `mask` come from the frame (studio.frame); `product` is the object the video is about, as the
    look model found it. `cut` False draws the design with the frame itself instead of its cut-out. `extras`: the
    images (with their faces) the user added for a collage."""
    bottom = headline_bottom(spec)
    src = frame.convert("RGB")
    pal = palette_for(spec, src)
    pic = picture(spec, pal, src, bottom, list(faces), mask, product, cut, list(extras))
    heads = pic["heads"]
    end = wash_end(bottom, heads)
    base = _wash(pic["base"], bottom, pal.wash, end)
    canvas = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(canvas)
    if (spec.wave if spec.wave is not None else spec.recipe.wave) and not any(t + h > WAVE_TOP for _, t, _, h in heads):
        _wave(d, pal.pill)
    under = canvas.copy()  # what the words are checked against
    words, boxes, sizes = _words(spec, pal)
    marks: dict = {}
    if spec.arrow and product and pic["t"] and pic["mode"] != "silhouette":  # a ring round a black shape says nothing
        s, ox, oy = pic["t"]
        pw, ph = src.size
        px = (round(product[0] * pw * s + ox), round(product[1] * ph * s + oy), round(product[2] * pw * s), round(product[3] * ph * s))
        marks = plan_marks(px, heads, list(boxes.values()), bottom)
        boxes.update(_draw_marks(d, marks, pal.glow))
    canvas = canvas.convert("RGBA")
    canvas.alpha_composite(words)
    taken = [_xyxy(hd) for hd in heads] + [b for k, b in boxes.items() if k not in ("arrow",)] \
        + [(px - 24, py - 24, px + 24, py + 24) for px, py in marks.get("arrow", [])]  # the arrow's line, not its box
    badges = []
    if spec.offer and spec.recipe.badge:
        badges.append(("offer", price_tag(spec.offer, pal.pill)))
    if spec.qr_url:
        badges.append(("qr", qr_card(spec.qr_url, spec.qr_label)))
    left_out = []
    for name, img in badges:  # each in the first free place, so badges never cover a face, a mark, the words or each other
        w, h = img.size
        # the price tag is the point of its design: any free place down either side, highest first; the others: corners
        scan = [(x, y) for y in range(bottom + 20, SAFE[3] - h, 40) for x in (SAFE[2] - 10 - w, SAFE[0] + 10)] if name == "offer" else []
        spot = place(img.size, scan or corners(img.size, bottom), taken)
        if spot is None:
            left_out.append(name)
            continue
        canvas.alpha_composite(img, spot)
        boxes[name] = (spot[0], spot[1], spot[0] + img.width, spot[1] + img.height)
        taken.append(boxes[name])
    rep = {"boxes": boxes, "heads": heads, "wash_end": end, "light": pal.light, "mode": pic["mode"], "cut": pic["mode"] in CUT_MODES,
           "colour1": pal.small, "left_out": left_out, **sizes}
    rep["checks"] = checks(spec, rep, under, pal)
    return canvas.convert("RGB"), rep


def _mean(img: Image.Image, box) -> tuple[int, int, int]:
    l, t, r, b = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
    if r <= l or b <= t:
        return 255, 255, 255
    return tuple(int(v) for v in np.asarray(img.crop((l, t, r, b))).reshape(-1, 3).mean(0))


def checks(spec: Spec, rep: dict, under: Image.Image, pal: Pal) -> list[dict]:
    out = []
    l, t, r, b = SAFE
    outside = [k for k, (x0, y0, x1, y1) in rep["boxes"].items() if x0 < l or y0 < t or x1 > r or y1 > b]
    out.append({"id": "safe_zone", "ok": not outside,
                "detail": "Everything sits inside the profile grid's 3:4 crop." if not outside else f"Cut off in the profile grid: {', '.join(outside)}."})
    worst = None
    for k, box in rep["boxes"].items():
        if k in ("line1", "subline"):
            c = contrast(pal.small, _mean(under, box))
        elif k.startswith("line2"):
            c = max(contrast(pal.big, _mean(under, box)), contrast(rep["stroke"], _mean(under, box)) if spec.recipe.hollow
                    else contrast(rep["stroke"], pal.big))
        else:
            continue
        worst = c if worst is None else min(worst, c)
    out.append({"id": "contrast", "ok": worst is None or worst >= MIN_CONTRAST,
                "detail": f"Lowest headline contrast {worst:.1f}:1 (needs {MIN_CONTRAST:g}:1)." if worst is not None else "No headline."})
    scale = GRID_W / W
    big, small = rep["line2_px"] * 0.72 * scale, (rep["line1_px"] or 0) * 0.72 * scale
    ok = big >= 9 and (not spec.line1 or small >= 5)
    out.append({"id": "grid_size", "ok": ok, "detail": f"In the grid the big line is {big:.0f} px tall"
                + (f" and the small line {small:.0f} px" if spec.line1 else "") + (" (readable)." if ok else " (too small: shorten it).")})
    hit = sorted({k for k, bx in rep["boxes"].items() for hd in rep["heads"] if overlap(bx, _xyxy(hd))})
    if any(t < rep["wash_end"] for _, t, _, _ in rep["heads"]):
        hit.append("the wash behind the headline")
    out.append({"id": "face_clear", "ok": not hit,
                "detail": (f"Nothing covers the {len(rep['heads'])} face(s)." if rep["heads"] else "No face in the frame.") if not hit
                else f"Covering a face: {', '.join(hit)}."})
    long = [n for n, text, most in (("small line", spec.line1, 6), ("big line", spec.line2, 3)) if len(text.split()) > most]
    out.append({"id": "short", "ok": not long, "detail": "Headline reads at a glance." if not long
                else f"Too many words in the {' and '.join(long)} (small line 6 at most, big line 3)."})
    names = {"offer": "the price tag", "qr": "the QR code"}
    if rep.get("left_out"):
        out.append({"id": "badges", "ok": False, "detail": f"No free place without covering a face or the words for {', '.join(names[n] for n in rep['left_out'])}."})
    return out
