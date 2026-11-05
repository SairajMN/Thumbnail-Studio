"""Drawing a reel cover that makes people stop scrolling, at 1080x1920, from a spec: the words are always exact.

Back to front:
1. punch-in: the frame zoomed so the subject (face, person, product) fills most of the width, colour graded
   (contrast, saturation, sharpness) and vignetted;
2. subject pop: behind the cut-out subject, a blurred copy of the frame tinted with the game's background; around it a
   glow in the game's accent and a white outline, the sticker look viral covers use;
3. a ring round the product the video is about and a hand-drawn arrow pointing at it;
4. the headline (a small line with one word highlighted, a huge line in the accent with a thick outline), the pill,
   the PEGI label and the brand wave, all with a drop shadow, over a wash that never reaches a face.
Then the cover is checked the way the profile grid shows it: the middle 3:4 (y 240-1680), about 240 px wide.
Nothing is ever drawn over a head: not a word, the pill, the wash, the ring, the arrow, the label or the wave.
"""

from functools import cache
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from pydantic import BaseModel, Field

W, H = 1080, 1920
SAFE = (60, 240, 1020, 1680)  # the 3:4 profile-grid crop, less a side margin: every word and mark stays inside
TOP = 290  # where the headline block starts
GRID_W = 240  # a cover's width in the profile grid on a phone
GAP = 60  # clear space between the headline block and the top of a head
FADE = 240  # how far below the words the wash fades out, when no head is nearer
WAVE_TOP = 1610  # the wave's highest point; a head reaching below it turns the wave off
BLUE, YELLOW, NAVY, WHITE, INK = "#0050D0", "#FFC400", "#0A1640", "#FFFFFF", "#111111"
MIN_CONTRAST = 3.0  # WCAG ratio for large text
PILL_H = 70
HEX = r"^#[0-9A-Fa-f]{6}$"

Template = Literal["staff_hook", "pedestal", "key_art", "group", "winner", "flyer"]
Box = tuple[float, float, float, float]  # x, y, w, h as fractions of the frame
PxBox = tuple[int, int, int, int]  # x, y, w, h in canvas pixels


class Spec(BaseModel):
    template: Template = "staff_hook"
    line1: str = ""
    line2: str
    highlight: str | None = None  # a word of the small line drawn in the accent, the one the eye should land on
    pill_left: str | None = None
    pill_right: str | None = None
    name_tag: str | None = None
    rating: str | None = None  # a brand-kit label (pegi_16), drawn lower left (lower right when a head is there)
    game: str | None = None
    bg: str | None = Field(None, pattern=HEX)  # the game's background colour; None: white or navy by the picture
    accent: str | None = Field(None, pattern=HEX)  # the game's accent colour; None: the house yellow
    pop: bool = True  # cut-out subject with outline and glow over a blurred background
    punch_in: bool = True  # zoom in on the subject, and grade the colours
    arrow: bool = True  # ring and arrow on the product, when the analysis found one
    wave: bool | None = None  # None: on light covers


# --- colour and type --------------------------------------------------------------------------

FONTS = {  # (file, style inside a .ttc) in order of preference; macOS first, then any machine's fallback
    "heavy": [("/System/Library/Fonts/Avenir Next.ttc", "Heavy"), ("/System/Library/Fonts/Supplemental/Arial Black.ttf", None)],
    "bold": [("/System/Library/Fonts/Avenir Next.ttc", "Bold"), ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", None)],
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


def rgb(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def lum(c) -> float:
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = c
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(a, b) -> float:
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def ink_on(fill: str) -> str:
    """Black or white, whichever reads better on `fill`."""
    return INK if contrast(rgb(INK), rgb(fill)) >= contrast(rgb(WHITE), rgb(fill)) else WHITE


def _fit(draw, text: str, weight: str, start: int, low: int, max_w: int) -> tuple[ImageFont.FreeTypeFont, int]:
    px = start
    while px > low and draw.textlength(text, font=font(weight, px)) > max_w:
        px -= 4
    return font(weight, px), px


def _line2(draw, text: str) -> tuple[list[str], ImageFont.FreeTypeFont, int]:
    """The big line, as large as fits 900 px (its outline takes the rest of the safe width); a long one goes onto two
    lines when that lets it stay bigger."""
    f, px = _fit(draw, text, "heavy", 240, 80, 900)
    words = text.split()
    if px < 150 and len(words) > 1:
        cut = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
        two = [" ".join(words[:cut]), " ".join(words[cut:])]
        f2, px2 = _fit(draw, max(two, key=lambda t: draw.textlength(t, font=font("heavy", 100))), "heavy", 210, 80, 900)
        if px2 > px:
            return two, f2, px2
    return [text], f, px


def headline_bottom(spec: Spec) -> int:
    """How far down the headline block reaches (the subject is placed below it)."""
    d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    y = TOP + (70 if spec.name_tag else 0)
    if spec.line1:
        y += round(_fit(d, spec.line1.upper(), "bold", 76, 40, 920)[1] * 1.05)
    lines, _, px2 = _line2(d, spec.line2.upper())
    y += round(px2 * 0.95) * len(lines)
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
              punch_in: bool, product: PxBox | None = None) -> tuple[float, float, float]:
    """Scale and offset (s, ox, oy) that put the frame on the canvas: the subject under the headline, every head at
    least GAP below the words; with punch-in, zoomed so the subject fills most of the width, but never so far that
    the product (the frame's pixels) drops out of the profile grid's crop."""
    cover = max(W / pw, H / ph)
    top = bottom + GAP
    if subject is None and heads:
        xs, ys = [x for x, _, w, _ in heads] + [x + w for x, _, w, _ in heads], [y for _, y, _, h in heads] + [y + h for _, y, _, h in heads]
        subject = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    s = cover
    if punch_in and subject:
        sx, sy, sw, sh = subject
        # fill 85% of the width, and let the subject run off the bottom by at most a quarter of its height
        s = min(0.85 * W / max(sw, 1), (H - top) / max(0.75 * (sh + (sy - min((y for _, y, _, _ in heads), default=sy))), 1))
        anchor = min((y for _, y, _, _ in heads), default=sy)
        if product and product[1] + product[3] > anchor:  # the product's bottom stays 80 px inside the crop
            s = min(s, (SAFE[3] - 80 - top) / (product[1] + product[3] - anchor))
        s = min(max(s, 0.55 * cover), 2.2 * cover)
    cx = (subject[0] + subject[2] / 2) if subject else pw / 2
    if heads:
        cx = sum(x + w / 2 for x, _, w, _ in heads) / len(heads) if not subject else cx
    ox = W / 2 - cx * s
    if pw * s >= W:
        ox = min(0, max(W - pw * s, ox))
    anchor = min((y for _, y, _, _ in heads), default=subject[1] if subject else 0)
    oy = top - anchor * s if (heads or subject) else 0
    if not punch_in:
        oy = min(max(0, oy), 0.6 * H)  # the original framing only ever moves down
    if subject and subject[1] + subject[3] >= 0.97 * ph:  # the subject is cut by the frame's bottom: keep that edge off-canvas
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


def picture(spec: Spec, frame: Image.Image, bottom: int, ground: str, faces: list[Box], mask: np.ndarray | None,
            product: Box | None = None) -> tuple[np.ndarray, list[PxBox], tuple[float, float, float], bool]:
    """The picture layers (1 and 2) as a float RGB array, the heads in canvas pixels, the transform used, and whether
    the subject was cut out. It is cut out only when it is a person or the product the video is about: on key art or
    a busy scene the most salient blob is often a glow or a logo, and a sticker of that looks broken."""
    src = frame.convert("RGB")
    pw, ph = src.size
    heads_src = [head((round(x * pw), round(y * ph), round(w * pw), round(h * ph))) for x, y, w, h in faces]
    subject = _bbox(mask) if mask is not None else None
    if subject and not heads_src:
        pb = product and (product[0] * pw, product[1] * ph, (product[0] + product[2]) * pw, (product[1] + product[3]) * ph)
        if not (pb and overlap(pb, _xyxy(subject))):
            subject = None  # no person, and not the product: nothing to cut out or zoom in on
    prod_src = product and (round(product[0] * pw), round(product[1] * ph), round(product[2] * pw), round(product[3] * ph))
    s, ox, oy = transform(pw, ph, bottom, heads_src, subject, spec.punch_in, prod_src or None)
    sharp_img = _graded(src) if spec.punch_in else src
    sharp = _warp(np.asarray(sharp_img, np.float32), s, ox, oy)
    valid = _warp(np.ones((ph, pw), np.float32), s, ox, oy, cv2.INTER_LINEAR)
    # the background: the frame covering the canvas, heavily blurred and tinted with the game's background
    cover = max(W / pw, H / ph)
    back = _warp(np.asarray(src, np.float32), cover, (W - pw * cover) / 2, (H - ph * cover) / 2)
    back = cv2.GaussianBlur(back, (0, 0), 28)
    back = back * 0.6 + np.float32(rgb(ground)) * 0.4
    cut = spec.pop and mask is not None and subject is not None
    if cut:
        m = _warp(mask, s, ox, oy, cv2.INTER_LINEAR) * valid
        solid = (m > 0.5).astype(np.uint8)
        solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        out = cv2.dilate(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))).astype(np.float32)
        glow = cv2.GaussianBlur(cv2.dilate(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))).astype(np.float32), (0, 0), 22)
        alpha = cv2.GaussianBlur(solid.astype(np.float32), (0, 0), 1.6)[..., None]
        base = back * (1 - 0.85 * glow[..., None]) + np.float32(rgb(spec.accent or YELLOW)) * 0.85 * glow[..., None]
        base = np.where(out[..., None] > 0.5, np.float32(rgb(WHITE)), base)
        base = base * (1 - alpha) + sharp * alpha
    else:  # the sharp picture over the background, its edges feathered
        v = cv2.GaussianBlur(cv2.erode(valid, np.ones((25, 25), np.uint8)), (0, 0), 12)[..., None]
        base = back * (1 - v) + sharp * v
    if spec.punch_in:
        base = _vignette(base)
    heads = [(round(x * s + ox), round(y * s + oy), round(w * s), round(h * s)) for x, y, w, h in heads_src]
    return base, heads, (s, ox, oy), cut


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


def _draw_marks(d: ImageDraw.ImageDraw, marks: dict, accent: str) -> dict[str, tuple[int, int, int, int]]:
    boxes = {}
    edge = ink_on(accent)
    if "ring" in marks:
        r = marks["ring"]
        d.ellipse(r, outline=edge, width=22)
        d.ellipse(r, outline=accent, width=12)
        boxes["ring"] = r
    if "arrow" in marks:
        pts = marks["arrow"]
        d.line(pts, fill=edge, width=34, joint="curve")
        d.line(pts, fill=accent, width=20, joint="curve")
        (x1, y1), (x2, y2) = pts[-4], pts[-1]
        ang, size = np.arctan2(y2 - y1, x2 - x1), 64
        tip = (x2 + 14 * np.cos(ang), y2 + 14 * np.sin(ang))
        wing = lambda a: (tip[0] - size * np.cos(ang - a), tip[1] - size * np.sin(ang - a))  # noqa: E731
        d.polygon([tip, wing(0.5), wing(-0.5)], fill=accent, outline=edge, width=5)
        xs, ys = [p[0] for p in pts] + [tip[0]], [p[1] for p in pts] + [tip[1]]
        boxes["arrow"] = (round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys)))
    return boxes


def _wave(d: ImageDraw.ImageDraw, accent: str) -> None:
    xs = np.linspace(0, W, 60)
    for base, amp, phase, colour in ((1640, 26, 0.0, accent), (1676, 30, 1.3, BLUE)):
        d.polygon([(float(x), base + amp * np.sin(x / W * 2 * np.pi + phase)) for x in xs] + [(W, H), (0, H)], fill=colour)


def _pill(d: ImageDraw.ImageDraw, y: int, left: str | None, right: str | None, accent: str) -> tuple[int, int, int, int]:
    f = font("bold", 34)
    pad, h = 32, PILL_H
    parts = [t for t in (left, right) if t]
    widths = [round(d.textlength(t, font=f)) + 2 * pad for t in parts]
    x = (W - sum(widths)) // 2
    if len(parts) == 2:  # white half with a blue outline, then the filled accent half
        d.rounded_rectangle([x, y, x + sum(widths), y + h], radius=h // 2, fill=WHITE, outline=BLUE, width=3)
        d.rounded_rectangle([x + widths[0], y, x + sum(widths), y + h], radius=h // 2, fill=accent)
        d.rectangle([x + widths[0], y, x + widths[0] + h // 2, y + h], fill=accent)
        d.text((x + widths[0] / 2, y + h / 2), parts[0], font=f, fill=BLUE, anchor="mm")
        d.text((x + widths[0] + widths[1] / 2, y + h / 2), parts[1], font=f, fill=ink_on(accent), anchor="mm")
    else:
        d.rounded_rectangle([x, y, x + widths[0], y + h], radius=h // 2, fill=accent)
        d.text((x + widths[0] / 2, y + h / 2), parts[0], font=f, fill=ink_on(accent), anchor="mm")
    return x, y, x + sum(widths), y + h


def _words(spec: Spec, light: bool, accent: str) -> tuple[Image.Image, dict, dict]:
    """The words on their own transparent layer, with a drop shadow; their boxes and sizes."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    boxes: dict[str, tuple[int, int, int, int]] = {}
    colour1 = NAVY if light else WHITE
    stroke2 = INK if lum(rgb(accent)) > 0.3 else WHITE  # the big line always gets a thick outline: the sticker look
    y, px1 = TOP, None
    if spec.name_tag:
        f = font("bold", 34)
        tw = round(d.textlength(spec.name_tag.upper(), font=f))
        boxes["name_tag"] = ((W - tw) // 2 - 26, y, (W + tw) // 2 + 26, y + 54)
        d.rounded_rectangle(boxes["name_tag"], radius=27, outline=colour1, width=3)
        d.text((W / 2, y + 27), spec.name_tag.upper(), font=f, fill=colour1, anchor="mm")
        y += 70
    if spec.line1:
        text = spec.line1.upper()
        f1, px1 = _fit(d, text, "bold", 76, 40, 920)
        x = (W - d.textlength(text, font=f1)) / 2
        hi = (spec.highlight or "").upper().strip(" ,.!?")
        for word in text.split(" "):
            d.text((x, y), word, font=f1, fill=accent if hi and word.strip(",.!?") == hi else colour1)
            x += d.textlength(word + " ", font=f1)
        boxes["line1"] = d.textbbox((W / 2, y), text, font=f1, anchor="mt")
        boxes["line1"] = (boxes["line1"][0], y, boxes["line1"][2], boxes["line1"][3])
        y += round(px1 * 1.05)
    lines, f2, px2 = _line2(d, spec.line2.upper())
    sw = max(4, round(px2 * 0.05))
    for i, line in enumerate(lines):
        yy = y + i * round(px2 * 0.95)
        boxes[f"line2_{i}"] = d.textbbox((W / 2, yy), line, font=f2, anchor="mt", stroke_width=sw)
        d.text((W / 2, yy), line, font=f2, fill=accent, anchor="mt", stroke_width=sw, stroke_fill=stroke2)
    y += round(px2 * 0.95) * len(lines)
    if spec.pill_left or spec.pill_right:
        boxes["pill"] = _pill(d, y + 30, spec.pill_left, spec.pill_right, accent)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(layer.getchannel("A").point(lambda v: int(v * 0.6)).filter(ImageFilter.GaussianBlur(12)))
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.alpha_composite(shadow, (0, 10))
    out.alpha_composite(layer)
    return out, boxes, {"line1_px": px1, "line2_px": px2, "stroke": stroke2, "colour1": colour1}


# --- the cover ----------------------------------------------------------------------------------

def draw(spec: Spec, frame: Image.Image, faces: list[Box] = (), mask: np.ndarray | None = None, product: Box | None = None,
         rating_img: Path | None = None) -> tuple[Image.Image, dict]:
    """The cover and its report: the box of every word and mark, the heads, and the checks. `faces` and `mask` come
    from the frame (studio.frame); `product` is the object the video is about, as the look model found it."""
    bottom = headline_bottom(spec)
    src = frame.convert("RGB")
    light = lum(rgb(spec.bg)) > 0.4 if spec.bg else float(np.asarray(src.convert("L").resize((64, 64))).mean()) > 118
    ground, accent = spec.bg or (WHITE if light else NAVY), spec.accent or YELLOW
    base, heads, (s, ox, oy), cut = picture(spec, src, bottom, ground, list(faces), mask, product)
    end = wash_end(bottom, heads)
    base = _wash(base, bottom, ground, end)
    canvas = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(canvas)
    if (spec.wave if spec.wave is not None else light) and not any(t + h > WAVE_TOP for _, t, _, h in heads):
        _wave(d, accent)
    under = canvas.copy()  # what the words are checked against
    words, boxes, sizes = _words(spec, light, accent)
    marks: dict = {}
    if spec.arrow and product:
        pw, ph = src.size
        px = (round(product[0] * pw * s + ox), round(product[1] * ph * s + oy), round(product[2] * pw * s), round(product[3] * ph * s))
        marks = plan_marks(px, heads, list(boxes.values()), bottom)
        boxes.update(_draw_marks(d, marks, accent))
    canvas = canvas.convert("RGBA")
    canvas.alpha_composite(words)
    if rating_img and rating_img.exists():
        badge = Image.open(rating_img).convert("RGBA")
        badge = badge.resize((110, round(badge.height * 110 / badge.width)))
        low, high = SAFE[3] - 80 - badge.height, SAFE[3] - 420 - badge.height
        left, right = SAFE[0] + 10, SAFE[2] - 10 - badge.width
        spots = [(left, low), (right, low), (left, high), (right, high)]  # lower corners first, then higher up
        taken = [_xyxy(hd) for hd in heads] + ([boxes["ring"]] if "ring" in boxes else []) \
            + [(px - 24, py - 24, px + 24, py + 24) for px, py in marks.get("arrow", [])]  # the arrow's line, not its box
        bx, by = next((xy for xy in spots if not any(overlap((xy[0], xy[1], xy[0] + badge.width, xy[1] + badge.height), t)
                                                     for t in taken)), spots[0])
        canvas.alpha_composite(badge, (bx, by))
        boxes["rating"] = (bx, by, bx + badge.width, by + badge.height)
    rep = {"boxes": boxes, "heads": heads, "wash_end": end, "light": light, "cut": cut, **sizes}
    rep["checks"] = checks(spec, rep, under, accent)
    return canvas.convert("RGB"), rep


def _mean(img: Image.Image, box) -> tuple[int, int, int]:
    l, t, r, b = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
    if r <= l or b <= t:
        return 255, 255, 255
    return tuple(int(v) for v in np.asarray(img.crop((l, t, r, b))).reshape(-1, 3).mean(0))


def checks(spec: Spec, rep: dict, under: Image.Image, accent: str) -> list[dict]:
    out = []
    l, t, r, b = SAFE
    outside = [k for k, (x0, y0, x1, y1) in rep["boxes"].items() if x0 < l or y0 < t or x1 > r or y1 > b]
    out.append({"id": "safe_zone", "ok": not outside,
                "detail": "Everything sits inside the profile grid's 3:4 crop." if not outside else f"Cut off in the profile grid: {', '.join(outside)}."})
    worst = None
    for k, box in rep["boxes"].items():
        if k == "line1":
            c = contrast(rgb(rep["colour1"]), _mean(under, box))
        elif k.startswith("line2"):
            c = max(contrast(rgb(accent), _mean(under, box)), contrast(rgb(rep["stroke"]), rgb(accent)))
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
    return out
