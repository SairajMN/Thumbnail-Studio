"""The designs: 18 recipes taken from the store's best covers (PLAN_CHAT_STUDIO.md, section 3), all drawn by code from
the user's frame. A recipe names a background, how the subject is shown, how the words sit, and the decorations;
design.py does the drawing, the face protection and the checks for all of them alike.

Subject treatments:
- cutout: the person or product cut out, white outline and a glow, over the background
- pedestal: the product cut out standing on a podium (glossy, cube, rock or drape)
- silhouette: the subject cut out and filled near-black, in front of a sun or a light beam
- card: the frame in a white-bordered card, tilted a little
- frame: the frame itself, zoomed in on the subject
- collage: the frame and up to 3 more images the user adds, in a grid
A design whose subject can't be cut out (no person, no product) falls back to its card or the frame; a collage with
no images added shows the frame as a panel.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from studio.palette import BLUE, NAVY, WHITE, YELLOW, INK, Pal, rgb

W, H = 1080, 1920


@dataclass(frozen=True)
class Recipe:
    id: str
    name: str
    source: str  # the reference covers it comes from
    blurb: str
    background: str  # studio | flood | gradient | spotlight | space | sunset | blur | frame | pattern
    subject: str  # cutout | pedestal | silhouette | card | frame | collage
    podium: str | None = None  # glossy | cube | rock | drape
    align: str = "center"  # center | left
    spaced: bool = False  # the small line letter-spaced, a title look
    hollow: bool = False  # the big line drawn as an outline only
    decor: tuple[str, ...] = ()  # behind the subject: swoosh shapes stars sparkles confetti beam sun grain
    wave: bool = False
    colors: dict = field(default_factory=dict)  # the design's own colours, by role (palette.ROLES)
    sample: tuple[str, str] = ("SMALL LINE", "BIG LINE")  # the words a preview shows before any are written
    fits: tuple[str, ...] = ()  # person | product | art | group | event | offer: what frame it suits
    badge: bool = False  # the offer as a price tag, when the cover has one


DESIGNS: dict[str, Recipe] = {r.id: r for r in [
    Recipe("staff_hook", "Staff Hook", "FC27 tournament host, Happy Customer",
           "A person cut out on a white studio with blue and yellow swooshes", "studio", "cutout",
           decor=("swoosh", "shapes"), wave=True, sample=("INTRODUCING", "THE GROUNDS"), fits=("person",)),
    Recipe("question_hook", "Question Hook", "Which's your favourite childhood game?",
           "A person cut out against a starry space theme, with a question to the viewer", "space", "cutout",
           decor=("stars", "sparkles"), sample=("WHICH'S YOUR FAVOURITE", "GAME?"), fits=("person",)),
    Recipe("studio_pedestal", "Studio Pedestal", "Street Fighter 6, Final Fantasy Resonance",
           "The product on a glossy white podium in a soft studio", "studio", "pedestal", podium="glossy",
           sample=("AVAILABLE NOW", "STREET FIGHTER 6"), fits=("product",)),
    Recipe("spotlight_unbox", "Spotlight Unbox", "T.Flight Hotas 5, Thrustmaster 248R",
           "The product on a cube under one spotlight in the dark", "spotlight", "pedestal", podium="cube",
           decor=("beam",), colors={"big": WHITE, "highlight": "#5AA9FF"}, sample=("LET'S UNBOX", "T.FLIGHT HOTAS"), fits=("product",)),
    Recipe("colour_flood", "Colour Flood", "Wolverine 'Unbox & Tag Us'",
           "The product on a dark rock over a flood of the game's colour", "flood", "pedestal", podium="rock",
           decor=("grain",), colors={"bg": YELLOW, "big": INK, "small": INK, "pill": BLUE}, sample=("UNBOX & TAG US", "WOLVERINE"), fits=("product",)),
    Recipe("key_art", "Key Art Cinematic", "Assassin's Creed Black Flag, Wolverine pre-order",
           "The frame full-bleed and moody, the title on top", "frame", "frame",
           colors={"big": WHITE, "pill": YELLOW}, sample=("PRE-ORDER NOW", "BLACK FLAG"), fits=("art",)),
    Recipe("midnight_launch", "Midnight Launch", "Wolverine Midnight Launch",
           "The product on a blue drape under a night sky", "gradient", "pedestal", podium="drape",
           decor=("stars",), colors={"bg": NAVY, "big": YELLOW, "small": "#7FB2FF"}, sample=("MIDNIGHT", "LAUNCH"), fits=("product", "event")),
    Recipe("group_event", "Group Event", "FC26 Tournament, Midnight Launch crowd",
           "The group cut out on white with confetti", "studio", "cutout",
           decor=("confetti",), wave=True, sample=("FC26", "TOURNAMENT"), fits=("group", "event")),
    Recipe("winner_card", "Winner Card", "Tokon Tournament winner, Lucky Draw winner",
           "The winner cut out on deep blue with sparkles and a name tag", "gradient", "cutout",
           decor=("sparkles", "shapes"), colors={"bg": "#0B2A8A", "big": YELLOW}, sample=("TOKON TOURNAMENT", "WINNER"), fits=("person", "event")),
    Recipe("event_flyer", "Event Flyer", "The Battle Begins: Tokon tournament",
           "The frame in a tilted card on a warm light poster", "flood", "card",
           decor=("shapes",), colors={"bg": "#F3EDE3", "big": YELLOW, "small": INK}, sample=("TOKON TOURNAMENT", "THE BATTLE BEGINS"), fits=("event",)),
    Recipe("interview_moment", "Interview Moment", "Moments from the Wolverine launch",
           "The candid frame, zoomed in, under a two-line caption", "blur", "frame",
           decor=("shapes",), sample=("MOMENTS FROM", "WOLVERINE LAUNCH"), fits=("person", "group", "event")),
    Recipe("date_teaser", "Date Teaser", "007 First Light, 27.05.2026",
           "A dark silhouette in a beam of light, a spaced title and a big date", "spotlight", "silhouette",
           spaced=True, decor=("beam",), colors={"big": WHITE, "small": "#D8D8D8", "glow": "#FFE7A8"}, sample=("FIRST LIGHT", "27.05.2026"), fits=("art", "person")),
    Recipe("framed_card", "Framed Card", "Mortal Kombat Tournament",
           "The frame in a white card over a colour flood, the big word hollow", "flood", "card", hollow=True,
           colors={"bg": BLUE, "big": YELLOW, "small": YELLOW}, sample=("MORTAL KOMBAT", "TOURNAMENT"), fits=("event", "group")),
    Recipe("sunset_poster", "Sunset Poster", "Red Dead Redemption",
           "The subject as a silhouette on a huge sun in a red sky", "sunset", "silhouette",
           decor=("sun",), colors={"bg": "#5A0A0A", "big": "#FFF1D6", "small": "#FFC23D", "glow": "#FFB000"}, sample=("THE WEST IS BACK", "RED DEAD"), fits=("art", "person")),
    Recipe("celebration", "Celebration", "10 Years of Uncharted 4, 50K Family",
           "The frame full-bleed with confetti, a left-aligned 'celebrating' headline", "frame", "frame", align="left",
           decor=("confetti",), colors={"big": YELLOW}, sample=("CELEBRATING", "10 YEARS"), fits=("art", "group", "event")),
    Recipe("hand_product", "Hand + Product", "WD SN850 'Play More, Store More'",
           "The product in hand, cut out over a pattern of its name", "pattern", "cutout",
           wave=True, colors={"big": BLUE, "small": NAVY, "highlight": BLUE}, sample=("PLAY MORE,", "STORE MORE"), fits=("product",)),
    Recipe("collage", "Collage", "Pre-Order Now: First Light, Kingdom Come, LEGO Batman, Tokon",
           "Your frame and up to 3 more images in a grid on a light poster", "flood", "collage",
           decor=("shapes",), wave=True, colors={"bg": "#F4F1EA", "big": YELLOW, "small": INK, "pill": YELLOW},
           sample=("PRE-ORDER", "NOW"), fits=("event", "product")),
    Recipe("offer", "Offer", "PS Wallet Top-ups, Days of Play sale",
           "The product cut out on blue with a white swoosh and the offer on a price tag", "gradient", "cutout",
           decor=("swoosh",), badge=True, colors={"bg": "#1D2BD8", "big": WHITE, "small": "#BFD4FF", "pill": YELLOW, "glow": "#36E2C8"},
           sample=("THIS WEEK ONLY", "DAYS OF PLAY"), fits=("offer", "product")),
]}

# the writer's template -> the design a new cover starts on
FROM_TEMPLATE = {"staff_hook": "staff_hook", "pedestal": "studio_pedestal", "key_art": "key_art", "group": "group_event",
                 "winner": "winner_card", "flyer": "event_flyer"}


def wash_of(recipe: Recipe):
    """The colour behind the words for this design, given the background colour: designs with a background of their
    own (a studio, the night, a sunset) keep it whatever colour was chosen."""
    fixed = {"studio": "#F2F5FA", "pattern": "#F2F5FA", "spotlight": "#08090D", "space": "#0B0F2E", "sunset": "#5A0A0A"}
    return lambda bg: fixed.get(recipe.background, bg)


# --- backgrounds --------------------------------------------------------------------------------

def _vgrad(*stops: str) -> np.ndarray:
    ys = np.linspace(0, 1, H, dtype=np.float32)
    pos = np.linspace(0, 1, len(stops))
    cols = np.float32([rgb(c) for c in stops])
    g = np.stack([np.interp(ys, pos, cols[:, k]) for k in range(3)], -1)
    return np.broadcast_to(g[:, None, :], (H, W, 3)).copy()


def _mix(a: str, b: str, t: float) -> str:
    ca, cb = rgb(a), rgb(b)
    return "#%02X%02X%02X" % tuple(round(x + (y - x) * t) for x, y in zip(ca, cb))


def _radial(cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.clip(1 - np.hypot(xx - cx, yy - cy) / r, 0, 1)[..., None]


def background(recipe: Recipe, pal: Pal, frame_cover: np.ndarray, words: str) -> np.ndarray:
    """The design's background as a float RGB array. `frame_cover` is the frame filling the canvas (for blur and
    frame backgrounds); `words` the big line (for the pattern)."""
    kind = recipe.background
    if kind == "studio" or kind == "pattern":
        a = _vgrad("#FFFFFF", "#E3E8F1") * (1 - 0.25 * _radial(W / 2, H * 0.45, 900)) + 255 * 0.25 * _radial(W / 2, H * 0.45, 900)
        if kind == "pattern" and words:  # the name, faint, row after row (on its own layer: drawing translucent text
            from studio.design import font_for  # straight onto the picture would not blend)
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            txt = (words.upper() + "   ") * 5
            f = font_for("heavy", 120, txt)
            for i, y in enumerate(range(150, H, 165)):
                d.text((-80 - (i % 2) * 260, y), txt, font=f, fill=rgb(pal.big) + (26,))
            img.alpha_composite(layer)
            a = np.asarray(img.convert("RGB"), np.float32)
        return a
    if kind == "flood":
        return np.float32(rgb(pal.bg)) * (1 - 0.18 * _radial(W / 2, H * 0.55, 1000)) + 255 * 0.18 * _radial(W / 2, H * 0.55, 1000)
    if kind == "gradient":  # a blue night deepens towards PlayStation blue; any other colour towards its own dark
        r, g, b = rgb(pal.bg)
        mid = _mix(pal.bg, BLUE, 0.55) if b > r and b > g else _mix(pal.bg, "#000000", 0.25)
        return _vgrad(pal.bg, mid, _mix(pal.bg, "#000000", 0.45))
    if kind == "spotlight":
        return _vgrad(_mix("#08090D", pal.bg, 0.12), "#050507")
    if kind == "space":
        return _vgrad(_mix("#0B0F2E", pal.bg, 0.25), "#2A1466", "#4B1C7A")
    if kind == "sunset":
        return _vgrad("#5A0A0A", "#9E1B12", "#E8551D", "#FFB000")
    tint = np.float32(rgb(pal.bg))
    if kind == "blur":
        return cv2.GaussianBlur(frame_cover, (0, 0), 28) * 0.6 + tint * 0.4
    return frame_cover * 0.62 + tint * 0.12  # frame: the picture itself, darkened for the words


# --- decorations (behind the subject) -----------------------------------------------------------

def _shape_outline(d: ImageDraw.ImageDraw, kind: str, x: float, y: float, r: float, colour) -> None:
    w = max(4, int(r * 0.18))
    if kind == "circle":
        d.ellipse((x - r, y - r, x + r, y + r), outline=colour, width=w)
    elif kind == "square":
        d.rectangle((x - r * 0.8, y - r * 0.8, x + r * 0.8, y + r * 0.8), outline=colour, width=w)
    elif kind == "triangle":
        d.polygon([(x, y - r), (x + r * 0.9, y + r * 0.6), (x - r * 0.9, y + r * 0.6)], outline=colour, width=w)
    else:  # cross
        d.line((x - r * 0.7, y - r * 0.7, x + r * 0.7, y + r * 0.7), fill=colour, width=w)
        d.line((x - r * 0.7, y + r * 0.7, x + r * 0.7, y - r * 0.7), fill=colour, width=w)


def decorate(a: np.ndarray, recipe: Recipe, pal: Pal, seed: int, sun_at: tuple[float, float] | None = None) -> np.ndarray:
    """Decorations drawn into the background, so the cut-out subject always stands in front of them."""
    if not recipe.decor:
        return a
    rng = np.random.default_rng(seed)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for kind in recipe.decor:
        if kind == "swoosh":
            d.polygon([(0, H * 0.62), (W * 0.55, H * 0.44), (W * 0.62, H * 0.47), (0, H * 0.70)], fill=rgb(BLUE) + (60,))
            d.polygon([(W, H * 0.30), (W * 0.40, H * 0.52), (W * 0.46, H * 0.55), (W, H * 0.36)], fill=rgb(YELLOW) + (90,))
        elif kind == "shapes":
            for i, k in enumerate(["triangle", "circle", "cross", "square", "triangle", "circle"]):
                x, y = rng.uniform(70, W - 70), rng.uniform(H * 0.32, H * 0.86)
                _shape_outline(d, k, x, y, rng.uniform(30, 64), rgb(pal.glow) + (70,))
        elif kind == "stars":
            for _ in range(140):
                x, y, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(1, 3.4)
                d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, int(rng.uniform(80, 220))))
        elif kind == "sparkles":
            for _ in range(14):
                x, y, r = rng.uniform(60, W - 60), rng.uniform(H * 0.3, H * 0.9), rng.uniform(14, 34)
                d.polygon([(x, y - r), (x + r * 0.22, y - r * 0.22), (x + r, y), (x + r * 0.22, y + r * 0.22), (x, y + r),
                           (x - r * 0.22, y + r * 0.22), (x - r, y), (x - r * 0.22, y - r * 0.22)], fill=rgb(pal.glow) + (200,))
        elif kind == "confetti":
            colours = [rgb(c) for c in (pal.big, pal.highlight, BLUE, "#FF4F7B", "#36E2C8", WHITE)]
            for _ in range(90):
                x, y, s = rng.uniform(0, W), rng.uniform(H * 0.25, H), rng.uniform(10, 22)
                ang = rng.uniform(0, np.pi)
                pts = [(x + np.cos(ang + t) * s, y + np.sin(ang + t) * s * 0.45) for t in (0, np.pi / 2, np.pi, 3 * np.pi / 2)]
                d.polygon(pts, fill=colours[int(rng.integers(len(colours)))] + (230,))
        elif kind == "beam":
            beam = Image.new("L", (W, H), 0)
            ImageDraw.Draw(beam).polygon([(W / 2 - 70, 0), (W / 2 + 70, 0), (W / 2 + 470, H * 0.92), (W / 2 - 470, H * 0.92)], fill=70)
            beam = beam.filter(ImageFilter.GaussianBlur(60))
            over.alpha_composite(Image.merge("RGBA", (*Image.new("RGB", (W, H), rgb(pal.glow)).split(), beam)))
            d = ImageDraw.Draw(over)
        elif kind == "sun" and sun_at:
            x, y, r = sun_at[0], sun_at[1], W * 0.42
            for k in range(6, 0, -1):  # a soft glow, then the disc
                rr = r * (1 + k * 0.08)
                d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=rgb(pal.glow) + (int(18 + 4 * (6 - k)),))
            d.ellipse((x - r, y - r, x + r, y + r), fill=rgb("#FFD24A") + (255,))
    img.alpha_composite(over)
    out = np.asarray(img.convert("RGB"), np.float32)
    if "grain" in recipe.decor:
        out = out + rng.normal(0, 5, out.shape[:2])[..., None].astype(np.float32)
    return out


# --- podiums ------------------------------------------------------------------------------------

def podium(a: np.ndarray, kind: str, cx: float, top: float, width: float) -> np.ndarray:
    """A podium whose top face sits at `top`, under the subject, running off the canvas bottom."""
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    w2, eh = width / 2, width * 0.13
    face, body = {"glossy": ("#F6F8FB", "#C9D0DC"), "cube": ("#EEF1F6", "#B9C1CF"), "rock": ("#3A3A3A", "#1C1C1C"),
                  "drape": ("#2F6BFF", "#0A2C8F")}[kind]
    if kind == "cube":
        d.rectangle((cx - w2, top, cx + w2, H), fill=rgb(body) + (255,))
        d.polygon([(cx - w2, top), (cx + w2, top), (cx + w2 - 40, top - eh), (cx - w2 + 40, top - eh)], fill=rgb(face) + (255,))
    elif kind == "drape":
        for i in range(9):  # folds: alternating light and dark bands
            x0 = cx - w2 * 1.15 + i * width * 1.15 / 9 * 2 / 2
            d.rectangle((x0, top, x0 + width * 1.15 / 9 + 2, H), fill=rgb(body if i % 2 else face) + (255,))
        over = over.filter(ImageFilter.GaussianBlur(10))
        d = ImageDraw.Draw(over)
        d.ellipse((cx - w2 * 1.15, top - eh, cx + w2 * 1.15, top + eh), fill=rgb(face) + (255,))
    else:
        d.rectangle((cx - w2, top, cx + w2, H), fill=rgb(body) + (255,))
        d.ellipse((cx - w2, top - eh, cx + w2, top + eh), fill=rgb(face) + (255,))
        if kind == "glossy":  # a soft reflection on the top face
            d.ellipse((cx - w2 * 0.6, top - eh * 0.5, cx + w2 * 0.6, top + eh * 0.5), fill=(255, 255, 255, 120))
    shadow = over.getchannel("A").filter(ImageFilter.GaussianBlur(24)).point(lambda v: int(v * 0.35))
    img.alpha_composite(Image.merge("RGBA", (*Image.new("RGB", (W, H), (0, 0, 0)).split(), shadow)), (0, 18))
    img.alpha_composite(over)
    return np.asarray(img.convert("RGB"), np.float32)
