"""Colours by role. Every role can be set by the user (a preset, the game's colours, or picked one by one) or left to
the design; what neither sets comes from the house colours, chosen to read on the background."""

import json
import re
import threading
import uuid
from dataclasses import dataclass

from pydantic import BaseModel, Field

from studio.config import settings

HEX = r"^#[0-9A-Fa-f]{6}$"
BLUE, YELLOW, NAVY, WHITE, INK = "#0050D0", "#FFC400", "#0A1640", "#FFFFFF", "#111111"
ROLES = ("bg", "big", "small", "highlight", "pill", "glow")
ROLE_LABELS = {"bg": "Background", "big": "Big text", "small": "Small text", "highlight": "Highlighted word",
               "pill": "Pill", "glow": "Glow, ring and arrow"}


class Colors(BaseModel):
    """The roles the user can set; None leaves the role to the design."""
    bg: str | None = Field(None, pattern=HEX)
    big: str | None = Field(None, pattern=HEX)
    small: str | None = Field(None, pattern=HEX)
    highlight: str | None = Field(None, pattern=HEX)
    pill: str | None = Field(None, pattern=HEX)
    glow: str | None = Field(None, pattern=HEX)


PRESETS: dict[str, dict] = {  # name -> label and colours; "game" (the look model's pick) is added per cover
    "house": {"label": "House blue and yellow", "colors": {"bg": NAVY, "big": YELLOW, "small": WHITE, "highlight": YELLOW, "pill": YELLOW, "glow": YELLOW}},
    "house_light": {"label": "House on white", "colors": {"bg": WHITE, "big": BLUE, "small": NAVY, "highlight": YELLOW, "pill": YELLOW, "glow": BLUE}},
    "wolverine": {"label": "Wolverine yellow and black", "colors": {"bg": "#FFC400", "big": INK, "small": INK, "highlight": WHITE, "pill": INK, "glow": INK}},
    "fc": {"label": "EA FC navy and neon", "colors": {"bg": "#0A0A2E", "big": "#CDFD4A", "small": WHITE, "highlight": "#CDFD4A", "pill": "#CDFD4A", "glow": "#CDFD4A"}},
    "ghost": {"label": "Ghost white, gold and red", "colors": {"bg": "#F7F3EA", "big": "#B8862B", "small": INK, "highlight": "#C1272D", "pill": "#C1272D", "glow": "#B8862B"}},
    "assassins": {"label": "Assassin's navy and white", "colors": {"bg": "#0B1A2E", "big": WHITE, "small": "#C9D3E0", "highlight": "#E8B33D", "pill": "#E8B33D", "glow": WHITE}},
    "gta": {"label": "GTA pink and purple", "colors": {"bg": "#2B1055", "big": "#FF4FA3", "small": WHITE, "highlight": "#FFB86B", "pill": "#FF4FA3", "glow": "#FF4FA3"}},
    "red_dead": {"label": "Red Dead red and gold", "colors": {"bg": "#7A0E0E", "big": "#FFC23D", "small": "#FFE9C2", "highlight": WHITE, "pill": "#FFC23D", "glow": "#FFC23D"}},
    "days_of_play": {"label": "Days of Play blue", "colors": {"bg": "#1D2BD8", "big": WHITE, "small": "#BFD4FF", "highlight": "#36E2C8", "pill": "#36E2C8", "glow": "#36E2C8"}},
    "mono": {"label": "Black and white", "colors": {"bg": INK, "big": WHITE, "small": "#BBBBBB", "highlight": WHITE, "pill": WHITE, "glow": WHITE}},
}


# --- my palettes: colour sets the user saved, kept in data/palettes.json -------------------------

_SAVED_LOCK = threading.Lock()


def _saved_file():
    return settings.data_dir / "palettes.json"


def saved() -> list[dict]:
    p = _saved_file()
    return json.loads(p.read_text()) if p.exists() else []


def save(name: str, colors: Colors) -> dict:
    name = re.sub(r"\s+", " ", name).strip()[:40]
    if not name:
        raise ValueError("a palette needs a name")
    if not any(colors.model_dump().values()):
        raise ValueError("a palette needs at least one colour")
    item = {"id": "mine:" + uuid.uuid4().hex[:10], "label": name, "colors": colors.model_dump(), "mine": True}
    with _SAVED_LOCK:
        items = [x for x in saved() if x["label"].lower() != name.lower()] + [item]  # a name saved again replaces the old one
        _saved_file().parent.mkdir(parents=True, exist_ok=True)
        _saved_file().write_text(json.dumps(items[-50:], indent=1))
    return item


def forget(palette_id: str) -> bool:
    with _SAVED_LOCK:
        items = saved()
        kept = [x for x in items if x["id"] != palette_id]
        _saved_file().write_text(json.dumps(kept, indent=1))
    return len(kept) < len(items)


@dataclass(frozen=True)
class Pal:
    """Every role resolved to a colour, plus the wash behind the headline and whether the cover is light."""
    bg: str
    big: str
    small: str
    highlight: str
    pill: str
    glow: str
    wash: str
    light: bool


def rgb(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def lum(c) -> float:
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = rgb(c) if isinstance(c, str) else c
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(a, b) -> float:
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def ink_on(fill: str) -> str:
    """Black or white, whichever reads better on `fill`."""
    return INK if contrast(INK, fill) >= contrast(WHITE, fill) else WHITE


def resolve(chosen: Colors, design: dict, wash_of, picture_light: bool, keep: bool = True) -> Pal:
    """The user's colours first, then the design's own, then the house colours that read on the background.
    `wash_of(bg)` is the design's background colour behind the words for a given bg colour. `keep` False (a preset or
    the game's colours, not picked one by one) lets the small text and highlight change when they'd be unreadable on
    this design's background."""
    pick = lambda role: getattr(chosen, role) or design.get(role)  # noqa: E731
    bg = pick("bg") or (WHITE if picture_light else NAVY)
    wash = wash_of(bg)
    light = lum(wash) > 0.4
    big = pick("big") or (BLUE if light else YELLOW)
    small = pick("small") or (NAVY if light else WHITE)
    if (not chosen.small or not keep) and contrast(small, wash) < 3:  # a design's dark text on a background made dark (or the reverse)
        small = NAVY if light else WHITE
    highlight = pick("highlight") or (YELLOW if light else big)
    if (not chosen.highlight or not keep) and contrast(highlight, wash) < 3:  # a default that vanishes on this background: the first that reads
        highlight = next((c for c in (big, BLUE, YELLOW, ink_on(wash)) if contrast(c, wash) >= 3 and c != small), ink_on(wash))
    return Pal(bg=bg, big=big, small=small, highlight=highlight, pill=pick("pill") or (YELLOW if light else big), glow=pick("glow") or big,
               wash=wash, light=light)


def readability(pal: Pal) -> list[str]:
    """Plain-language warnings for colour pairs that are hard to read (the big line always gets an outline, so it is
    not listed)."""
    out = []
    if contrast(pal.small, pal.wash) < 3:
        out.append("The small text is hard to read on this background.")
    if pal.highlight != pal.small and contrast(pal.highlight, pal.wash) < 3:
        out.append("The highlighted word is hard to read on this background.")
    return out
