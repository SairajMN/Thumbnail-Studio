"""PEGI labels and verified game ratings (kit/kit.json, copied from Guideline Copilot's brand kit). A cover shows a game's
label only when the kit has that game's rating; a game with no rating gets none."""

import json
import re
from functools import cache
from pathlib import Path

FOLDER = Path(__file__).with_name("kit")


@cache
def load() -> dict:
    return json.loads((FOLDER / "kit.json").read_text())


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def games_in(text: str) -> set[str]:
    """Kit games a text names: every word of the game's main title (before any colon) or of one of its aliases."""
    words = _words(text)
    return {g["game"] for g in load()["game_ratings"]
            if any(_words(n) and _words(n) <= words for n in [g["game"].split(":")[0], *g.get("aliases", [])])}


def rating_for(text: str) -> str | None:
    named = games_in(text)
    rated = sorted(g["image"] for g in load()["game_ratings"] if g["game"] in named and g.get("rating") and g.get("image"))
    return rated[0] if rated else None


def label_path(name: str | None) -> Path | None:
    item = load()["images"].get(name or "")
    return FOLDER / item["file"] if item else None
