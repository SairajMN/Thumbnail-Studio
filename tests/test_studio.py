"""Covers with fake Nebius replies: drawing rules, the flow, Nebius-only fallbacks, the budget, edits."""

import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from studio import covers, design, nebius
from studio.app import app
from studio.config import settings
from studio.design import SAFE, Spec


def _frame(size=(720, 1280), colour=(40, 60, 120)) -> Image.Image:
    """A plain frame with a bright box in the middle, so the cut-out has a subject."""
    img = Image.new("RGB", size, colour)
    ImageDraw.Draw(img).rectangle((size[0] * 0.3, size[1] * 0.45, size[0] * 0.7, size[1] * 0.75), fill=(240, 240, 240))
    return img


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _mask(size=(720, 1280)) -> np.ndarray:
    m = np.zeros((size[1], size[0]), np.float32)
    m[int(size[1] * 0.25):, int(size[0] * 0.25):int(size[0] * 0.75)] = 1  # a person-shaped block from a quarter down
    return m


def _inside(b) -> bool:
    return b[0] >= SAFE[0] and b[1] >= SAFE[1] and b[2] <= SAFE[2] and b[3] <= SAFE[3]


SPEC = Spec(line1="You won't believe", line2="This SSD", highlight="believe", pill_left="Available", pill_right="Now",
            bg="#0A0A2E", accent="#CDFD4A")


def test_a_cover_with_every_effect_keeps_words_and_marks_off_the_face():
    face = [(0.38, 0.27, 0.24, 0.12)]
    product = (0.4, 0.62, 0.2, 0.1)
    img, rep = design.draw(SPEC, _frame(), face, _mask(), product, design.Path("studio/kit/pegi_16.jpg"))
    assert img.size == (1080, 1920)
    assert {"ring", "arrow", "rating"} <= set(rep["boxes"]) and all(_inside(b) for b in rep["boxes"].values())
    assert all(c["ok"] for c in rep["checks"]), rep["checks"]
    words_bottom = max(b[3] for k, b in rep["boxes"].items() if k.startswith(("line", "pill")))
    assert all(t >= words_bottom + design.GAP - 1 and t >= rep["wash_end"] for _, t, _, _ in rep["heads"])


def test_punch_in_makes_the_subject_bigger_and_pop_draws_an_outline():
    face, mask = [(0.45, 0.4, 0.1, 0.06)], _mask()
    _, small = design.draw(SPEC.model_copy(update={"punch_in": False}), _frame(), face, mask)
    _, big = design.draw(SPEC, _frame(), face, mask)
    assert big["heads"][0][2] > small["heads"][0][2] * 1.2
    flat, _ = design.draw(SPEC.model_copy(update={"pop": False, "punch_in": False}), _frame(), face, mask)
    popped, rep = design.draw(SPEC.model_copy(update={"punch_in": False}), _frame(), face, mask)
    white = lambda im: (np.asarray(im) > 250).all(-1).sum()  # noqa: E731  the outline is pure white
    assert white(popped) > white(flat) + 2000


def test_no_cut_out_without_a_person_or_the_product():
    plain, _ = design.draw(SPEC.model_copy(update={"pop": False}), _frame(), [], _mask())
    no_subject, _ = design.draw(SPEC, _frame(), [], _mask())  # a salient blob, but no face and no product
    assert np.abs(np.asarray(plain, np.int16) - np.asarray(no_subject, np.int16)).mean() < 1


def test_the_arrow_is_left_out_when_every_route_crosses_a_face():
    faces = [(0.0, 0.78, 0.28, 0.1), (0.72, 0.78, 0.28, 0.1)]  # faces where both arrows would start
    _, rep = design.draw(SPEC.model_copy(update={"punch_in": False}), _frame(), faces, _mask(), (0.42, 0.66, 0.16, 0.08))
    assert "arrow" not in rep["boxes"]
    assert next(c for c in rep["checks"] if c["id"] == "face_clear")["ok"]


LOOK = {"game": "WD_BLACK SN850P", "seen": "A hand holds the WD_BLACK SN850P SSD box.", "background": "#0a0a2e", "accent": "#cdfd4a",
        "product_box": [0.4, 0.5, 0.2, 0.2], "hook": "Never delete a game again: 2TB at 7,300 MB/s"}
WORDS = {"template": "pedestal", "line1": "Never delete a game", "line2": "Again", "highlight": "delete", "pill_left": "Available",
         "pill_right": "Now", "name_tag": None, "arrow": True, "why": "a pain every PS5 owner has"}


@pytest.fixture
def fake(monkeypatch):
    seen = {"calls": [], "fail": set()}

    def reply(model, system, text, image):
        seen["calls"].append((model, image))
        if model in seen["fail"]:
            raise RuntimeError("provider down")
        return LOOK if "<frame_read_by_code>" in text else WORDS

    monkeypatch.setattr(nebius, "FAKE", reply)
    return seen


def _make(c, context="WD_BLACK SN850P 2TB, 7,300 MB/s, available now"):
    r = c.post("/api/covers", files={"frame": ("f.png", _png(_frame()), "image/png")}, data={"context": context})
    assert r.status_code == 201, r.text
    for _ in range(100):  # TestClient runs the background task before returning, but keep this honest
        got = c.get(f"/api/covers/{r.json()['id']}").json()
        if got["status"] in ("done", "failed"):
            return got
        time.sleep(0.05)
    raise AssertionError("cover never finished")


def test_frame_and_context_become_a_cover_kimi_looks_glm_writes(fake):
    c = TestClient(app)
    opts = c.get("/api/options").json()
    assert [m["model"] for m in opts["models"]][1:3] == ["Kimi K3", "GLM 5.3"] and 0 < opts["est_usd"] < opts["budget_usd"] == 0.15
    cov = _make(c)
    assert cov["status"] == "done", cov["error"]
    assert fake["calls"] == [("kimi-k3", True), ("glm-5.3", False)]  # only the vision model gets the frame
    assert cov["look"]["hook"].startswith("Never delete") and cov["look"]["product"] == [0.4, 0.5, 0.2, 0.2]
    s = cov["spec"]
    assert (s["line2"], s["highlight"], s["bg"], s["accent"], s["arrow"]) == ("Again", "delete", "#0A0A2E", "#CDFD4A", True)
    assert [(r["step"], r["model"]) for r in cov["costs"]] == [("look", "Kimi K3"), ("write", "GLM 5.3")]
    assert cov["used_usd"] == 0 and cov["budget_usd"] == 0.15  # fake replies cost nothing
    # two versions from one run: the subject cut out, and the frame as it is
    cut, plain = (np.asarray(Image.open(io.BytesIO(c.get(cov["images"][v]).content)).convert("RGB")) for v in ("cut", "plain"))
    assert cut.shape == plain.shape == (1920, 1080, 3) and cov["cut_available"]
    assert np.abs(cut.astype(int) - plain.astype(int)).mean() > 5  # outline, glow and blurred background only in the cut one
    assert set(cov["checks"]) == {"cut", "plain"} and all(k["ok"] for v in cov["checks"].values() for k in v), cov["checks"]
    dl = c.get(cov["images"]["plain"] + "&download=1")
    assert "attachment" in dl.headers["content-disposition"] and "-plain.png" in dl.headers["content-disposition"]

    r = c.patch(f"/api/covers/{cov['id']}", json={"line2": "For good", "pill_left": "", "punch_in": False})
    assert r.status_code == 200, r.text
    e = r.json()
    assert (e["spec"]["line2"], e["spec"]["pill_left"], e["spec"]["punch_in"]) == ("For good", None, False)
    assert all(e["images"][v] != cov["images"][v] for v in ("cut", "plain"))  # both redrawn
    assert len(fake["calls"]) == 2  # an edit calls no model
    assert c.patch(f"/api/covers/{cov['id']}", json={"accent": "green"}).status_code == 422
    assert c.patch(f"/api/covers/{cov['id']}", json={"line2": " "}).status_code == 422


def test_fallbacks_stay_on_nebius_and_glm_never_gets_the_frame(fake):
    fake["fail"] = {"kimi-k3", "kimi-k2.6", "qwen3.8-27b", "glm-5.3"}
    cov = _make(TestClient(app))
    assert cov["status"] == "done", cov["error"]
    assert fake["calls"] == [("kimi-k3", True), ("kimi-k2.6", True), ("qwen3.8-27b", True), ("glm-5.3", False), ("glm-5.3-flash", False),
                             ("glm-5.3", False), ("glm-5.2", False)]
    assert [(r["model"], r["ok"]) for r in cov["costs"] if r["step"] == "look"] == [
        ("Kimi K3", 0), ("Kimi K2.6", 0), ("Qwen 3.8 27B", 0), ("GLM 5.3", 0), ("GLM 5.3 Flash", 1)]


def test_no_call_is_made_that_could_pass_the_budget(fake, monkeypatch):
    monkeypatch.setattr(settings, "budget_usd", 0.003)  # only GLM 5.3 Flash fits at its worst
    cov = _make(TestClient(app))
    assert cov["status"] == "done", cov["error"]
    assert fake["calls"] == [("glm-5.3-flash", False), ("glm-5.3-flash", False)]
    monkeypatch.setattr(settings, "budget_usd", 0.0001)
    cov = _make(TestClient(app))
    assert cov["status"] == "failed" and "budget" in cov["error"] and len(fake["calls"]) == 2


def test_the_cut_out_model_can_run_from_several_threads():
    from concurrent.futures import ThreadPoolExecutor

    from studio import frame
    masks = list(ThreadPoolExecutor(4).map(frame.subject_mask, [_frame() for _ in range(8)]))
    assert all(m.shape == (1280, 720) and np.allclose(m, masks[0], atol=1e-4) for m in masks)


def test_without_a_person_or_product_both_versions_are_the_same(fake):
    fake_look = {**LOOK, "product_box": None}
    nebius.FAKE = lambda model, system, text, image: fake_look if "<frame_read_by_code>" in text else WORDS
    c = TestClient(app)
    r = c.post("/api/covers", files={"frame": ("f.png", _png(_frame()), "image/png")}, data={"context": "new game"})
    cov = c.get(f"/api/covers/{r.json()['id']}").json()
    assert cov["status"] == "done" and not cov["cut_available"]
    assert c.get(cov["images"]["cut"]).content == c.get(cov["images"]["plain"]).content


def test_bad_input_is_refused():
    c = TestClient(app)
    assert c.post("/api/covers", files={"frame": ("f.txt", b"not an image", "text/plain")}).status_code == 422
    assert c.get("/api/covers/../../etc").status_code == 404
    assert c.get("/api/covers/" + "0" * 32).status_code == 404
