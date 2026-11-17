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
    img, rep = design.draw(SPEC, _frame(), face, _mask(), product)
    assert img.size == (1080, 1920)
    assert {"ring", "arrow"} <= set(rep["boxes"]) and all(_inside(b) for b in rep["boxes"].values())
    assert all(c["ok"] for c in rep["checks"]), rep["checks"]
    words_bottom = max(b[3] for k, b in rep["boxes"].items() if k.startswith(("line", "pill")))
    assert all(t >= words_bottom + design.GAP - 1 and t >= rep["wash_end"] for _, t, _, _ in rep["heads"])


def test_punch_in_makes_the_subject_bigger_and_pop_draws_an_outline():
    face, mask = [(0.45, 0.4, 0.1, 0.06)], _mask()
    _, small = design.draw(SPEC.model_copy(update={"punch_in": False}), _frame(), face, mask)
    _, big = design.draw(SPEC, _frame(), face, mask)
    assert big["heads"][0][2] > small["heads"][0][2] * 1.2
    flat, _ = design.draw(SPEC.model_copy(update={"punch_in": False}), _frame(), face, mask, cut=False)
    popped, rep = design.draw(SPEC.model_copy(update={"punch_in": False}), _frame(), face, mask)
    white = lambda im: (np.asarray(im) > 250).all(-1).sum()  # noqa: E731  the outline is pure white
    assert white(popped) > white(flat) + 2000


def test_no_cut_out_without_a_person_or_the_product():
    plain, _ = design.draw(SPEC, _frame(), [], _mask(), cut=False)
    no_subject, _ = design.draw(SPEC, _frame(), [], _mask())  # a salient blob, but no face and no product
    assert np.abs(np.asarray(plain, np.int16) - np.asarray(no_subject, np.int16)).mean() < 1


def test_the_arrow_is_left_out_when_every_route_crosses_a_face():
    faces = [(0.0, 0.78, 0.28, 0.1), (0.72, 0.78, 0.28, 0.1)]  # faces where both arrows would start
    _, rep = design.draw(SPEC.model_copy(update={"punch_in": False}), _frame(), faces, _mask(), (0.42, 0.66, 0.16, 0.08))
    assert "arrow" not in rep["boxes"]
    assert next(c for c in rep["checks"] if c["id"] == "face_clear")["ok"]


LOOK = {"game": "WD_BLACK SN850P", "seen": "A hand holds the WD_BLACK SN850P SSD box.", "background": "#0a0a2e", "accent": "#cdfd4a",
        "product_box": [0.4, 0.5, 0.2, 0.2], "hook": "Never delete a game again: 2TB at 7,300 MB/s",
        "designs": ["hand_product", "made_up_design", "spotlight_unbox"]}
WORDS = {"template": "pedestal", "line1": "Never delete a game", "line2": "Again", "highlight": "delete", "pill_left": "Available",
         "pill_right": "Now", "name_tag": None, "arrow": True, "why": "a pain every PS5 owner has"}


HEADLINES = {"options": [WORDS,
                         {**WORDS, "line1": "Up to", "line2": "7,300 MB/s", "highlight": "Up", "why": "a specific number"},
                         {**WORDS, "line1": "Is your PS5", "line2": "Out of space?", "highlight": "PS5", "pill_left": None, "pill_right": None,
                          "arrow": False, "why": "a question"}]}


@pytest.fixture
def fake(monkeypatch):
    seen = {"calls": [], "fail": set(), "edit": {"reply": "Shorter now.", "line1": "No more", "line2": "Deleting", "highlight": None,
                                                    "pill_left": "", "pill_right": None, "design": "made_up", "colors": {"big": "#ff0000", "nope": "#000000"},
                                                    "arrow": None, "punch_in": None}}

    def reply(model, system, text, image):
        seen["calls"].append((model, image))
        if model in seen["fail"]:
            raise RuntimeError("provider down")
        if "<message>" in text:
            return seen["edit"]
        return LOOK if "<frame_read_by_code>" in text else HEADLINES

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
    assert (s["line2"], s["highlight"], s["arrow"]) == ("Again", "delete", True)  # the first option
    # Kimi's picks, the made-up one dropped and the list filled from what the frame holds (a product, no face)
    assert cov["recommended"][:2] == ["hand_product", "spotlight_unbox"] and len(cov["recommended"]) == 3
    assert s["design"] == "hand_product" and [h["line2"] for h in cov["headlines"]] == ["Again", "7,300 MB/s", "Out of space?"]
    assert cov["headline"] == 0
    assert (s["colors"]["bg"], s["colors"]["big"], s["colors"]["highlight"]) == ("#0A0A2E", "#CDFD4A", None)  # the game's, by role
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
    assert c.patch(f"/api/covers/{cov['id']}", json={"colors": {"big": "green"}}).status_code == 422
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
    nebius.FAKE = lambda model, system, text, image: fake_look if "<frame_read_by_code>" in text else HEADLINES
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


# --- designs and colours ------------------------------------------------------------------------

from studio import designs, palette  # noqa: E402


@pytest.mark.parametrize("design_id", list(designs.DESIGNS))
def test_every_design_draws_both_versions_without_covering_a_face(design_id):
    spec = SPEC.model_copy(update={"design": design_id, "colors": palette.Colors(), "subline": "15 SEPT, 12:00 AM"})
    for faces, product in (([(0.38, 0.27, 0.24, 0.12)], (0.4, 0.62, 0.2, 0.1)), ([], (0.4, 0.5, 0.2, 0.2)), ([], None)):
        for cut in (True, False):
            img, rep = design.draw(spec, _frame(), faces, _mask(), product, cut=cut)
            assert img.size == (1080, 1920)
            bad = [c["detail"] for c in rep["checks"] if not c["ok"] and c["id"] in ("safe_zone", "face_clear", "contrast")]
            assert not bad, (design_id, cut, faces, bad)
            if not cut:
                assert not rep["cut"]


def test_eighteen_designs_each_with_its_reference():
    assert len(designs.DESIGNS) == 18
    assert all(r.source and r.blurb and r.subject in ("cutout", "pedestal", "silhouette", "card", "frame", "collage") for r in designs.DESIGNS.values())
    assert set(designs.FROM_TEMPLATE.values()) <= set(designs.DESIGNS)


def test_colours_by_role_user_first_then_design_then_house():
    wash = designs.wash_of(designs.DESIGNS["colour_flood"])
    pal = palette.resolve(palette.Colors(), designs.DESIGNS["colour_flood"].colors, wash, True)
    assert (pal.bg, pal.big, pal.small) == (palette.YELLOW, palette.INK, palette.INK)
    assert palette.contrast(pal.highlight, pal.wash) >= 3  # the default highlight can't vanish on the yellow
    mine = palette.resolve(palette.Colors(highlight="#FFC500", small="#FFC400"), designs.DESIGNS["colour_flood"].colors, wash, True)
    assert (mine.highlight, mine.small) == ("#FFC500", "#FFC400")  # what the user picks is kept...
    assert palette.readability(mine)  # ...and they are told it's hard to read
    dark = palette.resolve(palette.Colors(bg="#040F2C"), designs.DESIGNS["event_flyer"].colors, designs.wash_of(designs.DESIGNS["event_flyer"]), True)
    assert palette.contrast(dark.small, dark.wash) >= 3  # the design's dark small text turns light on a dark background


def test_designs_presets_and_previews_from_the_api(fake):
    c = TestClient(app)
    ds = c.get("/api/designs").json()
    assert len(ds) == 18 and {"id", "name", "from", "blurb"} <= set(ds[0])
    pal = c.get("/api/palettes").json()
    assert set(pal["roles"]) == set(palette.ROLES) and any(p["id"] == "wolverine" for p in pal["presets"])
    cov = _make(c)
    pv = c.get(f"/api/covers/{cov['id']}/previews").json()
    assert set(pv) == set(designs.DESIGNS)
    tile = Image.open(io.BytesIO(c.get(pv["sunset_poster"]).content))
    assert tile.size == (270, 480)
    assert c.get(f"/api/covers/{cov['id']}/previews").json() == pv  # cached while nothing changes

    r = c.patch(f"/api/covers/{cov['id']}", json={"design": "sunset_poster", "colors": "wolverine"})
    assert r.status_code == 200, r.text
    e = r.json()
    assert e["spec"]["design"] == "sunset_poster" and e["spec"]["colors"]["bg"] == "#FFC400"
    e = c.patch(f"/api/covers/{cov['id']}", json={"colors": {"big": "#FF0000", "bg": None}}).json()
    assert (e["spec"]["colors"]["big"], e["spec"]["colors"]["bg"], e["spec"]["colors"]["small"]) == ("#FF0000", None, palette.INK)
    assert c.get(f"/api/covers/{cov['id']}/previews").json() != pv  # new colours, new previews
    e = c.patch(f"/api/covers/{cov['id']}", json={"colors": "game"}).json()
    assert e["spec"]["colors"]["bg"] == "#0A0A2E"
    assert c.patch(f"/api/covers/{cov['id']}", json={"design": "nope"}).status_code == 422
    assert c.patch(f"/api/covers/{cov['id']}", json={"colors": "nope"}).status_code == 422
    assert c.get(f"/api/covers/{cov['id']}/previews/..%2Fcover.json").status_code == 404


def test_the_chat_picks_a_headline_by_number_and_can_write_its_own(fake):
    c = TestClient(app)
    cov = _make(c)
    e = c.patch(f"/api/covers/{cov['id']}", json={"headline": 2}).json()
    assert (e["spec"]["line1"], e["spec"]["line2"], e["spec"]["pill_left"], e["spec"]["arrow"], e["headline"]) == ("Is your PS5", "Out of space?", None, False, 2)
    assert e["why"] == "a question" and e["images"]["cut"] != cov["images"]["cut"]
    e = c.patch(f"/api/covers/{cov['id']}", json={"line1": "Two terabytes", "line2": "Of games"}).json()
    assert (e["spec"]["line2"], e["headline"]) == ("Of games", -1)  # the user's own words
    assert c.patch(f"/api/covers/{cov['id']}", json={"headline": 7}).status_code == 422
    assert len(fake["calls"]) == 2  # picking and writing call no model


def test_recommendations_are_filled_from_the_frame_when_the_model_gives_none():
    from studio import covers
    assert covers.recommend([], [(0.4, 0.2, 0.2, 0.1)], None, {})[:3] == [r.id for r in designs.DESIGNS.values() if "person" in r.fits][:3]
    assert all("art" in designs.DESIGNS[d].fits for d in covers.recommend(["nope"], [], None, {}))
    assert covers.recommend(["sunset_poster", "sunset_poster"], [], None, {})[0] == "sunset_poster"


def test_the_pill_ends_on_one_word():
    from studio import covers
    w = covers.Words(**{**WORDS, "pill_left": "In", "pill_right": "Stock Now"})
    assert (covers._pill(w).pill_left, covers._pill(w).pill_right) == ("In Stock", "Now")
    assert covers._pill(covers.Words(**WORDS)).pill_right == "Now"
    w = covers.Words(**{**WORDS, "pill_left": "20% Off", "pill_right": "This Week"})
    assert (covers._pill(w).pill_left, covers._pill(w).pill_right) == ("20% Off", "This Week")


def test_a_preset_stays_readable_on_any_design_but_custom_colours_are_kept(fake):
    c = TestClient(app)
    cov = _make(c)
    e = c.patch(f"/api/covers/{cov['id']}", json={"design": "staff_hook", "colors": "days_of_play"}).json()
    assert e["spec"]["colors_from"] == "days_of_play" and not e["readability"]
    assert all(k["ok"] for k in e["checks"]["cut"] if k["id"] == "contrast"), e["checks"]["cut"]
    e = c.patch(f"/api/covers/{cov['id']}", json={"colors": {"small": "#BFD4FF"}}).json()
    assert e["spec"]["colors_from"] == "custom" and e["readability"]  # picked by hand: kept, and flagged


# --- phase 3: edits in the chat, undo, my palettes, the library -----------------------------------

def _say(c, cid, message):
    r = c.post(f"/api/covers/{cid}/chat", json={"message": message})
    assert r.status_code == 200, r.text
    return r.json()


def test_simple_edits_are_done_by_code_for_free(fake):
    c = TestClient(app)
    cov = _make(c)
    cid, calls = cov["id"], len(fake["calls"])
    e = _say(c, cid, "Make the headline red")
    assert e["spec"]["colors"]["big"] == "#E53935" and e["spec"]["colors_from"] == "custom"
    e = _say(c, cid, "background #102030")
    assert e["spec"]["colors"]["bg"] == "#102030"
    e = _say(c, cid, "remove the arrow")
    assert e["spec"]["arrow"] is False
    e = _say(c, cid, "zoom out")
    assert e["spec"]["punch_in"] is False
    e = _say(c, cid, "remove the pill")
    assert e["spec"]["pill_left"] is None and e["spec"]["pill_right"] is None
    e = _say(c, cid, "use the sunset poster")
    assert e["spec"]["design"] == "sunset_poster"
    before = e["spec"]["design"]
    e = _say(c, cid, "another design")
    assert e["spec"]["design"] != before
    e = _say(c, cid, "other colours")
    assert e["spec"]["colors_from"] != "custom"
    e = _say(c, cid, "undo")
    assert e["spec"]["colors_from"] == "custom" and e["can_undo"]
    assert len(fake["calls"]) == calls  # no model was asked
    bots = [m for m in e["messages"] if m["role"] == "bot"]
    assert len(bots) == 9 and all(m["by"] == "code" and m["usd"] == 0 for m in bots)
    assert c.get(f"/api/covers/{cid}").json()["messages"] == e["messages"]  # kept on the cover


def test_anything_else_goes_to_glm_and_bad_changes_are_dropped(fake):
    c = TestClient(app)
    cov = _make(c)
    e = _say(c, cov["id"], "make it punchier and less wordy")
    assert fake["calls"][-1] == ("glm-5.3-flash", False)
    sp = e["spec"]
    assert (sp["line1"], sp["line2"], sp["pill_left"]) == ("No more", "Deleting", None)
    assert sp["design"] == cov["spec"]["design"]  # a made-up design is ignored
    assert sp["colors"]["big"] == "#FF0000" and "nope" not in sp["colors"]
    assert e["messages"][-1] == {**e["messages"][-1], "role": "bot", "text": "Shorter now.", "by": "glm-5.3-flash"}
    assert any(r["step"] == "edit" for r in e["costs"])


def test_with_no_budget_left_the_chat_says_so_and_the_buttons_still_work(fake, monkeypatch):
    c = TestClient(app)
    cov = _make(c)
    monkeypatch.setattr(settings, "budget_usd", 0.0000001)
    from studio import covers
    covers.update(cov["id"], budget_usd=0.0000001)
    calls = len(fake["calls"])
    e = _say(c, cov["id"], "rewrite it as a question")
    assert len(fake["calls"]) == calls and "budget" in e["messages"][-1]["text"]
    e = _say(c, cov["id"], "make the headline gold")  # free: still works
    assert e["spec"]["colors"]["big"] == "#D4AF37"


def test_undo_goes_back_one_version_at_a_time(fake):
    c = TestClient(app)
    cov = _make(c)
    cid = cov["id"]
    assert c.post(f"/api/covers/{cid}/undo").status_code == 409  # nothing yet
    c.patch(f"/api/covers/{cid}", json={"design": "sunset_poster"})
    c.patch(f"/api/covers/{cid}", json={"headline": 1})
    c.patch(f"/api/covers/{cid}", json={"headline": 1})  # no change: no new version
    e = c.post(f"/api/covers/{cid}/undo").json()
    assert (e["spec"]["design"], e["spec"]["line2"], e["headline"]) == ("sunset_poster", "Again", 0)
    e = c.post(f"/api/covers/{cid}/undo").json()
    assert e["spec"]["design"] == cov["spec"]["design"] and not e["can_undo"]


def test_my_palettes_are_saved_used_and_kept_exactly(fake):
    c = TestClient(app)
    r = c.post("/api/palettes", json={"name": "  Store  Blue ", "colors": {"bg": "#002B7F", "big": "#FFFFFF", "small": "#BFD4FF"}})
    assert r.status_code == 201, r.text
    mine = r.json()
    assert mine["label"] == "Store Blue" and mine["id"].startswith("mine:")
    assert c.post("/api/palettes", json={"name": "", "colors": {"bg": "#002B7F"}}).status_code == 422
    assert c.post("/api/palettes", json={"name": "Empty", "colors": {}}).status_code == 422
    assert [x["id"] for x in c.get("/api/palettes").json()["mine"]] == [mine["id"]]
    cov = _make(c)
    e = c.patch(f"/api/covers/{cov['id']}", json={"design": "staff_hook", "colors": mine["id"]}).json()
    assert e["spec"]["colors"]["small"] == "#BFD4FF" and e["spec"]["colors_from"] == mine["id"]
    assert e["readability"]  # light text on the white studio: the user's own colours are kept, and flagged
    assert c.delete("/api/palettes/" + mine["id"]).status_code == 200
    assert c.delete("/api/palettes/" + mine["id"]).status_code == 404
    assert c.patch(f"/api/covers/{cov['id']}", json={"colors": mine["id"]}).status_code == 422


def test_the_library_lists_covers_newest_first(fake):
    c = TestClient(app)
    first, second = _make(c), _make(c)
    ids = [x["id"] for x in c.get("/api/covers").json()]
    assert ids.index(second["id"]) < ids.index(first["id"])
    item = next(x for x in c.get("/api/covers").json() if x["id"] == second["id"])
    assert item["status"] == "done" and item["title"] and "/cut.png" in item["thumb"]


# --- phase 4: collage, offer, QR, sizes --------------------------------------------------------------

def test_a_collage_shows_the_added_images_and_keeps_their_faces_clear(fake):
    c = TestClient(app)
    cov = _make(c)
    cid = cov["id"]
    files = [("files", (f"x{i}.png", _png(_frame(colour=(20 * i, 80, 160))), "image/png")) for i in range(2)]
    e = c.post(f"/api/covers/{cid}/images", files=files).json()
    assert e["extras"] == 2
    e = c.patch(f"/api/covers/{cid}", json={"design": "collage"}).json()
    assert all(k["ok"] for k in e["checks"]["cut"]), e["checks"]["cut"]
    too_many = [("files", (f"y{i}.png", _png(_frame()), "image/png")) for i in range(2)]
    assert c.post(f"/api/covers/{cid}/images", files=too_many).status_code == 422  # 2 + 2 is over 3
    assert c.post(f"/api/covers/{cid}/images", files=[("files", ("z.txt", b"nope", "text/plain"))]).status_code == 422
    assert c.delete(f"/api/covers/{cid}/images").json()["extras"] == 0


def test_collage_tiles_never_crop_through_a_head():
    face = [(0.4, 0.05, 0.2, 0.1)]  # a face right at the top of an image
    img, rep = design.draw(SPEC.model_copy(update={"design": "collage"}), _frame(), face, _mask(),
                           extras=[(_frame(), face), (_frame(), []), (_frame(), face)])
    assert rep["mode"] == "collage" and len(rep["heads"]) == 3
    words_bottom = max(b[3] for k, b in rep["boxes"].items() if k.startswith(("line", "pill")))
    assert all(t > words_bottom for _, t, _, _ in rep["heads"])
    assert next(k for k in rep["checks"] if k["id"] == "face_clear")["ok"]


def test_an_offer_in_the_video_puts_the_offer_design_first_and_shows_its_tag(fake, monkeypatch):
    monkeypatch.setitem(HEADLINES["options"][0], "offer", "20% OFF")
    c = TestClient(app)
    cov = _make(c)
    assert cov["recommended"][0] == "offer" and cov["spec"]["design"] == "offer" and cov["spec"]["offer"] == "20% OFF"
    img, rep = design.draw(design.Spec.model_validate(cov["spec"]), _frame(), [(0.38, 0.27, 0.24, 0.12)], _mask(), (0.4, 0.62, 0.2, 0.1))
    assert "offer" in rep["boxes"] and all(k["ok"] for k in rep["checks"]), rep["checks"]


def test_offer_and_qr_by_chat_for_free(fake):
    c = TestClient(app)
    cov = _make(c)
    cid, calls = cov["id"], len(fake["calls"])
    e = _say(c, cid, "price ₹4,999")
    assert (e["spec"]["offer"], e["spec"]["design"]) == ("₹4,999", "offer")
    e = _say(c, cid, "remove the offer")
    assert e["spec"]["offer"] is None
    e = _say(c, cid, "use the offer design")
    assert e["spec"]["design"] == "offer" and e["spec"]["offer"] is None
    e = _say(c, cid, "add a qr https://example.com/register.")
    assert e["spec"]["qr_url"] == "https://example.com/register"
    assert all(k["ok"] for k in e["checks"]["cut"]), e["checks"]["cut"]
    e = _say(c, cid, "qr code please")
    assert "link" in e["messages"][-1]["text"] and e["spec"]["qr_url"] == "https://example.com/register"
    e = _say(c, cid, "remove the qr")
    assert e["spec"]["qr_url"] is None
    assert len(fake["calls"]) == calls
    assert c.patch(f"/api/covers/{cid}", json={"qr_url": "javascript:alert(1)"}).status_code == 422


def test_other_sizes_keep_every_word(fake):
    c = TestClient(app)
    cov = _make(c)
    for size, dims in {"reel": (1080, 1920), "feed": (1080, 1350), "square": (1080, 1080), "youtube": (1280, 720)}.items():
        r = c.get(f"/api/covers/{cov['id']}/export/{size}.png?version=plain&download=1")
        assert r.status_code == 200 and Image.open(io.BytesIO(r.content)).size == dims
        assert f"-plain-{size}.png" in r.headers["content-disposition"]
    assert c.get(f"/api/covers/{cov['id']}/export/poster.png").status_code == 404


def test_text_with_a_rupee_sign_falls_back_to_a_face_that_has_it():
    if not design._face_file("heavy_fallback"):
        pytest.skip("no fallback face on this machine")
    assert design.font_for("heavy", 60, "₹4,999").path != design.font("heavy", 60).path or not design._has("heavy", "₹")
    assert design.font_for("heavy", 60, "NOW").path == design.font("heavy", 60).path
