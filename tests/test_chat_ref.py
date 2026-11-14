import io

from fastapi.testclient import TestClient
from PIL import Image

from studio import covers, nebius
from studio.app import app
from studio.design import Spec


def _png_bytes(color: tuple[int, int, int] = (20, 40, 90), size: tuple[int, int] = (320, 480)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _make_done_cover() -> str:
    c = covers.create(_png_bytes(), "New PS5 game launch")
    cid = c["id"]
    spec = Spec(
        template="key_art",
        design="key_art",
        line1="OUT NOW",
        line2="PLAY TODAY",
        highlight="NOW",
    )
    covers.update(cid, status="done", stage=None, spec=spec.model_dump())
    covers.draw(cid)
    return cid


def test_chat_with_reference_image_multipart(monkeypatch):
    cid = _make_done_cover()
    seen = {}

    def fake_ask(model_id: str, system: str, text: str, has_image: bool) -> dict:
        seen["model_id"] = model_id
        seen["has_image"] = has_image
        seen["text"] = text
        return {
            "reply": "Matched the bold neon-on-dark style from your reference image.",
            "line1": None,
            "line2": None,
            "highlight": None,
            "pill_left": "Watch",
            "pill_right": "Now",
            "design": "spotlight_unbox",
            "colors": {"bg": "#111827", "big": "#CDFD4A", "highlight": "#FFFFFF"},
            "arrow": None,
            "offer": None,
            "punch_in": True,
        }

    monkeypatch.setattr(nebius, "FAKE", fake_ask)

    client = TestClient(app)
    ref_bytes = _png_bytes((200, 50, 50), (400, 400))
    resp = client.post(
        f"/api/covers/{cid}/chat",
        data={"message": "make it like that"},
        files={"image": ("ref.png", ref_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert seen["model_id"] == "kimi-k3"
    assert seen["has_image"] is True
    assert "make it like that" in seen["text"]

    assert body["spec"]["design"] == "spotlight_unbox"
    assert body["spec"]["colors"]["bg"] == "#111827"
    assert body["spec"]["colors"]["big"] == "#CDFD4A"
    assert body["spec"]["pill_left"] == "Watch"

    user_msg = body["messages"][-2]
    assert user_msg["role"] == "user"
    assert user_msg["text"] == "make it like that"
    assert user_msg["image"] == f"/api/covers/{cid}/ref/ref-1.png"

    ref_resp = client.get(user_msg["image"])
    assert ref_resp.status_code == 200
    assert ref_resp.headers["content-type"] == "image/png"


def test_chat_json_still_works():
    cid = _make_done_cover()
    client = TestClient(app)
    resp = client.post(f"/api/covers/{cid}/chat", json={"message": "make the headline red"})
    assert resp.status_code == 200
    assert resp.json()["spec"]["colors"]["big"] == "#E53935"
