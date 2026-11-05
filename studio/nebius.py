"""Nebius Token Factory, the only provider (OpenAI-compatible chat). A reply is JSON checked against a Pydantic model.

Prices are USD per 1M tokens, as Nebius lists them (Oct 2026). `extra` turns off hidden thinking, which otherwise
spends thousands of tokens per call (measured in Guideline Copilot). `vision`: the model really sees an image sent to
it. GLM 5.3 Flash is listed by Nebius as a vision model but drops images (Nov 5 2026: a 33-token prompt and "I don't
see any image"), so it is sent text only; Kimi K3 named the game in a real frame the same day.
"""

import base64
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from pydantic import BaseModel, ValidationError

from studio.config import settings

URL = "https://api.tokenfactory.nebius.com/v1/chat/completions"
IMAGE_TOKENS = 1500  # what one frame costs in input tokens, roughly (the price check uses it before a call)
NO_THINKING = {"chat_template_kwargs": {"thinking": False}}
MODELS = {
    "kimi-k3": {"label": "Kimi K3", "api_id": "moonshotai/Kimi-K3", "in": 3.0, "out": 15.0, "vision": True, "extra": NO_THINKING},
    "kimi-k2.6": {"label": "Kimi K2.6", "api_id": "moonshotai/Kimi-K2.6", "in": 0.95, "out": 4.0, "vision": True, "extra": NO_THINKING},
    "qwen3.8-27b": {"label": "Qwen 3.8 27B", "api_id": "Qwen/Qwen3.8-27B", "in": 0.45, "out": 3.0, "vision": True,
                    "extra": {"chat_template_kwargs": {"enable_thinking": False}}},
    "glm-5.3": {"label": "GLM 5.3", "api_id": "zai-org/GLM-5.3", "in": 1.4, "out": 4.4, "vision": False, "extra": {"reasoning_effort": "low"}},
    "glm-5.2": {"label": "GLM 5.2", "api_id": "zai-org/GLM-5.2", "in": 1.4, "out": 4.4, "vision": False,
                "extra": {"chat_template_kwargs": {"enable_thinking": False}}},
    "glm-5.1": {"label": "GLM 5.1", "api_id": "zai-org/GLM-5.1", "in": 1.4, "out": 4.4, "vision": False,
                "extra": {"chat_template_kwargs": {"enable_thinking": False}}},
    "glm-5.3-flash": {"label": "GLM 5.3 Flash", "api_id": "zai-org/GLM-5.3-Flash", "in": 0.15, "out": 0.5, "vision": False,
                      "extra": {"reasoning_effort": "low"}},
}


@dataclass
class Call:
    """One request, kept on the cover so its cost is shown model by model."""
    step: str
    model: str
    ok: bool
    usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    ms: int = 0
    error: str | None = None

    def dict(self) -> dict:
        return asdict(self)


class CallFailed(Exception):
    def __init__(self, call: Call):
        super().__init__(call.error)
        self.call = call


# Tests install a fake: (model_id, system, text, has_image) -> reply dict
FAKE: Callable[[str, str, str, bool], dict] | None = None


def usd(model_id: str, tokens_in: int, tokens_out: int) -> float:
    m = MODELS[model_id]
    return (tokens_in * m["in"] + tokens_out * m["out"]) / 1e6


def estimate_in(system: str, text: str, image: bool) -> int:
    """Input tokens a call will send, roughly: about 3.5 characters a token, plus a frame."""
    return round((len(system) + len(text)) / 3.5) + (IMAGE_TOKENS if image else 0)


def _strip_fence(t: str) -> str:
    t = t.strip()
    return t.split("\n", 1)[1].rsplit("```", 1)[0] if t.startswith("```") else t


def ask(step: str, model_id: str, system: str, text: str, out: type[BaseModel], max_tokens: int,
        image: Path | None = None) -> tuple[BaseModel, Call]:
    """One call. The image goes only to a model that sees it. Raises CallFailed with the call (and its cost)."""
    m = MODELS[model_id]
    send_image = image is not None and m["vision"]
    schema = json.dumps(out.model_json_schema())
    system = f"{system}\n\nReply with one JSON object that matches this JSON Schema (every property present):\n{schema}"
    t0 = time.monotonic()
    call = Call(step=step, model=model_id, ok=False)
    try:
        if FAKE is not None:
            raw, tin, tout = json.dumps(FAKE(model_id, system, text, send_image)), 0, 0
        else:
            if not settings.nebius_api_key:
                raise RuntimeError("NEBIUS_API_KEY is not set in .env")
            content = [{"type": "text", "text": text}]
            if send_image:
                data = base64.b64encode(image.read_bytes()).decode()
                content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
            r = httpx.post(URL, headers={"Authorization": f"Bearer {settings.nebius_api_key}"}, timeout=180,
                           json={"model": m["api_id"], "max_tokens": max_tokens, "response_format": {"type": "json_object"},
                                 **m["extra"], "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]})
            r.raise_for_status()
            body = r.json()
            choice, u = body["choices"][0], body.get("usage") or {}
            tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            call.tokens_in, call.tokens_out, call.usd = tin, tout, round(usd(model_id, tin, tout), 6)
            if choice.get("finish_reason") == "length":
                raise RuntimeError("reply cut off at max_tokens")
            raw = choice["message"]["content"] or ""
        parsed = out.model_validate_json(_strip_fence(raw))
    except (ValidationError, ValueError) as e:
        call.error = f"reply did not match the schema: {str(e)[:200]}"
    except Exception as e:  # network, HTTP status, a cut-off reply
        call.error = f"{type(e).__name__}: {str(e)[:200]}"
    call.ms = int((time.monotonic() - t0) * 1000)
    if call.error:
        raise CallFailed(call)
    call.ok = True
    return parsed, call
