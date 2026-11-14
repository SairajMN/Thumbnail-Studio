"""Post-export thumbnail agent for video editors and video-generation pipelines.

Drop-in integration layer over Thumbnail Studio (`studio/`): call it right after
the user clicks Export in a video editor to turn the exported keyframe and video
context (transcript, caption, timeline notes, game/product/offer metadata) into
platform-ready thumbnails without modifying the core `studio` package.

Supports two execution modes:
- In-process Python mode (`base_url=None`): calls `studio.covers` and `studio.chat` directly.
- HTTP service mode (`base_url="http://127.0.0.1:8300"`): talks to a running Thumbnail Studio server.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, model_validator

VALID_SIZES = ("reel", "feed", "square", "youtube")
VALID_VERSIONS = ("cut", "plain")


class VideoExportContext(BaseModel):
    """Context emitted by a video editor/generator right after video export."""

    frame_path: Path | None = None
    frame_bytes: bytes | None = None
    title: str | None = None
    caption: str | None = None
    transcript: str | None = None
    timeline_notes: str | None = None
    game: str | None = None
    offer: str | None = None
    extra_frames: list[Path | bytes] = Field(default_factory=list)
    export_sizes: list[Literal["reel", "feed", "square", "youtube"]] = Field(default_factory=lambda: ["reel"])
    version: Literal["cut", "plain"] = "cut"
    preferred_design: str | None = None
    preferred_palette: str | dict[str, str | None] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_inputs(self) -> VideoExportContext:
        if self.frame_path is None and self.frame_bytes is None:
            raise ValueError("Provide either frame_path or frame_bytes")
        if len(self.extra_frames) > 3:
            raise ValueError("extra_frames supports at most 3 images")
        if not self.export_sizes:
            raise ValueError("export_sizes must include at least one size")
        return self

    def read_frame_bytes(self) -> bytes:
        if self.frame_bytes is not None:
            return self.frame_bytes
        assert self.frame_path is not None
        if not self.frame_path.exists():
            raise FileNotFoundError(f"Frame not found: {self.frame_path}")
        return self.frame_path.read_bytes()

    def read_extra_frames(self) -> list[bytes]:
        out: list[bytes] = []
        for item in self.extra_frames:
            if isinstance(item, bytes):
                out.append(item)
            else:
                if not item.exists():
                    raise FileNotFoundError(f"Extra frame not found: {item}")
                out.append(item.read_bytes())
        return out

    def to_context_json(self) -> str:
        payload: dict[str, Any] = {}
        for key in ("title", "caption", "transcript", "timeline_notes", "game", "offer"):
            val = getattr(self, key)
            if val:
                payload[key] = val
        if self.metadata:
            payload["metadata"] = self.metadata
        return json.dumps(payload, ensure_ascii=False) if payload else ""


class ThumbnailResult(BaseModel):
    """Result returned to the video editor after post-export thumbnail generation."""

    cover_id: str
    status: str
    used_usd: float = 0.0
    recommended_designs: list[str] = Field(default_factory=list)
    chosen_spec: dict[str, Any] | None = None
    headline_options: list[dict[str, Any]] = Field(default_factory=list)
    why: str | None = None
    checks: dict[str, Any] = Field(default_factory=dict)
    cut_available: bool = False
    exports: dict[str, str] = Field(default_factory=dict)


class ThumbnailAgent:
    """Post-export thumbnail orchestrator for video editing pipelines."""

    def __init__(
        self,
        base_url: str | None = None,
        output_dir: Path | None = None,
        poll_interval: float = 1.0,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.poll_interval = poll_interval
        self.timeout = timeout

    def generate_after_export(self, ctx: VideoExportContext) -> ThumbnailResult:
        """Run the full post-export pipeline: analyze frame, write hooks, draw cover, export sizes."""
        frame_data = ctx.read_frame_bytes()
        extra_data = ctx.read_extra_frames()
        context_str = ctx.to_context_json()

        if self.base_url:
            cover = self._run_http(frame_data, context_str, extra_data, ctx)
        else:
            cover = self._run_direct(frame_data, context_str, extra_data, ctx)

        exports = self.export_sizes(cover["id"], list(ctx.export_sizes), version=ctx.version)
        return ThumbnailResult(
            cover_id=cover["id"],
            status=cover["status"],
            used_usd=float(cover.get("used_usd", 0.0)),
            recommended_designs=list(cover.get("recommended") or []),
            chosen_spec=cover.get("spec"),
            headline_options=list(cover.get("headlines") or []),
            why=cover.get("why"),
            checks=dict(cover.get("checks") or {}),
            cut_available=bool(cover.get("cut_available", False)),
            exports=exports,
        )

    def refine(
        self,
        cover_id: str,
        *,
        message: str | None = None,
        edits: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Refine an existing cover via natural-language chat or structured field edits."""
        if not message and not edits:
            raise ValueError("Provide message or edits")
        if self.base_url:
            with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                if edits:
                    r = client.patch(f"/api/covers/{cover_id}", json=edits)
                    r.raise_for_status()
                    data = r.json()
                if message:
                    r = client.post(f"/api/covers/{cover_id}/chat", json={"message": message})
                    r.raise_for_status()
                    data = r.json()
                return data

        from studio import chat, covers

        data = covers.out(covers.edit(cover_id, edits)) if edits else None
        if message:
            data = covers.out(chat.say(cover_id, message))
        assert data is not None
        return data

    def export_sizes(
        self,
        cover_id: str,
        sizes: list[str],
        version: str = "cut",
    ) -> dict[str, str]:
        """Export a generated cover into one or more target platform sizes."""
        if version not in VALID_VERSIONS:
            raise ValueError(f"version must be one of {VALID_VERSIONS}")
        for s in sizes:
            if s not in VALID_SIZES:
                raise ValueError(f"size {s!r} must be one of {VALID_SIZES}")

        if self.base_url:
            exports: dict[str, str] = {}
            with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                for size in sizes:
                    endpoint = f"/api/covers/{cover_id}/export/{size}.png"
                    if self.output_dir:
                        self.output_dir.mkdir(parents=True, exist_ok=True)
                        r = client.get(endpoint, params={"version": version, "download": "1"})
                        r.raise_for_status()
                        dest = self.output_dir / f"{cover_id}-{version}-{size}.png"
                        dest.write_bytes(r.content)
                        exports[size] = str(dest)
                    else:
                        exports[size] = f"{self.base_url}{endpoint}?version={version}"
            return exports

        from studio import covers

        return {size: str(covers.export(cover_id, size, version)) for size in sizes}

    def _run_direct(
        self,
        frame_data: bytes,
        context_str: str,
        extra_data: list[bytes],
        ctx: VideoExportContext,
    ) -> dict[str, Any]:
        from studio import covers

        created = covers.create(frame_data, context_str)
        cid = created["id"]
        covers.run(cid)
        cover = covers.load(cid)
        if cover["status"] != "done":
            raise RuntimeError(cover.get("error") or f"Cover generation ended with status={cover['status']}")

        if extra_data:
            covers.add_images(cid, extra_data)

        edits = _build_edits(ctx)
        if edits:
            cover = covers.edit(cid, edits)

        return covers.out(cover)

    def _run_http(
        self,
        frame_data: bytes,
        context_str: str,
        extra_data: list[bytes],
        ctx: VideoExportContext,
    ) -> dict[str, Any]:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            r = client.post(
                "/api/covers",
                files={"frame": ("frame.png", frame_data, "image/png")},
                data={"context": context_str},
            )
            r.raise_for_status()
            cover = r.json()
            cid = cover["id"]

            deadline = time.monotonic() + self.timeout
            while cover["status"] not in ("done", "failed"):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for cover {cid}")
                time.sleep(self.poll_interval)
                poll = client.get(f"/api/covers/{cid}")
                poll.raise_for_status()
                cover = poll.json()

            if cover["status"] != "done":
                raise RuntimeError(cover.get("error") or f"Cover generation ended with status={cover['status']}")

            if extra_data:
                files = [("files", (f"extra-{i}.png", b, "image/png")) for i, b in enumerate(extra_data)]
                up = client.post(f"/api/covers/{cid}/images", files=files)
                up.raise_for_status()
                cover = up.json()

            edits = _build_edits(ctx)
            if edits:
                ed = client.patch(f"/api/covers/{cid}", json=edits)
                ed.raise_for_status()
                cover = ed.json()

            return cover


def _build_edits(ctx: VideoExportContext) -> dict[str, Any]:
    edits: dict[str, Any] = {}
    if ctx.preferred_design:
        edits["design"] = ctx.preferred_design
    if ctx.preferred_palette is not None:
        edits["colors"] = ctx.preferred_palette
    return edits


def generate_after_export(
    ctx: VideoExportContext,
    *,
    base_url: str | None = None,
    output_dir: Path | None = None,
) -> ThumbnailResult:
    """Convenience entrypoint for video-editor post-export hooks."""
    return ThumbnailAgent(base_url=base_url, output_dir=output_dir).generate_after_export(ctx)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-export Thumbnail Studio agent")
    p.add_argument("--frame", type=Path, required=True, help="Path to exported video frame (PNG/JPEG/WebP)")
    p.add_argument("--title", default=None, help="Video title")
    p.add_argument("--caption", default=None, help="Post caption or description")
    p.add_argument("--transcript", default=None, help="Video transcript text")
    p.add_argument("--notes", default=None, help="Editor timeline notes")
    p.add_argument("--game", default=None, help="Game or subject name")
    p.add_argument("--offer", default=None, help="Promo/price tag (e.g. '20%% OFF')")
    p.add_argument("--design", default=None, help="Preferred design id (e.g. 'key_art', 'staff_hook', 'offer')")
    p.add_argument("--palette", default=None, help="Preferred palette preset name (e.g. 'game', 'house', 'fc')")
    p.add_argument("--sizes", nargs="+", default=["reel"], choices=VALID_SIZES, help="Export sizes")
    p.add_argument("--version", default="cut", choices=VALID_VERSIONS, help="Cover version: cut or plain")
    p.add_argument("--base-url", default=None, help="Optional Thumbnail Studio server URL (uses direct mode if omitted)")
    p.add_argument("--output-dir", type=Path, default=None, help="Directory to save downloaded PNGs in HTTP mode")
    p.add_argument("--json", action="store_true", help="Print full JSON result")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ctx = VideoExportContext(
        frame_path=args.frame,
        title=args.title,
        caption=args.caption,
        transcript=args.transcript,
        timeline_notes=args.notes,
        game=args.game,
        offer=args.offer,
        preferred_design=args.design,
        preferred_palette=args.palette,
        export_sizes=args.sizes,
        version=args.version,
    )
    result = generate_after_export(ctx, base_url=args.base_url, output_dir=args.output_dir)
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(f"Cover {result.cover_id} ({result.status}) — cost ${result.used_usd:.4f}")
        for size, path in result.exports.items():
            print(f"  {size}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
