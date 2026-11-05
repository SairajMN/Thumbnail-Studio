"""What code finds in the frame on this machine, for free: faces, the subject's outline, the main colours, on-screen text.

Models (in studio/models): OpenCV Zoo's YuNet face detector (MIT) and U²-Net-p salient-object segmentation (Apache-2.0,
the file rembg distributes), both run with OpenCV's own DNN module.
"""

import threading
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

MODELS = Path(__file__).with_name("models")
Box = tuple[float, float, float, float]  # x, y, w, h as fractions of the frame
SEGMENT_LOCK = threading.Lock()  # one OpenCV net is shared, and it is not safe to run from two threads at once


def faces(img: Image.Image) -> list[Box]:
    """The faces the cover is about. Faces under 4% of the frame's width, or under half the largest face, are crowd."""
    pic = img.convert("RGB")
    k = min(1.0, 1280 / max(pic.size))  # detection needs no more than this; big frames would only be slow
    small = pic.resize((round(pic.width * k), round(pic.height * k))) if k < 1 else pic
    det = cv2.FaceDetectorYN.create(str(MODELS / "face_detection_yunet_2023mar.onnx"), "", small.size, 0.7)
    _, found = det.detect(np.ascontiguousarray(np.asarray(small)[:, :, ::-1]))
    w, h = small.size
    boxes = [tuple(float(v) for v in r[:4]) for r in (found if found is not None else [])]
    big = max((fw for _, _, fw, _ in boxes), default=0)
    return [(x / w, y / h, fw / w, fh / h) for x, y, fw, fh in boxes if fw >= 0.04 * w and fw >= 0.5 * big]


@cache
def _segmenter() -> cv2.dnn.Net:
    return cv2.dnn.readNetFromONNX(str(MODELS / "u2netp.onnx"))


def subject_mask(img: Image.Image) -> np.ndarray:
    """The frame's main subject (people with what they hold, or a product) as a 0-1 mask at the frame's size."""
    pic = img.convert("RGB")
    a = (np.asarray(pic.resize((320, 320)), np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    with SEGMENT_LOCK:
        net = _segmenter()
        net.setInput(a.transpose(2, 0, 1)[None].astype(np.float32))
        m = net.forward()[0, 0].copy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    return cv2.resize(m.astype(np.float32), pic.size, interpolation=cv2.INTER_LINEAR)


def palette(img: Image.Image, k: int = 5) -> list[tuple[str, float]]:
    """The main colours, most of the frame first, as (#RRGGBB, share of the frame)."""
    px = np.asarray(img.convert("RGB").resize((96, 96)), np.float32).reshape(-1, 3)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centres = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(labels.ravel(), minlength=k) / len(labels)
    return [("#%02X%02X%02X" % tuple(int(v) for v in centres[i]), round(float(share[i]), 2)) for i in np.argsort(-share)]


def text(img: Image.Image, min_conf: float = 60) -> list[str]:
    """On-screen text, line by line (Tesseract in sparse-text mode: frames hold a little text scattered over a picture)."""
    d = pytesseract.image_to_data(img.convert("RGB"), config="--psm 11", output_type=pytesseract.Output.DICT)
    rows: dict[int, list[tuple[int, str]]] = {}
    for i, w in enumerate(d["text"]):
        if w.strip() and float(d["conf"][i]) >= min_conf:
            rows.setdefault(round((d["top"][i] + d["height"][i] / 2) / max(8, d["height"][i])), []).append((d["left"][i], w))
    return [" ".join(w for _, w in sorted(r)) for _, r in sorted(rows.items())][:20]


def read(img: Image.Image) -> dict:
    """The facts a model gets about the frame, besides the frame itself."""
    return {"text_in_frame": text(img), "main_colours": [{"hex": h, "share": s} for h, s in palette(img)],
            "faces": len(faces(img)), "shape": "portrait" if img.height > img.width else "landscape"}
