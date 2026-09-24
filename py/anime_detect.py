"""
anime_detect_standalone.py

Standalone, dependency-light drop-in replacement for the 4 detection functions of
``dghs-imgutils==0.19.0``:

    imgutils.detect.detect_person / detect_halfbody / detect_heads / detect_faces

Faithfully re-implements ``imgutils/generic/yolo.py`` (YOLOv8 ONNX path) and is
validated bit-exact against imgutils' own unit tests.

Dependencies: numpy, Pillow, onnxruntime   (numpy>=2 compatible)
Models: HuggingFace repos ``deepghs/anime_*`` (ONNX, dynamic input).

Model resolution order for a given ``model_name``:
  1. ``ANIME_DETECT_MODELS_HOME/<model_name>/model.onnx``
  2. download from ``ANIME_DETECT_ENDPOINT`` (default https://hf-mirror.com) into (1)
"""

from __future__ import annotations

import ast
import json
import math
import os
import threading
import urllib.request
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

import onnxruntime as ort

BBox = Tuple[int, int, int, int]
Detection = Tuple[BBox, str, float]
ImageTyping = Union[str, "os.PathLike[str]", bytes, bytearray, Image.Image]

_REPOS = {
    "person": "deepghs/anime_person_detection",
    "halfbody": "deepghs/anime_halfbody_detection",
    "head": "deepghs/anime_head_detection",
    "face": "deepghs/anime_face_detection",
}

_ENDPOINT = os.environ.get("ANIME_DETECT_ENDPOINT", "https://hf-mirror.com").rstrip("/")
_HOME = os.environ.get(
    "ANIME_DETECT_MODELS_HOME", os.path.expanduser("~/.cache/anime_detect")
)

_CACHE = {}
_CACHE_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# image loading / encoding (mirrors imgutils.data)
# --------------------------------------------------------------------------- #
def _load_rgb(image: ImageTyping) -> Image.Image:
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, (bytes, bytearray)):
        from io import BytesIO

        img = Image.open(BytesIO(image))
    else:
        img = Image.open(image)
    if img.mode not in ("RGB", "L"):
        if "A" in img.getbands() or "transparency" in img.info:
            bg = Image.new("RGBA", img.size, "white")
            bg.paste(img.convert("RGBA"), (0, 0), img.convert("RGBA"))
            img = bg
    return img.convert("RGB")


def _rgb_encode_chw(image: Image.Image) -> np.ndarray:
    array = np.asarray(image)                      # HWC uint8
    array = np.transpose(array, (2, 0, 1))         # CHW
    return (array / 255.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# model opening + metadata
# --------------------------------------------------------------------------- #
def _download_onnx(repo_id: str, model_name: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = f"{_ENDPOINT}/{repo_id}/resolve/main/{model_name}/model.onnx"
    tmp = dst + ".part"
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dst)


def _parse_names(names_str: str) -> List[str]:
    names_map = ast.literal_eval(names_str)        # e.g. "{0: 'face'}"
    return [str(names_map[i]) for i in range(len(names_map))]


def _open(model_name: str) -> Tuple[ort.InferenceSession, int, List[str], str]:
    with _CACHE_LOCK:
        if model_name in _CACHE:
            return _CACHE[model_name]

        onnx_path = os.path.join(_HOME, model_name, "model.onnx")
        if not os.path.exists(onnx_path):
            repo_id = next((r for r in _REPOS.values() if model_name.split("_detect")[0] in r), None)
            if repo_id is None:
                raise FileNotFoundError(
                    f"model {model_name!r} not found in {_HOME}; "
                    f"place it at {onnx_path} or set ANIME_DETECT_MODELS_HOME"
                )
            _download_onnx(repo_id, model_name, onnx_path)

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        meta = sess.get_modelmeta().custom_metadata_map
        if "imgsz" in meta:
            size = tuple(json.loads(meta["imgsz"]))
            assert len(size) == 2, f"imgsz should have 2 dims, got {size!r}"
        else:
            size = 640
        labels = _parse_names(meta["names"])

        # optional sibling metadata files
        model_dir = os.path.dirname(onnx_path)
        model_type = "yolo"
        try:
            with open(os.path.join(model_dir, "model_type.json"), "r") as f:
                model_type = json.load(f)["model_type"]
        except FileNotFoundError:
            pass

        _CACHE[model_name] = (sess, size, labels, model_type)
        return _CACHE[model_name]


# --------------------------------------------------------------------------- #
# preprocessing / postprocessing (verbatim imgutils logic)
# --------------------------------------------------------------------------- #
def _image_preprocess(image: Image.Image, size, allow_dynamic: bool = False, align: int = 32):
    if isinstance(size, int):
        max_w, max_h = size, size
    else:
        max_w, max_h = size[0], size[1]
    old_w, old_h = image.width, image.height
    if allow_dynamic:
        r = min(max_w / old_w, max_h / old_h)
        nw, nh = (old_w * r, old_h * r) if r < 1 else (old_w, old_h)
        nw = int(math.ceil(nw / align) * align)
        nh = int(math.ceil(nh / align) * align)
    else:
        nw, nh = int(max_w), int(max_h)
    return image.resize((nw, nh)), (old_w, old_h), (nw, nh)


def _xy_postprocess(x, y, old_size, new_size):
    x = int(np.clip(x / new_size[0] * old_size[0], 0, old_size[0]).round())
    y = int(np.clip(y / new_size[1] * old_size[1], 0, old_size[1]).round())
    return x, y


def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.7) -> List[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def _nms_postprocess(output, conf_threshold, iou_threshold, old_size, new_size, labels):
    assert output.shape[0] == 4 + len(labels), f"unexpected output shape {output.shape}"
    max_scores = output[4:, :].max(axis=0)
    output = output[:, max_scores > conf_threshold].transpose(1, 0)
    if not output.size:
        return []
    boxes = _xywh2xyxy(output[:, :4])
    scores = output[:, 4:]
    filtered = scores.max(axis=1)
    idx = _nms(boxes, filtered, iou_threshold)
    boxes, scores = boxes[idx], scores[idx]
    detections = []
    for box, score in zip(boxes, scores):
        x0, y0 = _xy_postprocess(box[0], box[1], old_size, new_size)
        x1, y1 = _xy_postprocess(box[2], box[3], old_size, new_size)
        mid = int(score.argmax())
        detections.append(((x0, y0, x1, y1), labels[mid], float(score[mid])))
    return detections


def _end2end_postprocess(output, conf_threshold, old_size, new_size, labels):
    output = output[output[:, 4] > conf_threshold]
    idx = _nms(output[:, :4], output[:, 4], 0.7)   # imgutils ignores iou_threshold here
    detections = []
    for x0, y0, x1, y1, score, cls in output[idx]:
        x0, y0 = _xy_postprocess(x0, y0, old_size, new_size)
        x1, y1 = _xy_postprocess(x1, y1, old_size, new_size)
        detections.append(((x0, y0, x1, y1), labels[int(cls.item())], float(score)))
    return detections


def _postprocess(output, model_type, conf_threshold, iou_threshold, old_size, new_size, labels):
    if model_type == "rtdetr":
        return _nms_postprocess(output.transpose(1, 0), conf_threshold, iou_threshold,
                                old_size, (1.0, 1.0), labels)
    if output.shape[-1] == 6:                       # end2end (yolov10/v11)
        return _end2end_postprocess(output, conf_threshold, old_size, new_size, labels)
    return _nms_postprocess(output, conf_threshold, iou_threshold, old_size, new_size, labels)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def _predict(image, model_name, conf_threshold, iou_threshold, allow_dynamic=False):
    sess, size, labels, model_type = _open(model_name)
    img = _load_rgb(image)
    resized, old_size, new_size = _image_preprocess(img, size, allow_dynamic=allow_dynamic)
    data = _rgb_encode_chw(resized)[None, ...]
    output, = sess.run(["output0"], {"images": data})
    return _postprocess(output[0], model_type, conf_threshold, iou_threshold,
                        old_size, new_size, labels)


def detect_person(image: ImageTyping, level: str = "m", version: str = "v1.1",
                  model_name: Optional[str] = None, conf_threshold: float = 0.3,
                  iou_threshold: float = 0.5, **kwargs) -> List[Detection]:
    return _predict(image, model_name or f"person_detect_{version}_{level}",
                    conf_threshold, iou_threshold, **kwargs)


def detect_halfbody(image: ImageTyping, level: str = "s", version: str = "v1.0",
                    model_name: Optional[str] = None, conf_threshold: float = 0.5,
                    iou_threshold: float = 0.7, **kwargs) -> List[Detection]:
    return _predict(image, model_name or f"halfbody_detect_{version}_{level}",
                    conf_threshold, iou_threshold, **kwargs)


def detect_heads(image: ImageTyping, level: Optional[str] = None,
                 model_name: Optional[str] = "head_detect_v2.0_s",
                 conf_threshold: float = 0.4, iou_threshold: float = 0.7,
                 **kwargs) -> List[Detection]:
    return _predict(image, model_name or f"head_detect_v0_{level or 's'}",
                    conf_threshold, iou_threshold, **kwargs)


def detect_faces(image: ImageTyping, level: str = "s", version: str = "v1.4",
                 model_name: Optional[str] = None, conf_threshold: float = 0.25,
                 iou_threshold: float = 0.7, **kwargs) -> List[Detection]:
    return _predict(image, model_name or f"face_detect_{version}_{level}",
                    conf_threshold, iou_threshold, **kwargs)
