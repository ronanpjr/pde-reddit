import os
import io
import json
import traceback
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

_yolo = None
_easyocr = None
_fer = None
_text_model = None


def get_yolo(model_name: str = "yolov5s"):
    global _yolo
    if _yolo is None:
        # lazy import to avoid heavy imports at module load in driver
        # Try to support multiple YOLO loaders / package APIs (yolov5, ultralytics, etc.)
        try:
            # yolov5 python package exposes load
            from yolov5 import load as yolo_load
            try:
                _yolo = yolo_load(model_name, pretrained=True)
            except TypeError:
                # older/newer API may not accept pretrained kwarg
                _yolo = yolo_load(model_name)
        except Exception:
            try:
                # ultralytics package uses YOLO class
                from ultralytics import YOLO

                try:
                    _yolo = YOLO(model_name)
                except TypeError:
                    # some wrappers expect weights=...
                    _yolo = YOLO(weights=model_name)
            except Exception:
                # Could not load any YOLO implementation; leave _yolo as None
                _yolo = None
                try:
                    print(f"[feature_helpers] WARNING: failed to import yolov5 or ultralytics YOLO for model_name='{model_name}'.")
                except Exception:
                    pass

        # If model loaded, tune default confidence
        if _yolo is not None:
            try:
                _yolo.conf = 0.25
            except Exception:
                # some model objects don't expose conf attribute
                pass
    return _yolo


def get_easyocr(model_storage_directory: str = "/workspace/data/easyocr_models", gpu: bool = True, download_enabled: bool = False):
    global _easyocr
    if _easyocr is None:
        import easyocr

        _easyocr = easyocr.Reader(["en"], gpu=gpu, download_enabled=download_enabled, model_storage_directory=model_storage_directory, verbose=False)
    return _easyocr


def get_fer(mtcnn: bool = True):
    global _fer
    if _fer is None:
        from fer import FER

        _fer = FER(mtcnn=mtcnn)
    return _fer


def get_text_model(model_name: str = "all-MiniLM-L6-v2"):
    global _text_model
    if _text_model is None:
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _text_model = SentenceTransformer(model_name, device=device)
    return _text_model


def read_image_local(path: str) -> Optional[np.ndarray]:
    try:
        local = path.replace("file://", "").replace("file:", "")
        img = cv2.imread(local)
        return img
    except Exception:
        return None


def run_yolo_on_image(img_or_path: Any, model_name: str = "yolov5s") -> List[Dict[str, Any]]:
    """Return list of detection dicts: label, conf, bbox (x1,y1,x2,y2) and crop (numpy BGR)"""
    model = get_yolo(model_name)
    if model is None:
        # YOLO failed to load; inform and return no detections
        try:
            print(f"[feature_helpers] INFO: YOLO model not available for model_name='{model_name}'. Skipping object detection for this image.")
        except Exception:
            pass
        return []
    # accept path or numpy
    if isinstance(img_or_path, str):
        results = model(img_or_path)
        df = results.pandas().xyxy[0]
        img = read_image_local(img_or_path)
    else:
        results = model(img_or_path)
        df = results.pandas().xyxy[0]
        img = img_or_path

    recs = []
    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax)
        crop = None
        if img is not None and y2 > y1 and x2 > x1:
            h, w = img.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            if x2c > x1c and y2c > y1c:
                crop = img[y1c:y2c, x1c:x2c]
        recs.append({
            "label": row["name"],
            "conf": float(row["confidence"]),
            "bbox": [x1, y1, x2, y2],
            "crop": crop,
        })
    return recs


def color_stats_from_crop(crop: Optional[np.ndarray]) -> Dict[str, float]:
    if crop is None:
        return {}
    try:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(rgb)
        stats = {
            "r_mean": float(r.mean()),
            "g_mean": float(g.mean()),
            "b_mean": float(b.mean()),
            "r_std": float(r.std()),
            "g_std": float(g.std()),
            "b_std": float(b.std()),
        }
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        stats.update({"h_mean": float(h.mean()), "s_mean": float(s.mean()), "v_mean": float(v.mean())})
        return stats
    except Exception:
        return {}


def run_fer_on_crop(crop: Optional[np.ndarray]) -> Dict[str, Any]:
    if crop is None:
        return {}
    try:
        fer = get_fer()
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        faces = fer.detect_emotions(rgb)
        if not faces:
            return {}
        top = faces[0]
        emotions = top.get("emotions", {})
        dominant = max(emotions.items(), key=lambda x: x[1])[0] if emotions else None
        return {"emotions": emotions, "dominant_emotion": dominant}
    except Exception:
        return {"fer_error": True}


def run_easyocr_on_array_or_path(img_or_path: Any, model_storage_directory: str = "/workspace/data/easyocr_models", gpu: bool = True, download_enabled: bool = False, detail: int = 1) -> List[Any]:
    reader = get_easyocr(model_storage_directory=model_storage_directory, gpu=gpu, download_enabled=download_enabled)
    try:
        if isinstance(img_or_path, str):
            local = img_or_path.replace("file://", "").replace("file:", "")
            return reader.readtext(local, detail=detail)
        else:
            # assume numpy array BGR -> convert to RGB
            rgb = cv2.cvtColor(img_or_path, cv2.COLOR_BGR2RGB)
            return reader.readtext(rgb, detail=detail)
    except Exception:
        return []


def text_to_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    if text is None:
        text = ""
    text = text.strip()
    model = get_text_model(model_name)
    if text == "":
        dim = model.get_sentence_embedding_dimension()
        return [0.0] * dim
    try:
        emb = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return emb.astype(float).tolist()
    except Exception:
        # fallback empty vector
        dim = model.get_sentence_embedding_dimension()
        return [0.0] * dim


def safe_extract_features(image_path: str, model_name: str = "yolov5s", easyocr_model_dir: str = "/workspace/data/easyocr_models", easyocr_download: bool = False) -> List[Dict[str, Any]]:
    """Return list of records (per-detection). Each record is JSON-serializable."""
    records = []
    try:
        img = read_image_local(image_path)
        # YOLO detections
        dets = run_yolo_on_image(img if img is not None else image_path, model_name=model_name)

        # OCR global once
        ocr_global = run_easyocr_on_array_or_path(image_path, model_storage_directory=easyocr_model_dir, gpu=torch.cuda.is_available(), download_enabled=easyocr_download, detail=1)
        ocr_global_text = " ".join([r[1] for r in ocr_global]) if ocr_global else ""

        if dets:
            for i, d in enumerate(dets):
                crop = d.get("crop")
                cstats = color_stats_from_crop(crop)
                fer_res = run_fer_on_crop(crop)
                ocr_crop = run_easyocr_on_array_or_path(crop, model_storage_directory=easyocr_model_dir, gpu=torch.cuda.is_available(), download_enabled=easyocr_download, detail=1) if crop is not None else []
                ocr_text = " ".join([r[1] for r in ocr_crop]) if ocr_crop else ""
                emb = text_to_embedding(ocr_text if ocr_text else ocr_global_text)
                rec = {
                    "image_path": image_path,
                    "detection_id": i,
                    "label": d.get("label"),
                    "conf": d.get("conf"),
                    "bbox": d.get("bbox"),
                    "ocr_text": ocr_text,
                    "ocr_global_text": ocr_global_text,
                    "embedding": emb,
                }
                rec.update(cstats)
                rec.update(fer_res if isinstance(fer_res, dict) else {})
                records.append(rec)
        else:
            # no detections: record global OCR
            emb = text_to_embedding(ocr_global_text)
            records.append({
                "image_path": image_path,
                "detection_id": -1,
                "label": None,
                "conf": None,
                "bbox": None,
                "ocr_text": "",
                "ocr_global_text": ocr_global_text,
                "embedding": emb,
            })
    except Exception as e:
        records.append({"image_path": image_path, "error": str(e), "traceback": traceback.format_exc()})
    return records
