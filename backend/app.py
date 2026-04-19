"""
backend/app.py
-----------------------------------------------------------------------------
FastAPI backend for the Face Mask Detection project.

Endpoints
    GET  /health                 liveness + which model is loaded
    POST /predict                classify whole image (single face/full frame)
    POST /predict/faces          detect faces, classify each bbox

Run locally:
    pip install -r backend/requirements.txt
    uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

Request
    multipart/form-data with a single 'file' field containing an image.

Response (JSON) - see endpoint docstrings for schema.
"""

import io
import os
import sys
from typing import List, Optional

import numpy as np
from PIL import Image

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make src/ importable so we reuse predict.py helpers (find_best_model, etc.)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from predict import (  # type: ignore[import-not-found]
    CLASS_NAMES, detect_faces, find_best_model, preprocess,
)

# --- App setup ---------------------------------------------------------------

app = FastAPI(
    title='Face Mask Detection API',
    version='1.0.0',
    description='Wraps a trained Keras model for mask / no-mask / incorrect-mask '
                'classification. Supports whole-frame and per-face inference.',
)

# Allow the Vite dev server and the built SPA to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)

# --- Model loading -----------------------------------------------------------

MODEL: Optional[tf.keras.Model] = None
MODEL_PATH: Optional[str] = None
IMG_SIZE: tuple = (224, 224)


def _load():
    global MODEL, MODEL_PATH, IMG_SIZE
    path = os.environ.get('MODEL_PATH') or find_best_model()
    if not path or not os.path.isfile(path):
        raise RuntimeError(
            'No trained model found. Train first (python run_all.py) or set '
            'MODEL_PATH env var to a .keras file.')
    MODEL = tf.keras.models.load_model(path, compile=False)
    MODEL_PATH = path
    H, W = MODEL.input_shape[1], MODEL.input_shape[2]
    IMG_SIZE = (H, W)
    print(f'[backend] loaded {path}  input={H}x{W}  classes={CLASS_NAMES}')


@app.on_event('startup')
def _on_start():
    _load()


# --- Schemas -----------------------------------------------------------------

class ClassProbs(BaseModel):
    with_mask: float
    without_mask: float
    mask_weared_incorrect: float


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: ClassProbs


class FaceBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class FacePrediction(Prediction):
    box: FaceBox


class FacesResponse(BaseModel):
    model: str
    num_faces: int
    faces: List[FacePrediction]


class HealthResponse(BaseModel):
    status: str
    model: Optional[str]
    input_size: List[int]
    classes: List[str]


# --- Helpers -----------------------------------------------------------------

def _read_image(file: UploadFile) -> Image.Image:
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(400, f'Expected an image, got {file.content_type}')
    data = file.file.read()
    if not data:
        raise HTTPException(400, 'Empty upload')
    try:
        return Image.open(io.BytesIO(data)).convert('RGB')
    except Exception as exc:
        raise HTTPException(400, f'Could not decode image: {exc}') from exc


def _classify(pil_img: Image.Image) -> Prediction:
    arr = preprocess(pil_img, IMG_SIZE, MODEL_PATH or '')
    probs = MODEL.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return Prediction(
        label=CLASS_NAMES[idx],
        confidence=float(probs[idx]),
        probabilities=ClassProbs(
            with_mask=float(probs[0]),
            without_mask=float(probs[1]),
            mask_weared_incorrect=float(probs[2]),
        ),
    )


# --- Routes ------------------------------------------------------------------

@app.get('/health', response_model=HealthResponse)
def health():
    return HealthResponse(
        status='ok' if MODEL is not None else 'model_not_loaded',
        model=os.path.basename(MODEL_PATH) if MODEL_PATH else None,
        input_size=list(IMG_SIZE),
        classes=CLASS_NAMES,
    )


@app.post('/predict', response_model=Prediction)
def predict(file: UploadFile = File(...)):
    """Classify the whole uploaded image. Use when the image is already
    cropped to a single face, or when you just want a coarse label."""
    if MODEL is None:
        raise HTTPException(503, 'Model not loaded')
    img = _read_image(file)
    return _classify(img)


@app.post('/predict/faces', response_model=FacesResponse)
def predict_faces(file: UploadFile = File(...), pad: float = 0.15):
    """Detect faces (Haar cascade), then classify each bbox.

    - `pad` pads each detected box by a fraction of its size before classifying
      (so the model sees a bit of context). Default 0.15.
    - If no face is detected, the `faces` list is empty; call `/predict` for
      whole-frame classification.
    """
    if MODEL is None:
        raise HTTPException(503, 'Model not loaded')
    img = _read_image(file)

    # detect_faces wants BGR np.array - convert from PIL RGB
    import cv2  # imported lazily to keep startup fast
    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    boxes = detect_faces(bgr)

    H, W = bgr.shape[:2]
    results: List[FacePrediction] = []
    for (x, y, w, h) in boxes:
        px, py = int(w * pad), int(h * pad)
        x0 = max(0, x - px); y0 = max(0, y - py)
        x1 = min(W, x + w + px); y1 = min(H, y + h + py)
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        pred = _classify(Image.fromarray(crop))
        results.append(FacePrediction(
            **pred.dict(),
            box=FaceBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0),
        ))

    return FacesResponse(
        model=os.path.basename(MODEL_PATH) if MODEL_PATH else '',
        num_faces=len(results),
        faces=results,
    )
