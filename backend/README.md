# Face Mask Detection — Backend API

FastAPI service that wraps the best trained Keras model and exposes HTTP
endpoints for classification and per-face detection.

## Run

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs for an interactive Swagger UI.

## Model selection

On startup the service loads the best available `.keras` checkpoint from
`results/` using the same priority list as `src/predict.py`. Override with:

```bash
MODEL_PATH=results/experiment2/mobilenetv2_finetuned/phase2_best.keras \
  uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

## Endpoints

| Method | Path              | Purpose                                  |
|--------|-------------------|------------------------------------------|
| GET    | `/health`         | Liveness + loaded model name             |
| POST   | `/predict`        | Classify whole uploaded image            |
| POST   | `/predict/faces`  | Detect faces and classify each bbox      |

### Example (curl)

```bash
# Whole-image classification
curl -F "file=@photo.jpg" http://localhost:8000/predict

# Per-face detection + classification
curl -F "file=@group_photo.jpg" http://localhost:8000/predict/faces
```

### Response shapes

`/predict` returns:
```json
{
  "label": "with_mask",
  "confidence": 0.982,
  "probabilities": {"with_mask": 0.982, "without_mask": 0.012, "mask_weared_incorrect": 0.006}
}
```

`/predict/faces` returns:
```json
{
  "model": "phase2_best.keras",
  "num_faces": 2,
  "faces": [
    {"label": "with_mask", "confidence": 0.97,
     "probabilities": {...},
     "box": {"x": 120, "y": 80, "w": 150, "h": 150}}
  ]
}
```
