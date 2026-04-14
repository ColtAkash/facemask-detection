# Face Mask Detection — Deep Learning Group Project

End-to-end face mask classification: three-class detection
(`with_mask`, `without_mask`, `mask_weared_incorrect`) trained in TensorFlow/Keras,
served by a FastAPI backend with OpenCV face detection, and consumed by a React + Vite
web app with live webcam inference.

## Group Members

| # | Name | Responsibility |
|---|------|-----------------|
| 1 | Member 1 | Data preprocessing, shared training utilities |
| 2 | Member 2 | Experiment 1 — Custom CNN architectures |
| 3 | Member 3 | Experiment 2 — MobileNetV2 / VGG16 transfer learning |
| 4 | Member 4 | Experiment 3 — EfficientNet + ViT, comparison plots |
| 5 | Member 5 | Inference pipeline + FastAPI backend |
| 6 | Akash   | Colab training runner, frontend integration, repo scaffold |

## Repository Layout

```
.
├── src/                     # Python training + inference code
│   ├── data_preprocessing.py
│   ├── utils.py             # perf mode, AdamW, label smoothing, callbacks
│   ├── experiment1_custom_cnn.py
│   ├── experiment2_transfer_learning.py
│   ├── experiment3_sota.py
│   ├── compare_results.py
│   └── predict.py           # Haar face detection + model inference
├── backend/                 # FastAPI service wrapping the best .keras model
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── frontend/                # React + Vite + Tailwind webcam client
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── run_all.py               # Runs all 3 experiments + comparison
├── colab_train.ipynb        # Google Colab training notebook (T4 GPU)
├── requirements.txt         # Python deps for training
└── .github/workflows/       # Pages deploy (frontend/)
```

The `dataset/`, `dataset_processed/`, and `results/` folders are git-ignored — they
contain raw images and trained checkpoints.

## 1. Training (local or Colab)

### Local

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_all.py                 # full pipeline
python run_all.py --only 1 --fast # single experiment, mixed-precision + XLA
```

Dataset is expected under `dataset/` with `Train/Validation/Test` splits and
class subfolders (`with_mask/`, `without_mask/`, `mask_weared_incorrect/`).

### Google Colab (recommended for GPU runs)

Open `colab_train.ipynb` in Colab, select a T4 GPU runtime, and follow the cells.
The notebook unzips `project.zip` + `dataset_processed.zip` from your Drive,
runs `run_all.py --fast`, then exports a timestamped `results/` folder back to Drive.

## 2. Backend API

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

The service auto-selects the best `.keras` checkpoint under `results/`
(override with `MODEL_PATH=...`). Swagger UI at <http://localhost:8000/docs>.

Endpoints:

| Method | Path              | Purpose                                |
|--------|-------------------|----------------------------------------|
| GET    | `/health`         | Liveness + loaded model name           |
| POST   | `/predict`        | Classify whole uploaded image          |
| POST   | `/predict/faces`  | Detect faces (Haar) + classify each    |

See [backend/README.md](backend/README.md) for response schemas.

## 3. Frontend

```bash
cd frontend
npm install
cp .env.example .env.local        # point VITE_API_URL at your backend
npm run dev
```

The webcam page captures frames → JPEG blob → POSTs to `/predict` at ~5 fps,
renders the live class probabilities.

## Hardware

Training works on CPU (slow) but benefits heavily from a GPU. On Windows,
native TF 2.x has no GPU support — use Colab (T4) or WSL2. The included
Colab notebook does a full run in ~20 minutes with `--fast`.

## Troubleshooting

- **`FileNotFoundError: dataset/Train not found`** — ensure the dataset
  folder structure matches the one expected by `src/data_preprocessing.py`.
- **`ModuleNotFoundError: tensorflow`** — `pip install -r requirements.txt`.
- **GPU OOM** — lower `BATCH_SIZE` in the relevant experiment script.
- **ViT weights download fails** — pass `--skip_vit` to `run_all.py`.
- **Backend can't find a model** — set `MODEL_PATH` to a specific `.keras` file.
