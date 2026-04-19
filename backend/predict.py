"""
predict.py
-----------------------------------------------------------------------------
Run inference with any saved model on:
  - A single image file
  - A folder of images (saves a results grid)
  - Your webcam (live detection)

Usage:
    # Single image
    python src/predict.py --image path/to/photo.jpg

    # Folder of images
    python src/predict.py --folder dataset_processed/Test/with_mask

    # Webcam (press Q to quit)
    python src/predict.py --webcam

    # Pick a specific model (default = best available)
    python src/predict.py --image photo.jpg --model results/experiment2/mobilenetv2_finetuned/phase2_best.keras
"""

import os, sys, argparse
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress TF startup noise
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---- Config -----------------------------------------------------------------
CLASS_NAMES  = ['with_mask', 'without_mask', 'mask_weared_incorrect']
CLASS_COLORS = {
    'with_mask':              '#2ecc71',   # green
    'without_mask':           '#e74c3c',   # red
    'mask_weared_incorrect':  '#f39c12',   # orange
}
CLASS_EMOJI  = {
    'with_mask':             '[MASK ON]',
    'without_mask':          '[NO MASK]',
    'mask_weared_incorrect': '[MASK WRONG]',
}

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results')

# Haar-cascade face detector (ships with opencv-python, no download needed)
_FACE_CASCADE = None


def get_face_detector():
    """Lazy-load OpenCV's Haar cascade face detector."""
    global _FACE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE
    try:
        import cv2
        xml_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade  = cv2.CascadeClassifier(xml_path)
        if cascade.empty():
            print(f"  [Warn] Face cascade failed to load from {xml_path}")
            return None
        _FACE_CASCADE = cascade
        return cascade
    except Exception as e:
        print(f"  [Warn] Could not init face detector: {e}")
        return None


def detect_faces(frame_bgr, min_size=60):
    """Return list of (x, y, w, h) face boxes in a BGR frame."""
    cascade = get_face_detector()
    if cascade is None:
        return []
    import cv2
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)
    boxes = cascade.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=5,
        minSize=(min_size, min_size))
    return [tuple(map(int, b)) for b in boxes]

# Priority order: best model wins
MODEL_PRIORITY = [
    'experiment2/mobilenetv2_finetuned/phase2_best.keras',
    'experiment2/mobilenetv2_feature_extraction/best.keras',
    'experiment3/efficientnet_pretrained/phase2_best.keras',
    'experiment3/efficientnet_scratch/best.keras',
    'experiment2/vgg16_feature_extraction/best.keras',
    'experiment3/custom_vit/best.keras',
]


# ---- Model loading ----------------------------------------------------------

def find_best_model():
    for rel in MODEL_PRIORITY:
        path = os.path.join(RESULTS_DIR, rel)
        if os.path.isfile(path):
            return path
    # Fallback: any .keras file
    for root, _, files in os.walk(RESULTS_DIR):
        for f in files:
            if f.endswith('.keras'):
                return os.path.join(root, f)
    return None


def load_model(model_path):
    print(f"  Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Detect expected input size from model
    inp = model.input_shape   # (None, H, W, C)
    img_size = (inp[1], inp[2])
    print(f"  Input size : {img_size[0]}x{img_size[1]}")
    print(f"  Output     : {len(CLASS_NAMES)} classes -> {CLASS_NAMES}")
    return model, img_size


# ---- Preprocessing ----------------------------------------------------------

def preprocess(img_pil, img_size, model_path):
    """Resize + apply the correct backbone preprocessing."""
    img = img_pil.convert('RGB').resize(img_size)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, 0)          # (1, H, W, 3)

    path_lower = model_path.lower()
    if 'mobilenet' in path_lower:
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    elif 'vgg16' in path_lower:
        arr = tf.keras.applications.vgg16.preprocess_input(arr)
    elif 'efficientnet' in path_lower:
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    else:
        arr = arr / 255.0                 # custom CNN / ViT

    return arr


# ---- Single prediction ------------------------------------------------------

def predict_one(model, img_pil, img_size, model_path):
    """Return (class_name, confidence, all_probs)."""
    x     = preprocess(img_pil, img_size, model_path)
    probs = model.predict(x, verbose=0)[0]           # shape: (3,)
    idx   = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs


# ---- Display helpers --------------------------------------------------------

def annotate_image(img_pil, label, confidence, probs, save_path=None):
    """Draw prediction label + confidence bar on the image and save/show."""
    W, H = img_pil.size
    bar_h = max(60, H // 6)
    canvas = Image.new('RGB', (W, H + bar_h), (30, 30, 30))
    canvas.paste(img_pil, (0, 0))

    draw  = ImageDraw.Draw(canvas)
    color = CLASS_COLORS.get(label, '#ffffff')
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    # Filled bar
    draw.rectangle([0, H, W, H + bar_h], fill=(r, g, b))

    # Text
    tag  = CLASS_EMOJI.get(label, label.upper())
    text = f"{tag}  {confidence*100:.1f}%"
    try:
        font = ImageFont.truetype("arial.ttf", max(16, bar_h // 3))
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, H + bar_h // 4), text, fill='white', font=font)

    if save_path:
        canvas.save(save_path)
        print(f"  [Saved] {save_path}")

    return canvas


def print_result(label, confidence, probs, filename=""):
    tag = CLASS_EMOJI.get(label, label)
    bar_full = 30
    print(f"\n  {'='*50}")
    if filename:
        print(f"  File  : {os.path.basename(filename)}")
    print(f"  Result: {tag}")
    print(f"  Conf  : {confidence*100:.1f}%")
    print(f"\n  Class probabilities:")
    for i, (cls, p) in enumerate(zip(CLASS_NAMES, probs)):
        filled = int(p * bar_full)
        bar    = '#' * filled + '-' * (bar_full - filled)
        marker = ' <-- PREDICTION' if i == int(np.argmax(probs)) else ''
        print(f"    {cls:<30} [{bar}] {p*100:5.1f}%{marker}")
    print(f"  {'='*50}")


# ---- Image folder mode ------------------------------------------------------

def predict_folder(model, img_size, model_path, folder, max_images=16):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ])[:max_images]

    if not files:
        print(f"  [Error] No images found in {folder}")
        return

    print(f"\n  Running predictions on {len(files)} images from: {folder}")

    cols    = min(4, len(files))
    rows    = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes    = np.array(axes).flatten() if len(files) > 1 else [axes]

    counts  = {c: 0 for c in CLASS_NAMES}

    for i, fpath in enumerate(files):
        img = Image.open(fpath).convert('RGB')
        label, conf, probs = predict_one(model, img, img_size, model_path)
        counts[label] += 1
        print_result(label, conf, probs, fpath)

        color = CLASS_COLORS.get(label, 'white')
        axes[i].imshow(img)
        axes[i].set_title(
            f"{CLASS_EMOJI[label]}\n{conf*100:.1f}%",
            fontsize=9, color=color, fontweight='bold')
        axes[i].axis('off')
        # Colored border
        for spine in axes[i].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Legend
    patches = [mpatches.Patch(color=CLASS_COLORS[c], label=f"{c} ({counts[c]})")
               for c in CLASS_NAMES]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=10, frameon=True)

    plt.suptitle(f'Face Mask Predictions  |  {len(files)} images',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    save_path = os.path.join(RESULTS_DIR, 'prediction_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [Saved] Prediction grid -> {save_path}")

    print(f"\n  Summary:")
    for cls, cnt in counts.items():
        print(f"    {cls:<30} {cnt:>4} / {len(files)}")


# ---- Single image mode ------------------------------------------------------

def predict_single(model, img_size, model_path, image_path):
    if not os.path.isfile(image_path):
        print(f"  [Error] File not found: {image_path}")
        sys.exit(1)

    img = Image.open(image_path).convert('RGB')
    label, conf, probs = predict_one(model, img, img_size, model_path)
    print_result(label, conf, probs, image_path)

    save_path = os.path.join(
        RESULTS_DIR,
        'prediction_' + os.path.splitext(os.path.basename(image_path))[0] + '.png')
    annotate_image(img, label, conf, probs, save_path=save_path)


# ---- Webcam mode ------------------------------------------------------------

_BGR = {
    'with_mask':             (46, 204, 113),    # green
    'without_mask':          (60,  76, 231),    # red (BGR)
    'mask_weared_incorrect': (18, 156, 243),    # orange (BGR)
}


def _annotate_frame(frame, model, img_size, model_path, use_faces=True, pad=0.15):
    """Draw per-face bounding boxes + labels on `frame` (BGR).

    If use_faces=False or no faces are detected, classifies the whole frame.
    Returns the annotated frame.
    """
    import cv2
    H, W = frame.shape[:2]
    faces = detect_faces(frame) if use_faces else []

    if faces:
        for (x, y, w, h) in faces:
            # Pad the bbox so the classifier sees a bit of context
            px, py = int(w * pad), int(h * pad)
            x0 = max(0, x - px); y0 = max(0, y - py)
            x1 = min(W, x + w + px); y1 = min(H, y + h + py)
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            label, conf, _ = predict_one(model, img, img_size, model_path)
            color = _BGR.get(label, (255, 255, 255))
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            tag  = CLASS_EMOJI.get(label, label)
            text = f"{tag} {conf*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x0, y0 - th - 8), (x0 + tw + 6, y0), color, -1)
            cv2.putText(frame, text, (x0 + 3, y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"faces: {len(faces)}", (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        label, conf, _ = predict_one(model, img, img_size, model_path)
        color = _BGR.get(label, (255, 255, 255))
        tag   = CLASS_EMOJI.get(label, label)
        cv2.rectangle(frame, (0, 0), (W, 44), color, -1)
        note  = "no face" if use_faces else "full frame"
        cv2.putText(frame, f"{tag} {conf*100:.1f}%  ({note})", (10, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def predict_webcam(model, img_size, model_path, use_faces=True):
    try:
        import cv2
    except ImportError:
        print("  [Error] opencv-python is required for webcam mode.")
        print("  Install with:  pip install opencv-python")
        sys.exit(1)

    print(f"\n  Starting webcam ... Press Q to quit.  face_detector={use_faces}")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [Error] Could not open webcam.")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = _annotate_frame(frame, model, img_size, model_path, use_faces=use_faces)
        cv2.imshow('Face Mask Detection  (Q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  Webcam closed.")


def predict_video(model, img_size, model_path, video_path, use_faces=True,
                  save_path=None, show=True):
    """Classify each detected face in a video file; save annotated output."""
    try:
        import cv2
    except ImportError:
        print("  [Error] opencv-python is required for video mode.")
        sys.exit(1)

    if not os.path.isfile(video_path):
        print(f"  [Error] Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [Error] Could not open video: {video_path}")
        sys.exit(1)

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if save_path is None:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        save_path = os.path.join(RESULTS_DIR, f'video_{stem}_annotated.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    print(f"\n  Video: {video_path}  ({W}x{H} @ {fps:.1f} fps, {total} frames)")
    print(f"  Writing annotated output -> {save_path}")

    face_counts = {c: 0 for c in CLASS_NAMES}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Tally predictions per-face on each frame
        for (x, y, w, h) in (detect_faces(frame) if use_faces else []):
            crop = frame[max(0,y):y+h, max(0,x):x+w]
            if crop.size == 0: continue
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            label, _, _ = predict_one(model, img, img_size, model_path)
            face_counts[label] += 1

        frame = _annotate_frame(frame, model, img_size, model_path, use_faces=use_faces)
        writer.write(frame)

        if show:
            cv2.imshow('Face Mask Detection - Video  (Q to quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1
        if total and frame_idx % max(1, total // 20) == 0:
            print(f"    progress: {frame_idx}/{total} "
                  f"({100.0*frame_idx/total:.0f}%)")

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\n  [Done] Saved: {save_path}")
    print(f"  Face detections by class (across all frames):")
    total_faces = sum(face_counts.values())
    for cls, cnt in face_counts.items():
        pct = (100.0 * cnt / total_faces) if total_faces else 0.0
        print(f"    {cls:<30} {cnt:>6}  ({pct:5.1f}%)")


# ---- Entry point ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Face Mask Detection - Inference')
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image',   help='Path to a single image file')
    group.add_argument('--folder',  help='Path to a folder of images')
    group.add_argument('--video',   help='Path to a video file (per-face bounding boxes)')
    group.add_argument('--webcam',  action='store_true', help='Live webcam detection')
    parser.add_argument('--model',  default=None,
                        help='Path to a .keras model file (auto-selects best if omitted)')
    parser.add_argument('--max',    type=int, default=16,
                        help='Max images to process in --folder mode (default: 16)')
    parser.add_argument('--no-faces', dest='use_faces', action='store_false',
                        help='Disable face detector; classify whole frame instead')
    parser.add_argument('--no-show', dest='show', action='store_false',
                        help='Do not display video window (video mode only)')
    parser.set_defaults(use_faces=True, show=True)
    args = parser.parse_args()

    # Resolve model
    model_path = args.model
    if model_path is None:
        model_path = find_best_model()
    if model_path is None or not os.path.isfile(model_path):
        print("[Error] No trained model found.")
        print("  Run the experiments first:  python run_all.py --only 1 --data_dir dataset_processed")
        sys.exit(1)

    model, img_size = load_model(model_path)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.image:
        predict_single(model, img_size, model_path, args.image)
    elif args.folder:
        predict_folder(model, img_size, model_path, args.folder, max_images=args.max)
    elif args.video:
        predict_video(model, img_size, model_path, args.video,
                      use_faces=args.use_faces, show=args.show)
    elif args.webcam:
        predict_webcam(model, img_size, model_path, use_faces=args.use_faces)


if __name__ == '__main__':
    main()
