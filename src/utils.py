"""
utils.py - Shared utility functions for Face Mask Detection Project
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)


# --- Training performance helpers --------------------------------------------

def enable_performance_mode(mixed_precision: bool = False, xla: bool = True):
    """Enable GPU memory growth, optional mixed precision, and XLA JIT.

    Safe to call on CPU-only machines: mixed precision is skipped when no GPU
    is visible. Final classifier layers must use dtype='float32' so softmax
    stays numerically stable when mixed_precision=True.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    if mixed_precision and gpus:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("  [Perf] mixed_float16 enabled (keep final Dense dtype='float32')")
        except Exception as e:
            print(f"  [Perf] mixed precision unavailable: {e}")
    elif mixed_precision:
        print("  [Perf] mixed precision requested but no GPU detected - skipped")

    if xla:
        try:
            tf.config.optimizer.set_jit(True)
            print("  [Perf] XLA JIT enabled")
        except Exception as e:
            print(f"  [Perf] XLA unavailable: {e}")

    print(f"  [Perf] GPUs visible: {len(gpus)}")


def build_optimizer(lr: float, weight_decay: float = 1e-4):
    """AdamW with weight decay; falls back to Adam on older TF."""
    import tensorflow as tf
    try:
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    except AttributeError:
        return tf.keras.optimizers.Adam(learning_rate=lr)


def build_loss(label_smoothing: float = 0.1, num_classes: int = 3):
    """Categorical cross-entropy with label smoothing.

    Returns a loss that expects integer labels (SparseCategoricalCrossentropy-compatible)
    by one-hot encoding on the fly inside a closure.
    """
    import tensorflow as tf
    cce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(tf.reshape(y_true, [-1]), depth=num_classes)
        return cce(y_true, y_pred)
    return loss_fn


def standard_callbacks(ckpt_path: str, patience_es: int = 7, patience_lr: int = 3,
                       min_lr: float = 1e-7):
    import tensorflow as tf
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=patience_es,
            restore_best_weights=True, min_delta=1e-4, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=patience_lr,
            min_lr=min_lr, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path, monitor='val_accuracy',
            save_best_only=True, verbose=0),
    ]


# --- Directory helpers --------------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# --- Plotting -----------------------------------------------------------------

def plot_training_history(history, title, save_path):
    """Plot and save accuracy + loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['accuracy']) + 1)

    axes[0].plot(epochs, history['accuracy'],     'b-o', markersize=4, label='Train')
    axes[0].plot(epochs, history['val_accuracy'], 'r-o', markersize=4, label='Validation')
    axes[0].set_title('Accuracy');  axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_ylim([0, 1.05])

    axes[1].plot(epochs, history['loss'],     'b-o', markersize=4, label='Train')
    axes[1].plot(epochs, history['val_loss'], 'r-o', markersize=4, label='Validation')
    axes[1].set_title('Loss');  axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Training history -> {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor='gray', annot_kws={"size": 14})
    plt.title(title, fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Confusion matrix -> {save_path}")


def plot_sample_images(data_dir, class_names, save_path, n_samples=8):
    """Plot sample images from the dataset."""
    import random
    from PIL import Image

    all_images = []
    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            files = [f for f in os.listdir(cls_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for f in files:
                all_images.append((os.path.join(cls_path, f), cls))

    if not all_images:
        print(f"  [Warning] No images found in {data_dir}")
        return

    samples = random.sample(all_images, min(n_samples, len(all_images)))
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).flatten()

    for i, (img_path, label) in enumerate(samples):
        img = Image.open(img_path).convert('RGB')
        axes[i].imshow(img)
        color = 'green' if 'WithMask' in label or 'with_mask' in label else 'red'
        axes[i].set_title(label, fontsize=10, color=color, fontweight='bold')
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Sample Dataset Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Sample images -> {save_path}")


def plot_class_distribution(data_dir, class_names, save_path, title='Class Distribution'):
    """Bar chart of class counts."""
    counts = {}
    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            n = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            counts[cls] = n

    plt.figure(figsize=(8, 5))
    colors = ['steelblue', 'salmon', 'mediumseagreen']
    bars = plt.bar(counts.keys(), counts.values(),
                   color=colors[:len(counts)], edgecolor='black', alpha=0.85)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 20,
                 f'{int(bar.get_height()):,}', ha='center', va='bottom', fontweight='bold')
    plt.title(title, fontsize=13, fontweight='bold')
    plt.xlabel('Class'); plt.ylabel('Number of Images')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Class distribution -> {save_path}")


# --- Metrics ------------------------------------------------------------------

def compute_metrics(y_true, y_pred, class_names):
    avg = 'binary' if len(class_names) == 2 else 'weighted'
    return {
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'f1_score':  float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        'classification_report': classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0)
    }


def save_metrics(metrics_dict, save_path):
    saveable = {k: v for k, v in metrics_dict.items() if k != 'classification_report'}
    with open(save_path, 'w') as f:
        json.dump(saveable, f, indent=4)
    print(f"  [Saved] Metrics JSON -> {save_path}")


def load_metrics(path):
    with open(path, 'r') as f:
        return json.load(f)


def print_metrics(metrics, label=""):
    print(f"\n{'='*50}")
    if label:
        print(f"  Results: {label}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 Score : {metrics['f1_score']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    if 'classification_report' in metrics:
        print(f"\n{metrics['classification_report']}")
    print('='*50)


# --- Model summary ------------------------------------------------------------

def count_params(model):
    total = model.count_params()
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}


def get_predictions(model, dataset, steps=None):
    """Get true labels and predicted labels from a tf.data dataset."""
    y_true, y_prob = [], []
    for batch_x, batch_y in dataset:
        preds = model.predict(batch_x, verbose=0)
        y_prob.extend(preds.flatten() if preds.shape[-1] == 1 else preds.tolist())
        y_true.extend(batch_y.numpy().flatten().tolist())

    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob)
    if y_prob.ndim == 1:                   # binary sigmoid
        y_pred = (y_prob >= 0.5).astype(int)
    else:                                  # multi-class softmax
        y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred
