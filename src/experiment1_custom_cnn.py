"""
experiment1_custom_cnn.py
-----------------------------------------------------------------------------
Experiment 1 - Supervised Learning: Custom CNN Architectures
-----------------------------------------------------------------------------
Dataset: andrewmvd/face-mask-detection (3 classes)
  -> with_mask | without_mask | mask_weared_incorrect

Trains and compares three hand-crafted CNN architectures:
  Arch A - Small  (2 conv blocks, no regularisation)
  Arch B - Medium (3 conv blocks, BatchNorm + Dropout)
  Arch C - Large  (4 conv blocks, BatchNorm + Dropout + L2)

Each architecture is tested with two learning rates: 1e-3 and 1e-4.
Class weights are applied to handle the imbalanced label distribution.

Usage:
    python src/experiment1_custom_cnn.py [--data_dir DATASET_PATH]
"""

import os, sys, argparse, json, time
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from utils import (ensure_dir, plot_training_history, plot_confusion_matrix,
                   plot_sample_images, plot_class_distribution,
                   compute_metrics, save_metrics, print_metrics, get_predictions,
                   enable_performance_mode, build_optimizer, build_loss,
                   standard_callbacks)

# --- Configuration ------------------------------------------------------------
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
EPOCHS      = 30
AUTOTUNE    = tf.data.AUTOTUNE
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'experiment1')
CLASS_NAMES = ['with_mask', 'without_mask', 'mask_weared_incorrect']
NUM_CLASSES = len(CLASS_NAMES)


# --- Data loading -------------------------------------------------------------

def compute_class_weights(train_dir):
    """Compute inverse-frequency class weights to handle imbalance."""
    counts = np.array([
        len(os.listdir(os.path.join(train_dir, c)))
        for c in CLASS_NAMES
        if os.path.isdir(os.path.join(train_dir, c))
    ], dtype=float)
    weights = counts.sum() / (NUM_CLASSES * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def load_datasets(data_dir):
    train_dir = os.path.join(data_dir, 'Train')
    val_dir   = os.path.join(data_dir, 'Validation')
    test_dir  = os.path.join(data_dir, 'Test')

    for d in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Expected directory not found: {d}")

    norm = tf.keras.layers.Rescaling(1.0 / 255)
    aug  = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ], name='augmentation')

    def aug_and_norm(x, y): return norm(aug(x, training=True)), y
    def norm_only(x, y):    return norm(x), y

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        seed=42, label_mode='int', class_names=CLASS_NAMES
    ).map(aug_and_norm, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        seed=42, label_mode='int', class_names=CLASS_NAMES, shuffle=False
    ).map(norm_only, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        seed=42, label_mode='int', class_names=CLASS_NAMES, shuffle=False
    ).map(norm_only, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    class_weights = compute_class_weights(train_dir)
    print(f"  Class weights: { {CLASS_NAMES[i]: f'{w:.2f}' for i, w in class_weights.items()} }")

    return train_ds, val_ds, test_ds, class_weights


# --- Model definitions --------------------------------------------------------

def build_arch_a(input_shape, name='Arch_A_Small'):
    """Architecture A - 2 conv blocks, no regularisation."""
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    return tf.keras.Model(inputs, outputs, name=name)


def build_arch_b(input_shape, name='Arch_B_Medium'):
    """Architecture B - 3 conv blocks, BatchNorm + Dropout."""
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for filters in [32, 64, 128]:
        x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.50)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    return tf.keras.Model(inputs, outputs, name=name)


def build_arch_c(input_shape, name='Arch_C_Large'):
    """Architecture C - 4 conv blocks, BatchNorm + Dropout + L2."""
    reg = tf.keras.regularizers.l2(1e-4)
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for i, filters in enumerate([32, 64, 128, 256]):
        x = tf.keras.layers.Conv2D(filters, 3, padding='same',
                                   kernel_regularizer=reg)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same',
                                   kernel_regularizer=reg)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.20 + 0.05 * i)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(0.50)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    return tf.keras.Model(inputs, outputs, name=name)


ARCHITECTURES  = {'Arch_A_Small': build_arch_a,
                  'Arch_B_Medium': build_arch_b,
                  'Arch_C_Large': build_arch_c}
LEARNING_RATES = [1e-3, 1e-4]


# --- Training -----------------------------------------------------------------

def train_model(model, train_ds, val_ds, lr, run_name, result_dir, class_weights):
    model.compile(
        optimizer=build_optimizer(lr, weight_decay=1e-4),
        loss=build_loss(label_smoothing=0.1, num_classes=NUM_CLASSES),
        metrics=['accuracy']
    )
    callbacks = standard_callbacks(
        os.path.join(result_dir, f'{run_name}_best.keras'),
        patience_es=6, patience_lr=3, min_lr=1e-6)
    print(f"\n  Training: {run_name}  (lr={lr})")
    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=callbacks,
        class_weight=class_weights, verbose=1
    )
    return history, time.time() - t0


# --- Main ---------------------------------------------------------------------

def main(data_dir, fast=False):
    ensure_dir(RESULTS_DIR)
    enable_performance_mode(mixed_precision=fast, xla=True)
    print(f"\n{'='*60}")
    print("  EXPERIMENT 1 - Custom CNN Architectures")
    print(f"  Dataset : {data_dir}")
    print(f"  Classes : {CLASS_NAMES}")
    print(f"  Results : {RESULTS_DIR}")
    print('='*60)

    # EDA
    train_dir = os.path.join(data_dir, 'Train')
    plot_sample_images(train_dir, CLASS_NAMES,
                       os.path.join(RESULTS_DIR, 'sample_images.png'))
    plot_class_distribution(train_dir, CLASS_NAMES,
                            os.path.join(RESULTS_DIR, 'class_distribution.png'),
                            title='Training Set - Class Distribution')

    train_ds, val_ds, test_ds, class_weights = load_datasets(data_dir)
    input_shape = IMG_SIZE + (3,)
    all_results = {}

    for arch_name, build_fn in ARCHITECTURES.items():
        for lr in LEARNING_RATES:
            run_name = f'{arch_name}_lr{lr}'
            run_dir  = os.path.join(RESULTS_DIR, run_name)
            ensure_dir(run_dir)

            model   = build_fn(input_shape)
            history, elapsed = train_model(
                model, train_ds, val_ds, lr, run_name, run_dir, class_weights)

            plot_training_history(
                history.history, f'{arch_name} | lr={lr}',
                os.path.join(run_dir, 'training_history.png'))

            y_true, y_pred = get_predictions(model, test_ds)
            metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
            metrics.update({
                'training_time_s': round(elapsed, 1),
                'params': model.count_params(),
                'learning_rate': lr,
                'architecture': arch_name,
            })
            print_metrics(metrics, label=run_name)

            plot_confusion_matrix(
                y_true, y_pred, CLASS_NAMES,
                f'Confusion Matrix - {arch_name} (lr={lr})',
                os.path.join(run_dir, 'confusion_matrix.png'))

            save_metrics(metrics, os.path.join(run_dir, 'metrics.json'))
            all_results[run_name] = {k: v for k, v in metrics.items()
                                     if k != 'classification_report'}

    # -- Summary bar chart --------------------------------------------------
    import matplotlib.pyplot as plt
    names      = list(all_results.keys())
    accuracies = [all_results[n]['accuracy'] for n in names]
    f1s        = [all_results[n]['f1_score']  for n in names]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.8), 6))
    ax.bar(x - w/2, accuracies, w, label='Accuracy', color='steelblue', alpha=0.85)
    ax.bar(x + w/2, f1s,        w, label='F1 Score',  color='salmon',    alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylim([0, 1.05]); ax.set_ylabel('Score')
    ax.set_title('Experiment 1 - Custom CNN Comparison')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'experiment1_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    with open(os.path.join(RESULTS_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n  [Done] Results saved to {RESULTS_DIR}")
    print(f"\n{'Experiment 1 - Summary':^70}")
    print(f"{'Model':<40} {'Accuracy':>10} {'F1':>10} {'Params':>12}")
    print('-' * 75)
    for name, res in all_results.items():
        print(f"{name:<40} {res['accuracy']:>10.4f} "
              f"{res['f1_score']:>10.4f} {res['params']:>12,}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default=os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    parser.add_argument('--fast', action='store_true',
                        help='Enable mixed_float16 (GPU) + XLA JIT for faster training')
    args = parser.parse_args()
    main(os.path.abspath(args.data_dir), fast=args.fast)
