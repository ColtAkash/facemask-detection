"""
experiment2_transfer_learning.py
-----------------------------------------------------------------------------
Experiment 2 - Unsupervised Feature Extraction via Transfer Learning
-----------------------------------------------------------------------------
Dataset: andrewmvd/face-mask-detection (3 classes)

The pre-trained ImageNet backbone acts as a frozen unsupervised feature
extractor; only the new classification head is trained with supervised labels.

  Strategy 1 - MobileNetV2  feature extraction  (base fully frozen)
  Strategy 2 - MobileNetV2  fine-tuning         (top 30 layers unfrozen)
  Strategy 3 - VGG16        feature extraction  (base fully frozen)

Usage:
    python src/experiment2_transfer_learning.py [--data_dir DATASET_PATH]
"""

import os, sys, argparse, json, time
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from utils import (ensure_dir, plot_training_history, plot_confusion_matrix,
                   compute_metrics, save_metrics, print_metrics, get_predictions,
                   enable_performance_mode, build_optimizer, build_loss,
                   standard_callbacks)

# --- Configuration ------------------------------------------------------------
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS_FE   = 20
EPOCHS_FT   = 15
AUTOTUNE    = tf.data.AUTOTUNE
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'experiment2')
CLASS_NAMES = ['with_mask', 'without_mask', 'mask_weared_incorrect']
NUM_CLASSES = len(CLASS_NAMES)


# --- Class weights ------------------------------------------------------------

def compute_class_weights(train_dir):
    counts = np.array([
        len(os.listdir(os.path.join(train_dir, c)))
        for c in CLASS_NAMES
        if os.path.isdir(os.path.join(train_dir, c))
    ], dtype=float)
    weights = counts.sum() / (NUM_CLASSES * counts)
    return {i: float(w) for i, w in enumerate(weights)}


# --- Data loading -------------------------------------------------------------

def load_datasets(data_dir, preprocess_fn):
    train_dir = os.path.join(data_dir, 'Train')
    val_dir   = os.path.join(data_dir, 'Validation')
    test_dir  = os.path.join(data_dir, 'Test')

    for d in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Expected directory not found: {d}")

    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
    ], name='augmentation')

    def aug_pp(x, y):  return preprocess_fn(aug(x, training=True)), y
    def pp(x, y):      return preprocess_fn(x), y

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        seed=42, label_mode='int', class_names=CLASS_NAMES
    ).map(aug_pp, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        seed=42, label_mode='int', class_names=CLASS_NAMES, shuffle=False
    ).map(pp, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        seed=42, label_mode='int', class_names=CLASS_NAMES, shuffle=False
    ).map(pp, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


# --- Classifier head ----------------------------------------------------------

def build_head(backbone_output, dropout=0.50):
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone_output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    return tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)


# --- Model builders -----------------------------------------------------------

def build_mobilenetv2_fe():
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base.trainable = False
    inputs  = tf.keras.Input(shape=IMG_SIZE + (3,))
    x       = base(inputs, training=False)
    outputs = build_head(x)
    return tf.keras.Model(inputs, outputs, name='MobileNetV2_FE'), base


def build_mobilenetv2_ft():
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base.trainable = False            # frozen during phase 1
    inputs  = tf.keras.Input(shape=IMG_SIZE + (3,))
    x       = base(inputs, training=False)
    outputs = build_head(x)
    return tf.keras.Model(inputs, outputs, name='MobileNetV2_FT'), base


def build_vgg16_fe():
    base = tf.keras.applications.VGG16(
        input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base.trainable = False
    inputs  = tf.keras.Input(shape=IMG_SIZE + (3,))
    x       = base(inputs, training=False)
    outputs = build_head(x, dropout=0.60)
    return tf.keras.Model(inputs, outputs, name='VGG16_FE'), base


# --- t-SNE visualisation ------------------------------------------------------

def plot_tsne(model, test_ds, class_names, save_path):
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        gap_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer('global_average_pooling2d').output)

        feats, labels = [], []
        for x, y in test_ds:
            feats.append(gap_model.predict(x, verbose=0))
            labels.extend(y.numpy().flatten().tolist())

        feats  = np.vstack(feats)
        labels = np.array(labels, dtype=int)

        print("  Running t-SNE on extracted features ...")
        emb = TSNE(n_components=2, random_state=42, perplexity=30,
                   n_iter=1000).fit_transform(feats)

        plt.figure(figsize=(8, 6))
        colors = ['steelblue', 'salmon', 'mediumseagreen']
        for i, cname in enumerate(class_names):
            mask = labels == i
            plt.scatter(emb[mask, 0], emb[mask, 1],
                        c=colors[i], label=cname, alpha=0.6, s=20)
        plt.legend(fontsize=9)
        plt.title('t-SNE of MobileNetV2 Extracted Features')
        plt.xlabel('Component 1'); plt.ylabel('Component 2')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Saved] t-SNE -> {save_path}")
    except Exception as e:
        print(f"  [Warning] t-SNE skipped: {e}")


# --- Training helper ----------------------------------------------------------

def fit(model, train_ds, val_ds, lr, epochs, ckpt_path, label, class_weights):
    model.compile(
        optimizer=build_optimizer(lr, weight_decay=1e-4),
        loss=build_loss(label_smoothing=0.1, num_classes=NUM_CLASSES),
        metrics=['accuracy']
    )
    callbacks = standard_callbacks(ckpt_path, patience_es=5, patience_lr=3, min_lr=1e-7)
    print(f"\n  Training {label}  (lr={lr}, max_epochs={epochs}) ...")
    t0   = time.time()
    hist = model.fit(train_ds, validation_data=val_ds,
                     epochs=epochs, callbacks=callbacks,
                     class_weight=class_weights, verbose=1)
    return hist, time.time() - t0


# --- Main ---------------------------------------------------------------------

def main(data_dir, fast=False):
    ensure_dir(RESULTS_DIR)
    enable_performance_mode(mixed_precision=fast, xla=True)
    print(f"\n{'='*60}")
    print("  EXPERIMENT 2 - Transfer Learning & Feature Extraction")
    print(f"  Dataset : {data_dir}")
    print(f"  Classes : {CLASS_NAMES}")
    print(f"  Results : {RESULTS_DIR}")
    print('='*60)

    train_dir     = os.path.join(data_dir, 'Train')
    class_weights = compute_class_weights(train_dir)
    print(f"  Class weights: { {CLASS_NAMES[i]: f'{w:.2f}' for i, w in class_weights.items()} }")

    all_results = {}
    mob_prep    = tf.keras.applications.mobilenet_v2.preprocess_input
    vgg_prep    = tf.keras.applications.vgg16.preprocess_input

    # -- Strategy 1: MobileNetV2 Feature Extraction ---------------------------
    print("\n-- Strategy 1: MobileNetV2 Feature Extraction (frozen) --")
    s1_dir = os.path.join(RESULTS_DIR, 'mobilenetv2_feature_extraction')
    ensure_dir(s1_dir)

    train_ds, val_ds, test_ds = load_datasets(data_dir, mob_prep)
    model_s1, _ = build_mobilenetv2_fe()
    hist_s1, t_s1 = fit(model_s1, train_ds, val_ds, lr=1e-3, epochs=EPOCHS_FE,
                        ckpt_path=os.path.join(s1_dir, 'best.keras'),
                        label='MobileNetV2_FE', class_weights=class_weights)

    plot_training_history(hist_s1.history, 'MobileNetV2 - Feature Extraction',
                          os.path.join(s1_dir, 'training_history.png'))
    y_true, y_pred = get_predictions(model_s1, test_ds)
    m_s1 = compute_metrics(y_true, y_pred, CLASS_NAMES)
    m_s1.update({'training_time_s': round(t_s1, 1), 'strategy': 'MobileNetV2_FE'})
    print_metrics(m_s1, 'MobileNetV2 Feature Extraction')
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES,
                          'MobileNetV2 - Feature Extraction',
                          os.path.join(s1_dir, 'confusion_matrix.png'))
    save_metrics(m_s1, os.path.join(s1_dir, 'metrics.json'))
    all_results['MobileNetV2_FeatureExtraction'] = {k: v for k, v in m_s1.items()
                                                     if k != 'classification_report'}
    plot_tsne(model_s1, test_ds, CLASS_NAMES, os.path.join(s1_dir, 'tsne_features.png'))

    # -- Strategy 2: MobileNetV2 Fine-Tuning ----------------------------------
    print("\n-- Strategy 2: MobileNetV2 Fine-Tuning (phase 1 -> phase 2) --")
    s2_dir = os.path.join(RESULTS_DIR, 'mobilenetv2_finetuned')
    ensure_dir(s2_dir)

    train_ds, val_ds, test_ds = load_datasets(data_dir, mob_prep)
    model_s2, base_s2 = build_mobilenetv2_ft()

    # Phase 1 - head only
    hist_fe, _ = fit(model_s2, train_ds, val_ds, lr=1e-3, epochs=EPOCHS_FE,
                     ckpt_path=os.path.join(s2_dir, 'phase1_best.keras'),
                     label='MobileNetV2_FT Phase 1', class_weights=class_weights)

    # Phase 2 - unfreeze top 30 layers
    base_s2.trainable = True
    unfreeze_from = len(base_s2.layers) - 30
    for layer in base_s2.layers[:unfreeze_from]:
        layer.trainable = False
    print(f"\n  Unfreezing top 30 backbone layers for fine-tuning ...")

    hist_ft, t_s2 = fit(model_s2, train_ds, val_ds, lr=1e-5, epochs=EPOCHS_FT,
                        ckpt_path=os.path.join(s2_dir, 'phase2_best.keras'),
                        label='MobileNetV2_FT Phase 2', class_weights=class_weights)

    combined = {k: hist_fe.history[k] + hist_ft.history[k]
                for k in ['accuracy', 'val_accuracy', 'loss', 'val_loss']}
    plot_training_history(combined, 'MobileNetV2 - Fine-Tuning (Phase 1 + 2)',
                          os.path.join(s2_dir, 'training_history.png'))

    y_true, y_pred = get_predictions(model_s2, test_ds)
    m_s2 = compute_metrics(y_true, y_pred, CLASS_NAMES)
    m_s2.update({'training_time_s': round(t_s2, 1), 'strategy': 'MobileNetV2_FT'})
    print_metrics(m_s2, 'MobileNetV2 Fine-Tuned')
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES,
                          'MobileNetV2 - Fine-Tuned',
                          os.path.join(s2_dir, 'confusion_matrix.png'))
    save_metrics(m_s2, os.path.join(s2_dir, 'metrics.json'))
    all_results['MobileNetV2_FineTuned'] = {k: v for k, v in m_s2.items()
                                             if k != 'classification_report'}

    # -- Strategy 3: VGG16 Feature Extraction ---------------------------------
    print("\n-- Strategy 3: VGG16 Feature Extraction (frozen) --")
    s3_dir = os.path.join(RESULTS_DIR, 'vgg16_feature_extraction')
    ensure_dir(s3_dir)

    train_ds3, val_ds3, test_ds3 = load_datasets(data_dir, vgg_prep)
    model_s3, _ = build_vgg16_fe()
    hist_s3, t_s3 = fit(model_s3, train_ds3, val_ds3, lr=1e-3, epochs=EPOCHS_FE,
                        ckpt_path=os.path.join(s3_dir, 'best.keras'),
                        label='VGG16_FE', class_weights=class_weights)

    plot_training_history(hist_s3.history, 'VGG16 - Feature Extraction',
                          os.path.join(s3_dir, 'training_history.png'))
    y_true3, y_pred3 = get_predictions(model_s3, test_ds3)
    m_s3 = compute_metrics(y_true3, y_pred3, CLASS_NAMES)
    m_s3.update({'training_time_s': round(t_s3, 1), 'strategy': 'VGG16_FE'})
    print_metrics(m_s3, 'VGG16 Feature Extraction')
    plot_confusion_matrix(y_true3, y_pred3, CLASS_NAMES,
                          'VGG16 - Feature Extraction',
                          os.path.join(s3_dir, 'confusion_matrix.png'))
    save_metrics(m_s3, os.path.join(s3_dir, 'metrics.json'))
    all_results['VGG16_FeatureExtraction'] = {k: v for k, v in m_s3.items()
                                               if k != 'classification_report'}

    # -- Summary chart ---------------------------------------------------------
    import matplotlib.pyplot as plt
    names      = list(all_results.keys())
    accuracies = [all_results[n]['accuracy'] for n in names]
    f1s        = [all_results[n]['f1_score']  for n in names]

    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, accuracies, w, label='Accuracy', color='steelblue', alpha=0.85)
    ax.bar(x + w/2, f1s,        w, label='F1 Score',  color='salmon',    alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim([0, 1.05]); ax.set_ylabel('Score')
    ax.set_title('Experiment 2 - Transfer Learning Comparison')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'experiment2_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    with open(os.path.join(RESULTS_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n  [Done] Results saved to {RESULTS_DIR}")
    print(f"\n{'Experiment 2 - Summary':^60}")
    print(f"{'Strategy':<35} {'Accuracy':>10} {'F1':>10}")
    print('-' * 58)
    for name, res in all_results.items():
        print(f"{name:<35} {res['accuracy']:>10.4f} {res['f1_score']:>10.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default=os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    parser.add_argument('--fast', action='store_true',
                        help='Enable mixed_float16 (GPU) + XLA JIT for faster training')
    args = parser.parse_args()
    main(os.path.abspath(args.data_dir), fast=args.fast)
