"""
experiment3_sota.py
-----------------------------------------------------------------------------
Experiment 3 - State-of-the-Art Models
-----------------------------------------------------------------------------
Dataset: andrewmvd/face-mask-detection (3 classes)

  Model A - EfficientNetB0  from scratch          (random init)
  Model B - EfficientNetB0  ImageNet pre-trained  (phase 1 head -> phase 2 fine-tune)
  Model C - Custom Vision Transformer             (patch-based ViT, no deps)

Usage:
    python src/experiment3_sota.py [--data_dir DATASET_PATH] [--skip_vit]
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
IMG_SIZE       = (224, 224)
IMG_SIZE_VIT   = (128, 128)   # custom ViT uses smaller resolution for speed
BATCH_SIZE     = 32
EPOCHS_SCRATCH = 30
EPOCHS_FT      = 20
AUTOTUNE       = tf.data.AUTOTUNE
RESULTS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'results', 'experiment3')
CLASS_NAMES    = ['with_mask', 'without_mask', 'mask_weared_incorrect']
NUM_CLASSES    = len(CLASS_NAMES)
VIT_HUB_URL    = "https://tfhub.dev/google/vit_b16/classification/1"


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

def load_datasets(data_dir, preprocess_fn, img_size=None):
    size = img_size or IMG_SIZE
    train_dir = os.path.join(data_dir, 'Train')
    val_dir   = os.path.join(data_dir, 'Validation')
    test_dir  = os.path.join(data_dir, 'Test')

    for d in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Not found: {d}")

    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomBrightness(0.10),
    ], name='augmentation')

    def aug_pp(x, y): return preprocess_fn(aug(x, training=True)), y
    def pp(x, y):     return preprocess_fn(x), y

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=size, batch_size=BATCH_SIZE,
        seed=42, label_mode='int', class_names=CLASS_NAMES
    ).map(aug_pp, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=size, batch_size=BATCH_SIZE,
        seed=42, label_mode='int', class_names=CLASS_NAMES, shuffle=False
    ).map(pp, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=size, batch_size=BATCH_SIZE,
        seed=42, label_mode='int', class_names=CLASS_NAMES, shuffle=False
    ).map(pp, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


# --- EfficientNetB0 -----------------------------------------------------------

def _eff_head(x):
    x = tf.keras.layers.GlobalAveragePooling2D()(x)       # flatten features
    x = tf.keras.layers.BatchNormalization()(x)           # stabilize
    x = tf.keras.layers.Dense(256, activation='relu')(x)  # learn patterns
    x = tf.keras.layers.Dropout(0.4)(x)                   # prevent overfitting
    return tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)  # 3-class output

def build_efficientnet_scratch():
    base = tf.keras.applications.EfficientNetB0(
        input_shape=IMG_SIZE + (3,), include_top=False, weights=None)
    base.trainable = True
    inputs  = tf.keras.Input(shape=IMG_SIZE + (3,))
    outputs = _eff_head(base(inputs, training=True))
    return tf.keras.Model(inputs, outputs, name='EfficientNetB0_Scratch')


def build_efficientnet_pretrained():
    base = tf.keras.applications.EfficientNetB0(
        input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base.trainable = False
    inputs  = tf.keras.Input(shape=IMG_SIZE + (3,))
    outputs = _eff_head(base(inputs, training=False))
    return tf.keras.Model(inputs, outputs, name='EfficientNetB0_Pretrained'), base


# --- Custom Vision Transformer ------------------------------------------------

def build_custom_vit(image_size=128, patch_size=16,
                     projection_dim=64, num_heads=4,
                     transformer_layers=4, mlp_head_units=None):
    """
    Lightweight Vision Transformer implementation (no external dependencies).
    Patches the image, encodes patch positions, applies stacked Transformer
    blocks, then classifies via a small MLP head.
    """
    if mlp_head_units is None:
        mlp_head_units = [128, 64]

    num_patches = (image_size // patch_size) ** 2
    mlp_dim     = projection_dim * 2

    def extract_patches(images):
        patches = tf.image.extract_patches(
            images,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID')
        batch = tf.shape(images)[0]
        return tf.reshape(patches, [batch, -1, patch_size * patch_size * 3])

    def transformer_block(x):
        # Multi-Head Self-Attention
        x1   = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=0.1)(x1, x1)
        x    = tf.keras.layers.Add()([x, attn])
        # Feed-Forward
        x2   = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ff   = tf.keras.layers.Dense(mlp_dim, activation='gelu')(x2)
        ff   = tf.keras.layers.Dropout(0.1)(ff)
        ff   = tf.keras.layers.Dense(projection_dim)(ff)
        ff   = tf.keras.layers.Dropout(0.1)(ff)
        return tf.keras.layers.Add()([x, ff])

    # -- Build model ----------------------------------------------------------
    inputs    = tf.keras.Input(shape=(image_size, image_size, 3))
    patches   = tf.keras.layers.Lambda(extract_patches, name='patch_extraction')(inputs)
    encoded   = tf.keras.layers.Dense(projection_dim)(patches)

    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_emb   = tf.keras.layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim)(positions)
    encoded   = encoded + pos_emb

    for _ in range(transformer_layers):
        encoded = transformer_block(encoded)

    rep = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded)
    rep = tf.keras.layers.GlobalAveragePooling1D()(rep)
    rep = tf.keras.layers.Dropout(0.3)(rep)

    for units in mlp_head_units:
        rep = tf.keras.layers.Dense(units, activation='gelu')(rep)
        rep = tf.keras.layers.Dropout(0.2)(rep)

    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(rep)
    return tf.keras.Model(inputs, outputs, name='CustomViT')


# --- TF-Hub ViT (optional) ----------------------------------------------------

def build_hub_vit(trainable=False):
    try:
        import tensorflow_hub as hub  # type: ignore[import-untyped]
        print(f"  Loading ViT-B16 from TF-Hub  (trainable={trainable}) ...")
        vit_layer = hub.KerasLayer(VIT_HUB_URL, trainable=trainable, name='vit_b16')
        inputs  = tf.keras.Input(shape=(224, 224, 3))
        vit_out = vit_layer(inputs)
        features = vit_out if not isinstance(vit_out, dict) else \
                   vit_out.get('pre_logits', list(vit_out.values())[0])
        x       = tf.keras.layers.Dense(256, activation='relu')(features)
        x       = tf.keras.layers.Dropout(0.4)(x)
        outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
        return tf.keras.Model(inputs, outputs,
                              name=f'ViT_B16_{"FT" if trainable else "Frozen"}'), True
    except Exception as e:
        print(f"  [Warning] TF-Hub ViT unavailable: {e}")
        return None, False


# --- Training helper ----------------------------------------------------------

def fit(model, train_ds, val_ds, lr, epochs, ckpt_path, label, class_weights):
    model.compile(
        optimizer=build_optimizer(lr, weight_decay=1e-4),
        loss=build_loss(label_smoothing=0.1, num_classes=NUM_CLASSES),
        metrics=['accuracy']
    )
    callbacks = standard_callbacks(ckpt_path, patience_es=7, patience_lr=3, min_lr=1e-7)
    print(f"\n  Training {label}  (lr={lr}, max_epochs={epochs}) ...")
    t0   = time.time()
    hist = model.fit(train_ds, validation_data=val_ds,
                     epochs=epochs, callbacks=callbacks,
                     class_weight=class_weights, verbose=1)
    return hist, time.time() - t0


# --- Main ---------------------------------------------------------------------

def main(data_dir, skip_vit=False, fast=False):
    ensure_dir(RESULTS_DIR)
    enable_performance_mode(mixed_precision=fast, xla=True)
    print(f"\n{'='*60}")
    print("  EXPERIMENT 3 - State-of-the-Art Models")
    print(f"  Dataset : {data_dir}")
    print(f"  Classes : {CLASS_NAMES}")
    print(f"  Results : {RESULTS_DIR}")
    print('='*60)

    train_dir     = os.path.join(data_dir, 'Train')
    class_weights = compute_class_weights(train_dir)
    print(f"  Class weights: { {CLASS_NAMES[i]: f'{w:.2f}' for i, w in class_weights.items()} }")

    eff_prep    = tf.keras.applications.efficientnet.preprocess_input
    all_results = {}

    # =======================================================================
    # Model A - EfficientNetB0 From Scratch
    # =======================================================================
    print("\n== Model A: EfficientNetB0 - From Scratch ==")
    dir_a = os.path.join(RESULTS_DIR, 'efficientnet_scratch')
    ensure_dir(dir_a)

    train_ds, val_ds, test_ds = load_datasets(data_dir, eff_prep)
    model_a = build_efficientnet_scratch()
    hist_a, t_a = fit(model_a, train_ds, val_ds, lr=1e-3, epochs=EPOCHS_SCRATCH,
                      ckpt_path=os.path.join(dir_a, 'best.keras'),
                      label='EfficientNetB0 (scratch)', class_weights=class_weights)

    plot_training_history(hist_a.history, 'EfficientNetB0 - From Scratch',
                          os.path.join(dir_a, 'training_history.png'))
    y_true_a, y_pred_a = get_predictions(model_a, test_ds)
    m_a = compute_metrics(y_true_a, y_pred_a, CLASS_NAMES)
    m_a.update({'training_time_s': round(t_a, 1), 'model': 'EfficientNetB0_Scratch'})
    print_metrics(m_a, 'EfficientNetB0 From Scratch')
    plot_confusion_matrix(y_true_a, y_pred_a, CLASS_NAMES,
                          'EfficientNetB0 - From Scratch',
                          os.path.join(dir_a, 'confusion_matrix.png'))
    save_metrics(m_a, os.path.join(dir_a, 'metrics.json'))
    all_results['EfficientNetB0_Scratch'] = {k: v for k, v in m_a.items()
                                              if k != 'classification_report'}

    # =======================================================================
    # Model B - EfficientNetB0 Pre-Trained
    # =======================================================================
    print("\n== Model B: EfficientNetB0 - ImageNet Pre-Trained ==")
    dir_b = os.path.join(RESULTS_DIR, 'efficientnet_pretrained')
    ensure_dir(dir_b)

    train_ds, val_ds, test_ds = load_datasets(data_dir, eff_prep)
    model_b, base_b = build_efficientnet_pretrained()

    # Phase 1 - classification head only
    hist_b1, _ = fit(model_b, train_ds, val_ds, lr=1e-3, epochs=10,
                     ckpt_path=os.path.join(dir_b, 'phase1_best.keras'),
                     label='EfficientNetB0 pre-trained [Phase 1]',
                     class_weights=class_weights)

    # Phase 2 - unfreeze top 20 backbone layers
    base_b.trainable = True
    for layer in base_b.layers[:len(base_b.layers) - 20]:
        layer.trainable = False
    print(f"\n  Unfreezing top 20 backbone layers ...")

    hist_b2, t_b = fit(model_b, train_ds, val_ds, lr=1e-5, epochs=EPOCHS_FT,
                       ckpt_path=os.path.join(dir_b, 'phase2_best.keras'),
                       label='EfficientNetB0 pre-trained [Phase 2]',
                       class_weights=class_weights)

    combined_b = {k: hist_b1.history[k] + hist_b2.history[k]
                  for k in ['accuracy', 'val_accuracy', 'loss', 'val_loss']}
    plot_training_history(combined_b, 'EfficientNetB0 - Pre-Trained + Fine-Tuned',
                          os.path.join(dir_b, 'training_history.png'))

    y_true_b, y_pred_b = get_predictions(model_b, test_ds)
    m_b = compute_metrics(y_true_b, y_pred_b, CLASS_NAMES)
    m_b.update({'training_time_s': round(t_b, 1), 'model': 'EfficientNetB0_Pretrained'})
    print_metrics(m_b, 'EfficientNetB0 Pre-Trained')
    plot_confusion_matrix(y_true_b, y_pred_b, CLASS_NAMES,
                          'EfficientNetB0 - Pre-Trained',
                          os.path.join(dir_b, 'confusion_matrix.png'))
    save_metrics(m_b, os.path.join(dir_b, 'metrics.json'))
    all_results['EfficientNetB0_Pretrained'] = {k: v for k, v in m_b.items()
                                                 if k != 'classification_report'}

    # =======================================================================
    # Model C - Custom Vision Transformer
    # =======================================================================
    print("\n== Model C: Custom Vision Transformer ==")
    dir_c = os.path.join(RESULTS_DIR, 'custom_vit')
    ensure_dir(dir_c)

    vit_prep = lambda x: tf.cast(
        tf.image.resize(x, IMG_SIZE_VIT), tf.float32) / 255.0
    train_vit, val_vit, test_vit = load_datasets(
        data_dir, vit_prep, IMG_SIZE_VIT)

    model_c = build_custom_vit(
        image_size=IMG_SIZE_VIT[0], patch_size=16,
        projection_dim=64, num_heads=4, transformer_layers=4)

    hist_c, t_c = fit(model_c, train_vit, val_vit, lr=1e-3,
                      epochs=EPOCHS_SCRATCH,
                      ckpt_path=os.path.join(dir_c, 'best.keras'),
                      label='CustomViT', class_weights=class_weights)

    plot_training_history(hist_c.history, 'Custom Vision Transformer',
                          os.path.join(dir_c, 'training_history.png'))
    y_true_c, y_pred_c = get_predictions(model_c, test_vit)
    m_c = compute_metrics(y_true_c, y_pred_c, CLASS_NAMES)
    m_c.update({'training_time_s': round(t_c, 1), 'model': 'CustomViT'})
    print_metrics(m_c, 'Custom Vision Transformer')
    plot_confusion_matrix(y_true_c, y_pred_c, CLASS_NAMES,
                          'Custom Vision Transformer',
                          os.path.join(dir_c, 'confusion_matrix.png'))
    save_metrics(m_c, os.path.join(dir_c, 'metrics.json'))
    all_results['CustomViT'] = {k: v for k, v in m_c.items()
                                if k != 'classification_report'}

    # =======================================================================
    # Model D - TF-Hub ViT-B16 (optional, requires tensorflow_hub + internet)
    # =======================================================================
    if not skip_vit:
        print("\n== Model D: ViT-B16 via TF-Hub (optional) ==")
        dir_d = os.path.join(RESULTS_DIR, 'vit_b16_hub')
        ensure_dir(dir_d)

        model_d, hub_ok = build_hub_vit(trainable=True)
        if hub_ok and model_d is not None:
            hub_prep = lambda x: tf.cast(x, tf.float32) / 255.0
            train_hub, val_hub, test_hub = load_datasets(
                data_dir, hub_prep, (224, 224))

            hist_d, t_d = fit(model_d, train_hub, val_hub, lr=1e-5,
                              epochs=EPOCHS_FT,
                              ckpt_path=os.path.join(dir_d, 'best.keras'),
                              label='ViT-B16 (TF-Hub)',
                              class_weights=class_weights)

            plot_training_history(hist_d.history, 'ViT-B16 - TF-Hub Fine-Tuned',
                                  os.path.join(dir_d, 'training_history.png'))
            y_true_d, y_pred_d = get_predictions(model_d, test_hub)
            m_d = compute_metrics(y_true_d, y_pred_d, CLASS_NAMES)
            m_d.update({'training_time_s': round(t_d, 1), 'model': 'ViT_B16_Hub'})
            print_metrics(m_d, 'ViT-B16 TF-Hub')
            plot_confusion_matrix(y_true_d, y_pred_d, CLASS_NAMES,
                                  'ViT-B16 - TF-Hub',
                                  os.path.join(dir_d, 'confusion_matrix.png'))
            save_metrics(m_d, os.path.join(dir_d, 'metrics.json'))
            all_results['ViT_B16_Hub'] = {k: v for k, v in m_d.items()
                                          if k != 'classification_report'}

    # -- EfficientNet scratch vs pre-trained comparison ------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels   = ['Scratch', 'ImageNet Pre-Trained']
    acc_vals = [all_results.get('EfficientNetB0_Scratch',    {}).get('accuracy', 0),
                all_results.get('EfficientNetB0_Pretrained', {}).get('accuracy', 0)]
    f1_vals  = [all_results.get('EfficientNetB0_Scratch',    {}).get('f1_score', 0),
                all_results.get('EfficientNetB0_Pretrained', {}).get('f1_score', 0)]

    for ax, vals, metric in zip(axes, [acc_vals, f1_vals], ['Accuracy', 'F1 Score']):
        bars = ax.bar(labels, vals, color=['steelblue', 'mediumseagreen'],
                      edgecolor='k', alpha=0.85)
        ax.set_ylim([0, 1.05]); ax.set_title(f'EfficientNetB0 - {metric}')
        ax.set_ylabel(metric); ax.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2., v + 0.005,
                    f'{v:.4f}', ha='center', fontweight='bold')

    plt.suptitle('Experiment 3 - Scratch vs Pre-Trained (EfficientNetB0)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'efficientnet_scratch_vs_pretrained.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # -- All-model comparison --------------------------------------------------
    names      = list(all_results.keys())
    accuracies = [all_results[n]['accuracy'] for n in names]
    f1s        = [all_results[n]['f1_score']  for n in names]

    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 2.5), 6))
    ax.bar(x - w/2, accuracies, w, label='Accuracy', color='steelblue', alpha=0.85)
    ax.bar(x + w/2, f1s,        w, label='F1 Score',  color='salmon',    alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=9)
    ax.set_ylim([0, 1.05]); ax.set_ylabel('Score')
    ax.set_title('Experiment 3 - SOTA Model Comparison')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'experiment3_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    with open(os.path.join(RESULTS_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n  [Done] Results saved to {RESULTS_DIR}")
    print(f"\n{'Experiment 3 - Summary':^65}")
    print(f"{'Model':<35} {'Accuracy':>10} {'F1':>10} {'Time (s)':>10}")
    print('-' * 68)
    for name, res in all_results.items():
        print(f"{name:<35} {res['accuracy']:>10.4f} {res['f1_score']:>10.4f} "
              f"{res.get('training_time_s', 0):>10.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default=os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    parser.add_argument('--skip_vit', action='store_true',
                        help='Skip TF-Hub ViT-B16 (still runs Custom ViT)')
    parser.add_argument('--fast', action='store_true',
                        help='Enable mixed_float16 (GPU) + XLA JIT for faster training')
    args = parser.parse_args()
    main(os.path.abspath(args.data_dir), skip_vit=args.skip_vit, fast=args.fast)
