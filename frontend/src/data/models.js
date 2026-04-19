// ─────────────────────────────────────────────────────────────────────────────
// Real numbers from results/experiment{1,2,3}/all_results.json
// Images copied from results/ to frontend/public/images/
// ─────────────────────────────────────────────────────────────────────────────

// All 9 models ranked by accuracy (desc) — global leaderboard
export const ALL_MODELS = [
  {
    rank: 1, key: 'MobileNetV2_FineTuned',
    name: 'MobileNetV2 Fine-Tuned', short: 'MNv2-FT',
    experiment: 'Transfer Learning', expNum: 2, expColor: '#22c55e',
    accuracy: 0.9137, f1: 0.9063, precision: 0.9110, recall: 0.9137,
    time: 107.8, color: '#3b82f6', isWinner: true,
    description: 'Two-phase fine-tuning: head-only first, then top layers unfrozen.',
    images: { history: 'images/exp2_mnv2ft_history.png', cm: 'images/exp2_mnv2ft_cm.png' },
  },
  {
    rank: 2, key: 'MobileNetV2_FeatureExtraction',
    name: 'MobileNetV2 Feature Extraction', short: 'MNv2-FE',
    experiment: 'Transfer Learning', expNum: 2, expColor: '#22c55e',
    accuracy: 0.9104, f1: 0.9026, precision: 0.9019, recall: 0.9104,
    time: 129.0, color: '#22c55e',
    description: 'Frozen MobileNetV2 backbone — only the classification head trained.',
    images: { history: 'images/exp2_mnv2fe_history.png', cm: 'images/exp2_mnv2fe_cm.png' },
  },
  {
    rank: 3, key: 'EfficientNetB0_Pretrained',
    name: 'EfficientNetB0 Pretrained', short: 'EffNet-PT',
    experiment: 'SOTA', expNum: 3, expColor: '#a855f7',
    accuracy: 0.9072, f1: 0.9040, precision: 0.9022, recall: 0.9072,
    time: 153.4, color: '#a855f7',
    description: 'EfficientNetB0 with ImageNet weights — fast and accurate.',
    images: { history: 'images/exp3_effp_history.png', cm: 'images/exp3_effp_cm.png' },
  },
  {
    rank: 4, key: 'EfficientNetB0_Scratch',
    name: 'EfficientNetB0 Scratch', short: 'EffNet-S',
    experiment: 'SOTA', expNum: 3, expColor: '#a855f7',
    accuracy: 0.9055, f1: 0.8903, precision: 0.9074, recall: 0.9055,
    time: 549.0, color: '#f97316',
    description: 'EfficientNetB0 from random init — 3.6× slower than pretrained.',
    images: { history: 'images/exp3_effs_history.png', cm: 'images/exp3_effs_cm.png' },
  },
  {
    rank: 5, key: 'VGG16_FeatureExtraction',
    name: 'VGG16 Feature Extraction', short: 'VGG16-FE',
    experiment: 'Transfer Learning', expNum: 2, expColor: '#22c55e',
    accuracy: 0.8925, f1: 0.8807, precision: 0.8839, recall: 0.8925,
    time: 139.0, color: '#f59e0b',
    description: 'Heavy VGG16 backbone, frozen — good but memory intensive.',
    images: { history: 'images/exp2_vgg16_history.png', cm: 'images/exp2_vgg16_cm.png' },
  },
  {
    rank: 6, key: 'Arch_C_Large_lr0.001',
    name: 'CNN-C Large (lr=0.001)', short: 'CNN-C',
    experiment: 'Custom CNN', expNum: 1, expColor: '#60a5fa',
    accuracy: 0.8909, f1: 0.8752, precision: 0.8609, recall: 0.8909,
    time: 85.9, params: 1373731, color: '#60a5fa',
    description: '4 conv blocks, 32→256 filters, BatchNorm + Dropout(0.5) + L2.',
    images: { history: 'images/exp1_archC_lr001_history.png', cm: 'images/exp1_archC_lr001_cm.png' },
  },
  {
    rank: 7, key: 'Arch_B_Medium_lr0.001',
    name: 'CNN-B Medium (lr=0.001)', short: 'CNN-B',
    experiment: 'Custom CNN', expNum: 1, expColor: '#60a5fa',
    accuracy: 0.8860, f1: 0.8714, precision: 0.8574, recall: 0.8860,
    time: 90.6, params: 322595, color: '#93c5fd',
    description: '3 conv blocks, 32→128 filters, BatchNorm + Dropout(0.4).',
    images: { history: 'images/exp1_archB_lr001_history.png', cm: 'images/exp1_archB_lr001_cm.png' },
  },
  {
    rank: 8, key: 'Arch_A_Small_lr0.001',
    name: 'CNN-A Small (lr=0.001)', short: 'CNN-A',
    experiment: 'Custom CNN', expNum: 1, expColor: '#60a5fa',
    accuracy: 0.8371, f1: 0.8129, precision: 0.8001, recall: 0.8371,
    time: 50.8, params: 28099, color: '#bfdbfe',
    description: '2 conv blocks, 32→64 filters — lightweight baseline, no regularization.',
    images: { history: 'images/exp1_archA_lr001_history.png', cm: 'images/exp1_archA_lr001_cm.png' },
  },
  {
    rank: 9, key: 'CustomViT',
    name: 'Custom Vision Transformer', short: 'ViT',
    experiment: 'SOTA', expNum: 3, expColor: '#a855f7',
    accuracy: 0.7915, f1: 0.6994, precision: 0.6265, recall: 0.7915,
    time: 77.3, color: '#ef4444',
    description: 'Patch-based ViT from scratch — needs far more data to converge.',
    images: { history: 'images/exp3_vit_history.png', cm: 'images/exp3_vit_cm.png' },
  },
]

// ─── Experiment-specific lists (used in detail tabs) ─────────────────────────

export const EXP1_MODELS = [
  {
    label: 'Arch-A · lr=0.001', key: 'Arch_A_Small_lr0.001',
    accuracy: '83.71%', f1: '81.29%', params: '28 K', time: '51 s', color: '#3b82f6',
    description: '2 conv blocks, 32→64 filters. No regularization — lightest baseline.',
    images: { history: 'images/exp1_archA_lr001_history.png', cm: 'images/exp1_archA_lr001_cm.png' },
  },
  {
    label: 'Arch-A · lr=0.0001', key: 'Arch_A_Small_lr0.0001',
    accuracy: '79.15%', f1: '69.94%', params: '28 K', time: '13 s', color: '#3b82f6',
    description: 'Same architecture, lower LR — underfits significantly.',
    images: { history: 'images/exp1_archA_lr0001_history.png', cm: 'images/exp1_archA_lr0001_cm.png' },
  },
  {
    label: 'Arch-B · lr=0.001', key: 'Arch_B_Medium_lr0.001',
    accuracy: '88.60%', f1: '87.14%', params: '323 K', time: '91 s', color: '#22c55e',
    description: '3 conv blocks, 32→128 filters, BatchNorm + Dropout(0.4).',
    images: { history: 'images/exp1_archB_lr001_history.png', cm: 'images/exp1_archB_lr001_cm.png' },
  },
  {
    label: 'Arch-B · lr=0.0001', key: 'Arch_B_Medium_lr0.0001',
    accuracy: '66.94%', f1: '69.57%', params: '323 K', time: '65 s', color: '#22c55e',
    description: 'LR too small for the added depth — accuracy collapses.',
    images: { history: 'images/exp1_archB_lr0001_history.png', cm: 'images/exp1_archB_lr0001_cm.png' },
  },
  {
    label: 'Arch-C · lr=0.001', key: 'Arch_C_Large_lr0.001',
    accuracy: '89.09%', f1: '87.52%', params: '1.37 M', time: '86 s', color: '#a855f7', badge: 'Best CNN',
    description: '4 conv blocks, 32→256 filters, BN + Dropout(0.5) + L2.',
    images: { history: 'images/exp1_archC_lr001_history.png', cm: 'images/exp1_archC_lr001_cm.png' },
  },
  {
    label: 'Arch-C · lr=0.0001', key: 'Arch_C_Large_lr0.0001',
    accuracy: '74.59%', f1: '75.91%', params: '1.37 M', time: '77 s', color: '#a855f7',
    description: 'Aggressive regularization + slow LR = too conservative to learn.',
    images: { history: 'images/exp1_archC_lr0001_history.png', cm: 'images/exp1_archC_lr0001_cm.png' },
  },
]

export const EXP2_MODELS = [
  {
    label: 'MobileNetV2 Feature Extraction', key: 'MobileNetV2_FeatureExtraction',
    accuracy: '91.04%', f1: '90.26%', time: '129 s', color: '#22c55e',
    description: 'Frozen MobileNetV2 backbone — only the classification head trained.',
    images: { history: 'images/exp2_mnv2fe_history.png', cm: 'images/exp2_mnv2fe_cm.png' },
  },
  {
    label: 'MobileNetV2 Fine-Tuned', key: 'MobileNetV2_FineTuned',
    accuracy: '91.37%', f1: '90.63%', time: '108 s', color: '#3b82f6', badge: 'Best Overall',
    description: 'Phase 1: head only. Phase 2: top layers unfrozen. Best model overall.',
    images: { history: 'images/exp2_mnv2ft_history.png', cm: 'images/exp2_mnv2ft_cm.png' },
  },
  {
    label: 'VGG16 Feature Extraction', key: 'VGG16_FeatureExtraction',
    accuracy: '89.25%', f1: '88.07%', time: '139 s', color: '#f59e0b',
    description: 'Frozen VGG16 — heavier than MobileNetV2 but lower accuracy.',
    images: { history: 'images/exp2_vgg16_history.png', cm: 'images/exp2_vgg16_cm.png' },
  },
]

export const EXP3_MODELS = [
  {
    label: 'EfficientNetB0 Scratch', key: 'EfficientNetB0_Scratch',
    accuracy: '90.55%', f1: '89.03%', time: '549 s', color: '#f97316',
    description: 'Random init — strong result but 3.6× slower than pretrained.',
    images: { history: 'images/exp3_effs_history.png', cm: 'images/exp3_effs_cm.png' },
  },
  {
    label: 'EfficientNetB0 Pretrained', key: 'EfficientNetB0_Pretrained',
    accuracy: '90.72%', f1: '90.40%', time: '153 s', color: '#a855f7', badge: 'Best SOTA',
    description: 'ImageNet init — same accuracy as scratch but 3.6× faster to train.',
    images: { history: 'images/exp3_effp_history.png', cm: 'images/exp3_effp_cm.png' },
  },
  {
    label: 'Custom Vision Transformer', key: 'CustomViT',
    accuracy: '79.15%', f1: '69.94%', time: '77 s', color: '#ef4444',
    description: 'Patch-based ViT from scratch — data-hungry, needs 10× more images.',
    images: { history: 'images/exp3_vit_history.png', cm: 'images/exp3_vit_cm.png' },
  },
]
