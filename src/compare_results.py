"""
compare_results.py
-----------------------------------------------------------------------------
Loads saved metrics from all three experiments and produces:
  * A combined bar chart (accuracy + F1) across all models
  * A summary table printed to stdout
  * A combined_results.json in the results/ root

Usage:
    python src/compare_results.py
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import ensure_dir, load_metrics

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'results')

# Map experiment folder -> human-readable experiment label
EXP_DIRS = {
    'experiment1': 'Exp1 - Custom CNN',
    'experiment2': 'Exp2 - Transfer Learning',
    'experiment3': 'Exp3 - SOTA',
}


def collect_all_results():
    """Walk results/ and load every metrics.json found."""
    all_results = {}
    for exp_folder, exp_label in EXP_DIRS.items():
        exp_path = os.path.join(RESULTS_ROOT, exp_folder)
        if not os.path.isdir(exp_path):
            print(f"  [Skip] {exp_path} not found")
            continue

        # Each sub-folder within an experiment folder is one model run
        for run_name in sorted(os.listdir(exp_path)):
            run_path = os.path.join(exp_path, run_name)
            metrics_path = os.path.join(run_path, 'metrics.json')
            if os.path.isfile(metrics_path):
                metrics = load_metrics(metrics_path)
                short_label = f"[{exp_label}]\n{run_name}"
                all_results[short_label] = metrics
                all_results[short_label]['experiment'] = exp_label

    return all_results


def plot_combined_comparison(all_results, save_path):
    """Single grouped bar chart comparing all models across experiments."""
    names      = list(all_results.keys())
    accuracies = [all_results[n].get('accuracy',  0) for n in names]
    f1s        = [all_results[n].get('f1_score',   0) for n in names]
    precisions = [all_results[n].get('precision',  0) for n in names]
    recalls    = [all_results[n].get('recall',     0) for n in names]

    x     = np.arange(len(names))
    width = 0.20
    fig, ax = plt.subplots(figsize=(max(14, len(names) * 1.6), 7))

    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy',  color='steelblue',     alpha=0.85)
    ax.bar(x - 0.5*width, f1s,        width, label='F1 Score',  color='salmon',         alpha=0.85)
    ax.bar(x + 0.5*width, precisions, width, label='Precision', color='mediumseagreen', alpha=0.85)
    ax.bar(x + 1.5*width, recalls,    width, label='Recall',    color='gold',           alpha=0.85, edgecolor='k')

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', ' ') for n in names],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylim([0, 1.10])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('All Experiments - Model Performance Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.4, label='95% line')

    # Annotate best accuracy bar
    best_idx = int(np.argmax(accuracies))
    ax.annotate(f'Best: {accuracies[best_idx]:.4f}',
                xy=(x[best_idx] - 1.5*width, accuracies[best_idx]),
                xytext=(x[best_idx], accuracies[best_idx] + 0.03),
                fontsize=8, color='darkblue', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkblue'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Combined comparison chart -> {save_path}")


def plot_experiment_heatmap(all_results, save_path):
    """Heatmap of metrics across all models."""
    import seaborn as sns

    names   = list(all_results.keys())
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    data    = np.array([[all_results[n].get(m, 0) for m in metrics] for n in names])

    fig, ax = plt.subplots(figsize=(8, max(6, len(names) * 0.55)))
    sns.heatmap(data, annot=True, fmt='.4f', cmap='YlGnBu',
                xticklabels=['Accuracy', 'F1', 'Precision', 'Recall'],
                yticklabels=[n.split('\n')[-1] for n in names],
                linewidths=0.5, vmin=0, vmax=1, ax=ax,
                annot_kws={"size": 8})
    ax.set_title('Performance Heatmap - All Models', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Heatmap -> {save_path}")


def plot_experiment_radar(all_results, save_path):
    """Radar chart comparing best model from each experiment."""
    import matplotlib.patches as mpatches

    # Pick best model per experiment (highest accuracy)
    best_per_exp = {}
    for name, res in all_results.items():
        exp = res.get('experiment', 'Unknown')
        if exp not in best_per_exp or res['accuracy'] > best_per_exp[exp][1]['accuracy']:
            best_per_exp[exp] = (name, res)

    categories = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    N          = len(categories)
    angles     = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles    += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors  = ['steelblue', 'salmon', 'mediumseagreen', 'purple']

    for i, (exp_label, (name, res)) in enumerate(best_per_exp.items()):
        values = [
            res.get('accuracy',  0),
            res.get('f1_score',  0),
            res.get('precision', 0),
            res.get('recall',    0),
        ]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i % len(colors)],
                label=f'{exp_label}\n({name.split(chr(10))[-1]})')
        ax.fill(angles, values, alpha=0.10, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=8)
    ax.set_title('Radar Chart - Best Model per Experiment',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Radar chart -> {save_path}")


def print_summary_table(all_results):
    print(f"\n{'='*85}")
    print(f"  COMBINED RESULTS - FACE MASK DETECTION")
    print(f"{'='*85}")
    print(f"{'Model':<45} {'Accuracy':>10} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print('-' * 85)
    for name, res in all_results.items():
        short = name.split('\n')[-1]
        print(f"{short:<45} "
              f"{res.get('accuracy', 0):>10.4f} "
              f"{res.get('f1_score', 0):>8.4f} "
              f"{res.get('precision', 0):>10.4f} "
              f"{res.get('recall', 0):>8.4f}")
    print('='*85)

    # Find overall best
    best_name  = max(all_results, key=lambda n: all_results[n].get('accuracy', 0))
    best_acc   = all_results[best_name]['accuracy']
    best_short = best_name.split('\n')[-1]
    print(f"\n  Best overall model: {best_short}  (Accuracy = {best_acc:.4f})")


def main():
    ensure_dir(RESULTS_ROOT)
    print(f"\n{'='*60}")
    print("  FINAL COMPARISON - All Experiments")
    print(f"  Results root: {RESULTS_ROOT}")
    print('='*60)

    all_results = collect_all_results()
    if not all_results:
        print("  [Error] No metrics.json files found.  "
              "Run experiments first.")
        sys.exit(1)

    print(f"\n  Found {len(all_results)} model results.")

    plot_combined_comparison(
        all_results,
        os.path.join(RESULTS_ROOT, 'combined_comparison.png'))

    plot_experiment_heatmap(
        all_results,
        os.path.join(RESULTS_ROOT, 'performance_heatmap.png'))

    try:
        plot_experiment_radar(
            all_results,
            os.path.join(RESULTS_ROOT, 'radar_chart.png'))
    except Exception as e:
        print(f"  [Warning] Radar chart skipped: {e}")

    # Save combined JSON
    combined_json = {name.split('\n')[-1]: res for name, res in all_results.items()}
    with open(os.path.join(RESULTS_ROOT, 'combined_results.json'), 'w') as f:
        json.dump(combined_json, f, indent=4)
    print(f"  [Saved] combined_results.json")

    print_summary_table(all_results)


if __name__ == '__main__':
    main()
