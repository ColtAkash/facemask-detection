"""
run_all.py
-----------------------------------------------------------------------------
Master script that runs all three experiments sequentially and then
produces the cross-experiment comparison report.

Usage examples:
    # Run all experiments (uses ./dataset/ by default)
    python run_all.py

    # Custom dataset path
    python run_all.py --data_dir C:/path/to/Face-Mask-Dataset

    # Skip the ViT experiments in Exp 3 (faster, no internet needed)
    python run_all.py --skip_vit

    # Run only a specific experiment
    python run_all.py --only 1
    python run_all.py --only 2
    python run_all.py --only 3
    python run_all.py --only compare
"""

import os, sys, argparse, time

# Ensure src/ is on the path
SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, SRC_DIR)


def banner(text):
    width = 64
    print(f"\n{'#'*width}")
    print(f"#  {text:<{width-4}}#")
    print(f"{'#'*width}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run Face Mask Detection deep learning experiments')
    parser.add_argument('--data_dir',
                        default=os.path.join(os.path.dirname(__file__), 'dataset'),
                        help='Root directory of the dataset (must contain '
                             'Train/, Validation/, Test/)')
    parser.add_argument('--skip_vit', action='store_true',
                        help='Skip Vision Transformer in Experiment 3')
    parser.add_argument('--only',
                        choices=['1', '2', '3', 'compare'],
                        help='Run only the specified experiment')
    parser.add_argument('--fast', action='store_true',
                        help='Enable mixed_float16 (GPU) + XLA JIT for faster training')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    if not os.path.isdir(data_dir):
        print(f"\n[ERROR] Dataset directory not found: {data_dir}")
        print("Please download the dataset first (see README.md -> Dataset Setup).")
        sys.exit(1)

    # Verify expected structure
    for split in ('Train', 'Validation', 'Test'):
        split_path = os.path.join(data_dir, split)
        if not os.path.isdir(split_path):
            print(f"[ERROR] Missing split folder: {split_path}")
            print("Expected dataset structure:")
            print("  dataset/")
            print("  +-- Train/")
            print("  |   +-- with_mask/")
            print("  |   +-- without_mask/")
            print("  |   +-- mask_weared_incorrect/")
            print("  +-- Validation/")
            print("  +-- Test/")
            sys.exit(1)

    total_start = time.time()
    run_all     = args.only is None

    # -- Experiment 1 ----------------------------------------------------------
    if run_all or args.only == '1':
        banner("EXPERIMENT 1 - Custom CNN Architectures (Supervised Learning)")
        from experiment1_custom_cnn import main as exp1_main
        t0 = time.time()
        exp1_main(data_dir, fast=args.fast)
        print(f"\n  [Exp 1 completed in {time.time()-t0:.1f}s]")

    # -- Experiment 2 ----------------------------------------------------------
    if run_all or args.only == '2':
        banner("EXPERIMENT 2 - Transfer Learning (Unsupervised Feature Extraction)")
        from experiment2_transfer_learning import main as exp2_main
        t0 = time.time()
        exp2_main(data_dir, fast=args.fast)
        print(f"\n  [Exp 2 completed in {time.time()-t0:.1f}s]")

    # -- Experiment 3 ----------------------------------------------------------
    if run_all or args.only == '3':
        banner("EXPERIMENT 3 - State-of-the-Art Models (EfficientNet + ViT)")
        from experiment3_sota import main as exp3_main
        t0 = time.time()
        exp3_main(data_dir, skip_vit=args.skip_vit, fast=args.fast)
        print(f"\n  [Exp 3 completed in {time.time()-t0:.1f}s]")

    # -- Final comparison ------------------------------------------------------
    if run_all or args.only == 'compare':
        banner("FINAL COMPARISON - All Experiments")
        from compare_results import main as compare_main
        compare_main()

    if run_all:
        total = time.time() - total_start
        print(f"\n{'='*64}")
        print(f"  ALL EXPERIMENTS COMPLETE  -  Total time: {total:.1f}s  ({total/60:.1f} min)")
        print(f"  Results saved to: {os.path.join(os.path.dirname(__file__), 'results')}")
        print('='*64)


if __name__ == '__main__':
    main()
