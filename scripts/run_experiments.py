"""
Main experiment benchmark runner script.
Loads processed .pt benchmark samples and evaluates all 6 multimodal fusion strategies
(Concat, Gated Adaptive, Bilinear, Cross-Attention, Decision Ensemble, Quantum PQC).
"""

import argparse
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.benchmark_harness import BenchmarkEvaluator


def main():
    parser = argparse.ArgumentParser(description="6-Method Multimodal Fusion Benchmark Harness")
    parser.add_argument("--processed-path", type=str, default=r"d:\Final Year Implementation\data\processed\nf_cse_cic_ids2018_v2_synchronized_samples.pt")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per model")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    if not os.path.exists(args.processed_path):
        print(f"Error: Processed dataset not found at '{args.processed_path}'. Please run scripts/run_pipeline.py first!")
        sys.exit(1)

    print(f"--> Loading synchronized benchmark samples from: {args.processed_path}")
    samples = torch.load(args.processed_path, weights_only=False)
    print(f"--> Loaded {len(samples)} benchmark samples.")

    evaluator = BenchmarkEvaluator(embed_dim=64, num_classes=2)
    results = evaluator.run_fusion_benchmark(
        samples=samples,
        train_split=0.7,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 80)
    print(f"{'FUSION STRATEGY':<22} | {'F1-SCORE':<10} | {'ACCURACY':<10} | {'RECALL':<10} | {'FPR':<10} | {'LOSS':<10}")
    print("=" * 80)
    for fusion_name, res in results.items():
        print(
            f"{fusion_name.upper():<22} | {res['f1_score']:<10} | {res['accuracy']:<10} | {res['recall']:<10} | {res['false_positive_rate']:<10} | {res['final_train_loss']:<10}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
