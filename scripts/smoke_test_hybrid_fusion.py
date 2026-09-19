"""Smoke test script for Gradient-Based Adaptive Hybrid Loss across 6 Fusion Methods.

Verifies model initialization, forward pass, gradient calculation, loss weighting adaptation,
and trainer step across:
- concat
- gated
- bilinear
- cross_attention
- decision_ensemble
- quantum_pqc
"""

import os
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.classifiers import MultimodalFusionClassifier
from src.models.losses import GradientAdaptiveHybridLoss
from src.evaluation.trainer import ExperimentTrainer, set_seed


def run_smoke_test():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Smoke Test] Running on device: {device}")

    fusion_methods = [
        "concat",
        "gated",
        "bilinear",
        "cross_attention",
        "decision_ensemble",
        "quantum_pqc",
    ]

    batch_size = 16
    seq_len = 10
    seq_dim = 13
    node_dim = 5
    num_classes = 4

    # Dummy dataset
    x_seq = torch.randn(64, seq_len, seq_dim)
    x_node = torch.randn(64, node_dim)
    y = torch.randint(0, num_classes, (64,))

    dataset = TensorDataset(x_seq, x_node, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    class_weights = torch.tensor([1.0, 2.0, 1.5, 3.0])

    all_passed = True

    for fusion in fusion_methods:
        print(f"\n--- Testing Fusion Method: {fusion} ---")
        try:
            model = MultimodalFusionClassifier(
                fusion_type=fusion,
                seq_input_dim=seq_dim,
                node_input_dim=node_dim,
                rel_edge_dim=0,
                num_classes=num_classes,
                embed_dim=32,
            )

            loss_fn = GradientAdaptiveHybridLoss(
                class_weights=class_weights,
                prior_ce=0.50,
                prior_focal=0.35,
                prior_wce=0.15,
            )

            trainer = ExperimentTrainer(
                model=model,
                criterion=loss_fn,
                learning_rate=1e-3,
                device=device,
            )

            loss_val = trainer.train_epoch(loader)
            print(f"  [SUCCESS] 1 Epoch Train Loss: {loss_val:.4f}")
            diag = loss_fn.latest_diagnostics
            print(f"  [Diagnostics] alpha: {diag['alpha']:.4f}, beta: {diag['beta']:.4f}, gamma: {diag['gamma']:.4f}")
            print(f"  [Diagnostics] g_ce: {diag['g_ce']:.4e}, g_focal: {diag['g_focal']:.4e}, g_wce: {diag['g_wce']:.4e}")
            print(f"  [Diagnostics] Clipping: {diag['clipping_occurred']}, Fallback: {diag['fallback_occurred']}")

        except Exception as e:
            print(f"  [FAILED] Fusion method {fusion} raised exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    if all_passed:
        print("\n[ALL SMOKE TESTS PASSED] GradientAdaptiveHybridLoss works with all 6 fusion methods!")
    else:
        print("\n[SMOKE TEST FAILED] One or more fusion methods failed.")
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_test()
