"""Smoke test script for Single-Modality Ablation Classifiers.

Verifies:
1. TemporalOnlyClassifier initialization, forward pass, reference parameters, and parameter counts.
2. SpatialRelationalOnlyClassifier initialization, forward pass, reference parameters, and parameter counts.
3. 1-epoch training loop with GradientAdaptiveHybridLoss for both architectures.
"""

import os
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.single_modality import TemporalOnlyClassifier, SpatialRelationalOnlyClassifier
from src.models.losses import GradientAdaptiveHybridLoss
from src.evaluation.trainer import ExperimentTrainer, set_seed


def run_smoke_test():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Smoke Test] Running Modality Ablation Smoke Test on device: {device}")

    batch_size = 16
    seq_len = 10
    seq_dim = 13
    node_dim = 13
    num_classes = 4

    # Dummy dataset
    x_seq = torch.randn(64, seq_len, seq_dim)
    x_node = torch.randn(64, node_dim)
    y = torch.randint(0, num_classes, (64,))

    dataset = TensorDataset(x_seq, x_node, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    class_weights = torch.tensor([1.0, 2.0, 1.5, 3.0])

    all_passed = True

    # Test 1: Temporal-Only Classifier
    print("\n--- Testing Architecture: Temporal-Only Classifier ---")
    try:
        model_temp = TemporalOnlyClassifier(
            temp_input_dim=seq_dim,
            embed_dim=64,
            num_classes=num_classes,
        )
        param_info = model_temp.get_parameter_count()
        print(f"  [Parameter Count] Total: {param_info['total_parameters']:,}, Trainable: {param_info['trainable_parameters']:,}")

        loss_fn = GradientAdaptiveHybridLoss(class_weights=class_weights)
        trainer = ExperimentTrainer(model=model_temp, criterion=loss_fn, device=device)

        loss_val = trainer.train_epoch(loader)
        print(f"  [SUCCESS] 1 Epoch Train Loss: {loss_val:.4f}")
        diag = loss_fn.latest_diagnostics
        print(f"  [Diagnostics] alpha: {diag['alpha']:.4f}, beta: {diag['beta']:.4f}, gamma: {diag['gamma']:.4f}")
    except Exception as e:
        print(f"  [FAILED] Temporal-Only Classifier raised exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 2: Spatial/Relational-Only Classifier
    print("\n--- Testing Architecture: Spatial/Relational-Only Classifier ---")
    try:
        model_rel = SpatialRelationalOnlyClassifier(
            rel_node_dim=node_dim,
            rel_edge_dim=0,
            embed_dim=64,
            num_classes=num_classes,
        )
        param_info = model_rel.get_parameter_count()
        print(f"  [Parameter Count] Total: {param_info['total_parameters']:,}, Trainable: {param_info['trainable_parameters']:,}")

        loss_fn = GradientAdaptiveHybridLoss(class_weights=class_weights)
        trainer = ExperimentTrainer(model=model_rel, criterion=loss_fn, device=device)

        loss_val = trainer.train_epoch(loader)
        print(f"  [SUCCESS] 1 Epoch Train Loss: {loss_val:.4f}")
        diag = loss_fn.latest_diagnostics
        print(f"  [Diagnostics] alpha: {diag['alpha']:.4f}, beta: {diag['beta']:.4f}, gamma: {diag['gamma']:.4f}")
    except Exception as e:
        print(f"  [FAILED] Spatial/Relational-Only Classifier raised exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    if all_passed:
        print("\n[ALL SMOKE TESTS PASSED] Modality Ablation Classifiers ready for execution!")
    else:
        print("\n[SMOKE TEST FAILED] One or more modality ablation tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_test()
