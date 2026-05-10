"""
Sparse vs Standard Neural Network — Research Experiment
Hypothesis: Sparse activation improves computational efficiency
with minimal loss in accuracy in small neural networks.

ONE variable changed: sparsity level (top-k activation)
Everything else: identical architecture, dataset, training loop, seed.
"""

import time
import json
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ─── Reproducibility — CRITICAL for controlled experiment ─────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    # Architecture — IDENTICAL for both models
    input_dim: int = 784        # MNIST 28x28
    hidden_dims: List[int] = None
    output_dim: int = 10
    # Training — IDENTICAL for both models
    epochs: int = 50
    batch_size: int = 128
    lr: float = 0.001
    weight_decay: float = 1e-4
    # ONE variable that changes
    sparsity: float = 0.0       # 0.0 = standard, 0.3 = top-30% active
    model_name: str = "standard"
    # Energy simulation
    ops_per_multiply: float = 1.0   # relative FLOPs cost

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]


# ─── Sparse Activation Layer ──────────────────────────────────────────────────

class TopKSparse(nn.Module):
    """
    Brain-like sparse activation: only top-k% of neurons fire.
    Everything else is zeroed out — mimicking neural inhibition.

    At sparsity=0.3, only 30% of neurons are active per forward pass.
    This is the ONLY difference between Model A and Model B.
    """

    def __init__(self, sparsity: float = 0.3):
        super().__init__()
        self.sparsity = sparsity  # fraction of neurons ACTIVE (top-k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity >= 1.0:
            return F.relu(x)  # All neurons active = standard ReLU

        # Apply ReLU first (no negative activations)
        x = F.relu(x)

        # Keep only top-k% of activations per sample
        k = max(1, int(x.shape[-1] * self.sparsity))
        topk_vals, _ = torch.topk(x, k, dim=-1)
        threshold = topk_vals[:, -1:].detach()  # kth largest value
        mask = (x >= threshold).float()
        return x * mask

    def extra_repr(self):
        return f"sparsity={self.sparsity} (top {self.sparsity*100:.0f}% active)"


# ─── Neural Network Architecture ──────────────────────────────────────────────

class ResearchNet(nn.Module):
    """
    Identical architecture for both models.
    Only difference: activation function (ReLU vs TopKSparse).
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.activation_counts = []  # Track sparsity during inference

        layers = []
        in_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))

            # THE ONE VARIABLE: activation function
            if config.sparsity < 1.0 and config.sparsity > 0.0:
                layers.append(TopKSparse(sparsity=config.sparsity))
            else:
                layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Dropout(0.2))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, config.output_dim))
        self.network = nn.Sequential(*layers)

        # Xavier init — identical for both
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)

    def count_active_neurons(self, x: torch.Tensor) -> float:
        """Measure actual sparsity during a forward pass."""
        x = x.view(x.size(0), -1)
        total_neurons = 0
        active_neurons = 0

        hook_outputs = []
        hooks = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                hook_outputs.append(output.detach())
            return hook

        # Register hooks on activation layers
        for module in self.network:
            if isinstance(module, (TopKSparse, nn.ReLU)):
                hooks.append(module.register_forward_hook(make_hook(len(hooks))))

        with torch.no_grad():
            self.network(x)

        for h in hooks:
            h.remove()

        for out in hook_outputs:
            total_neurons += out.numel()
            active_neurons += (out > 0).sum().item()

        return active_neurons / total_neurons if total_neurons > 0 else 0.0

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops(self, active_fraction: float = 1.0) -> int:
        """Estimate FLOPs per forward pass, accounting for sparsity."""
        flops = 0
        in_dim = self.config.input_dim
        for hidden_dim in self.config.hidden_dims:
            # Each linear layer: 2 * in * out FLOPs (multiply + add)
            flops += 2 * in_dim * hidden_dim * active_fraction
            in_dim = hidden_dim
        flops += 2 * in_dim * self.config.output_dim
        return int(flops)


# ─── Energy Simulation ────────────────────────────────────────────────────────

class EnergySimulator:
    """
    Simulates energy consumption based on active neuron count.
    Based on: active multiply-accumulate ops ∝ energy usage.

    Reference: Sparse networks use ~sparsity% of energy vs dense.
    This is a simplified model — real hardware varies.
    """
    JOULES_PER_FLOP = 1e-12  # ~1 pJ per FLOP (rough estimate, modern GPU)

    def __init__(self):
        self.total_flops = 0.0
        self.total_energy_j = 0.0

    def log_forward_pass(self, flops: int, batch_size: int = 1):
        self.total_flops += flops * batch_size
        self.total_energy_j += flops * batch_size * self.JOULES_PER_FLOP

    @property
    def total_energy_uj(self):
        return self.total_energy_j * 1e6  # Convert to microjoules

    def reset(self):
        self.total_flops = 0.0
        self.total_energy_j = 0.0


# ─── Data Loading ─────────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    # Fixed seed for identical data order — controlled experiment
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=g, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0
    )
    return train_loader, test_loader


# ─── Training & Evaluation ────────────────────────────────────────────────────

@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    epoch_time_s: float
    energy_uj: float
    actual_sparsity: float  # measured, not set
    flops: int


def train_epoch(
    model: ResearchNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    energy_sim: EnergySimulator,
    device: str,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Log energy
        active_frac = model.config.sparsity if model.config.sparsity > 0 else 1.0
        flops = model.estimate_flops(active_fraction=active_frac)
        energy_sim.log_forward_pass(flops, batch_size=images.size(0))

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(
    model: ResearchNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    sparsity_measurements = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

        # Sample sparsity measurement (not every batch — expensive)
        if len(sparsity_measurements) < 5:
            sparsity_measurements.append(
                model.count_active_neurons(images[:16])
            )

    avg_sparsity = np.mean(sparsity_measurements) if sparsity_measurements else 0.0
    return total_loss / len(loader), correct / total, avg_sparsity


# ─── Full Experiment Runner ────────────────────────────────────────────────────

def run_experiment(config: ExperimentConfig, device: str = "cpu") -> List[EpochMetrics]:
    print(f"\n{'='*60}")
    print(f"  Running: {config.model_name.upper()}")
    print(f"  Sparsity: {config.sparsity} ({config.sparsity*100:.0f}% neurons active)")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    model = ResearchNet(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    energy_sim = EnergySimulator()

    train_loader, test_loader = get_dataloaders(config.batch_size)

    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Est. FLOPs/pass: {model.estimate_flops():,}")

    metrics_history = []
    epochs_to_90 = None

    for epoch in range(1, config.epochs + 1):
        energy_sim.reset()
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, energy_sim, device
        )
        val_loss, val_acc, actual_sparsity = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        epoch_time = time.time() - t0
        active_frac = config.sparsity if config.sparsity > 0 else 1.0
        flops = model.estimate_flops(active_fraction=active_frac)

        m = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            epoch_time_s=epoch_time,
            energy_uj=energy_sim.total_energy_uj,
            actual_sparsity=actual_sparsity,
            flops=flops,
        )
        metrics_history.append(m)

        # Track epochs to 90%
        if epochs_to_90 is None and val_acc >= 0.90:
            epochs_to_90 = epoch
            print(f"  ⚡ 90% accuracy reached at epoch {epoch}!")

        print(
            f"  Epoch {epoch:03d}/{config.epochs} | "
            f"Val Acc={val_acc:.4f} | "
            f"Active={actual_sparsity:.2%} | "
            f"Energy={energy_sim.total_energy_uj:.1f}μJ | "
            f"{epoch_time:.1f}s"
        )

    print(f"\n  Final Val Accuracy: {metrics_history[-1].val_acc:.4f}")
    print(f"  Epochs to 90%: {epochs_to_90 or 'Not reached'}")
    print(f"  Total Energy: {sum(m.energy_uj for m in metrics_history):.1f}μJ")

    return metrics_history


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_full_study(sparsity_levels: List[float] = None, epochs: int = 50):
    """Run experiments across multiple sparsity levels."""
    if sparsity_levels is None:
        sparsity_levels = [1.0, 0.5, 0.3, 0.2, 0.1]

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = {}
    for sparsity in sparsity_levels:
        name = "standard" if sparsity >= 1.0 else f"sparse_{int(sparsity*100)}pct"
        config = ExperimentConfig(
            sparsity=sparsity,
            model_name=name,
            epochs=epochs,
        )
        metrics = run_experiment(config, device=device)
        results[name] = [asdict(m) for m in metrics]

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to results/experiment_results.json")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--sparsity", nargs="+", type=float,
        default=[1.0, 0.5, 0.3, 0.2, 0.1],
        help="Sparsity levels to test (1.0=standard, 0.3=top30%% active)"
    )
    args = parser.parse_args()
    run_full_study(sparsity_levels=args.sparsity, epochs=args.epochs)
