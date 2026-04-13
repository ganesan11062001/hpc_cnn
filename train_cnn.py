"""
train_cnn.py
CNN training with Optuna HPO on Explorer HPC.
Saves per-epoch metrics for ALL trials and picks the best epoch/trial combo.

Usage:
  python train_cnn.py --n_trials 20 --output_dir results/
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna

import logging

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─── Model ───────────────────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    def __init__(self, n_filters1, n_filters2, dropout, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, n_filters1, 3, padding=1), nn.BatchNorm2d(n_filters1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(n_filters1, n_filters2, 3, padding=1), nn.BatchNorm2d(n_filters2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(n_filters2, n_filters2 * 2, 3, padding=1), nn.BatchNorm2d(n_filters2 * 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(n_filters2 * 2 * 16, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.net(x)

# ─── Data Loader ─────────────────────────────────────────────────────────────
def load_data(data_path, batch_size):
    data = np.load(data_path)
    def make_loader(X, y, shuffle):
        Xt = torch.tensor(X).permute(0, 3, 1, 2).float()
        yt = torch.tensor(y).long()
        return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return (
        make_loader(data["X_train"], data["y_train"], True),
        make_loader(data["X_val"],   data["y_val"],   False),
        make_loader(data["X_test"],  data["y_test"],  False),
    )

# ─── Single Trial Training ────────────────────────────────────────────────────
def train_trial(trial, args, device):
    # Suggest hyperparameters
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    n_filters1 = trial.suggest_categorical("n_filters1", [16, 32, 64])
    n_filters2 = trial.suggest_categorical("n_filters2", [32, 64, 128])
    dropout    = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    weight_decay   = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    train_loader, val_loader, _ = load_data(args.data_path, batch_size)
    model = SmallCNN(n_filters1, n_filters2, dropout).to(device)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    criterion = nn.CrossEntropyLoss()

    # Per-epoch log for this trial
    trial_log = []
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.max_epochs + 1):
        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
        scheduler.step()
        train_acc = correct / total
        train_loss /= total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * len(yb)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += len(yb)
        val_acc = val_correct / val_total
        val_loss /= val_total

        epoch_result = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc, 5),
            "val_loss":   round(val_loss, 5),
            "val_acc":    round(val_acc, 5),
        }
        trial_log.append(epoch_result)
        log.info(f"Trial {trial.number} | Epoch {epoch:3d} | "
                 f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.5f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"trial_{trial.number}_best.pt"))

        # Optuna pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            log.info(f"Trial {trial.number} pruned at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

    # Save full epoch log for this trial
    trial_result_path = os.path.join(args.output_dir, f"trial_{trial.number}_epochs.json")
    with open(trial_result_path, "w") as f:
        json.dump({
            "trial": trial.number,
            "params": trial.params,
            "epoch_log": trial_log,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
        }, f, indent=2)

    return best_val_acc

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="data/dataset.npz")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--n_trials",   type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Optuna study with pruning
    storage = f"sqlite:///{os.path.join(args.output_dir, 'optuna.db')}"
    study = optuna.create_study(
        study_name="cnn_hpo",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: train_trial(trial, args, device),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # ── Aggregate all results ──────────────────────────────────────────────
    all_results = []
    for fname in os.listdir(args.output_dir):
        if fname.endswith("_epochs.json"):
            with open(os.path.join(args.output_dir, fname)) as f:
                all_results.append(json.load(f))

    # Sort by best_val_acc descending
    all_results.sort(key=lambda x: x["best_val_acc"], reverse=True)

    best = all_results[0]
    summary = {
        "best_trial":     best["trial"],
        "best_epoch":     best["best_epoch"],
        "best_val_acc":   best["best_val_acc"],
        "best_params":    best["params"],
        "all_trials_ranked": [
            {
                "trial": r["trial"],
                "best_val_acc": r["best_val_acc"],
                "best_epoch": r["best_epoch"],
                "params": r["params"],
            }
            for r in all_results
        ],
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info(f"BEST TRIAL   : {best['trial']}")
    log.info(f"BEST EPOCH   : {best['best_epoch']}")
    log.info(f"BEST VAL ACC : {best['best_val_acc']:.5f}")
    log.info(f"BEST PARAMS  : {best['params']}")
    log.info(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
