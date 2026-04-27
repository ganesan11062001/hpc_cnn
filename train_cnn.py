"""
train_cnn.py
CNN training with full parameter support (up to 20 params).
Parameters are passed via --params_json from the agent.
No hardcoded hyperparameters — everything comes from the LLM via agent.py.

Usage:
  python src/train_cnn.py --params_json '{"lr": 0.001, ...}' --run_id 1
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TRAIN] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─── Activation helper ────────────────────────────────────────────────────────
def get_activation(name):
    return {
        "relu":       nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "gelu":       nn.GELU(),
        "selu":       nn.SELU(),
        "elu":        nn.ELU(),
    }.get(name, nn.ReLU())

# ─── Model ────────────────────────────────────────────────────────────────────
class FlexCNN(nn.Module):
    """
    Flexible CNN whose architecture is fully controlled by params.
    Supports 1-4 conv layers, variable filters, activations, pooling, etc.
    """
    def __init__(self, p, n_classes=3):
        super().__init__()
        act  = p.get("activation", "relu")
        pool = p.get("pooling_type", "max")
        use_bn = p.get("use_batchnorm", True)

        def pool_layer():
            return nn.MaxPool2d(2) if pool == "max" else nn.AvgPool2d(2)

        def conv_block(in_ch, out_ch):
            ks = p.get("kernel_size", 3)
            pad = ks // 2
            layers = [nn.Conv2d(in_ch, out_ch, ks, padding=pad)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(get_activation(act))
            return layers

        f1 = p.get("n_filters1", 32)
        f2 = p.get("n_filters2", 64)
        f3 = p.get("n_filters3", 128)
        f4 = p.get("n_filters4", 0)   # 0 = skip this layer
        n_conv = p.get("n_conv_layers", 3)

        layers = []
        layers += conv_block(3, f1);  layers.append(pool_layer())
        if n_conv >= 2:
            layers += conv_block(f1, f2); layers.append(pool_layer())
        if n_conv >= 3:
            layers += conv_block(f2, f3)
        if n_conv >= 4 and f4 > 0:
            layers += conv_block(f3, f4)

        last_filters = [f1, f2, f3, f4][n_conv - 1]
        layers.append(nn.AdaptiveAvgPool2d(4))
        layers.append(nn.Flatten())

        fc1 = p.get("fc_hidden1", 128)
        fc2 = p.get("fc_hidden2", 0)   # 0 = skip second FC
        dropout  = p.get("dropout", 0.3)
        dropout2 = p.get("dropout2", 0.0)

        layers.append(nn.Linear(last_filters * 16, fc1))
        layers.append(get_activation(act))
        layers.append(nn.Dropout(dropout))

        if fc2 > 0:
            layers.append(nn.Linear(fc1, fc2))
            layers.append(get_activation(act))
            if dropout2 > 0:
                layers.append(nn.Dropout(dropout2))
            layers.append(nn.Linear(fc2, n_classes))
        else:
            layers.append(nn.Linear(fc1, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ─── Data loader ──────────────────────────────────────────────────────────────
def load_data(data_path, batch_size):
    data = np.load(data_path)
    def make_loader(X, y, shuffle):
        Xt = torch.tensor(X).permute(0, 3, 1, 2).float()
        yt = torch.tensor(y).long()
        return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                          shuffle=shuffle, num_workers=2, pin_memory=True)
    return (
        make_loader(data["X_train"], data["y_train"], True),
        make_loader(data["X_val"],   data["y_val"],   False),
        make_loader(data["X_test"],  data["y_test"],  False),
    )

# ─── Training ─────────────────────────────────────────────────────────────────
def train(p, run_id, data_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device  : {device}")
    log.info(f"Run {run_id} params: {json.dumps(p, indent=2)}")

    batch_size  = p.get("batch_size", 64)
    max_epochs  = p.get("max_epochs", 30)
    lr          = p.get("lr", 0.001)
    opt_name    = p.get("optimizer", "Adam")
    wd          = p.get("weight_decay", 0.0001)
    grad_clip   = p.get("gradient_clip", 0.0)
    label_smooth = p.get("label_smoothing", 0.0)
    scheduler_type = p.get("scheduler", "cosine")
    warmup      = p.get("warmup_epochs", 0)

    train_loader, val_loader, _ = load_data(data_path, batch_size)
    n_classes = len(np.unique(np.load(data_path)["y_train"]))
    model = FlexCNN(p, n_classes).to(device)

    # Count model params
    n_params = sum(x.numel() for x in model.parameters())
    log.info(f"Model params: {n_params:,}")

    # Optimizer
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "SGD":
        momentum = p.get("momentum", 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Scheduler
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_type == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=max_epochs,
            steps_per_epoch=len(train_loader)
        )
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    epoch_log    = []
    best_val_acc = 0.0
    best_epoch   = 0

    for epoch in range(1, max_epochs + 1):
        # Warmup
        if warmup > 0 and epoch <= warmup:
            for g in optimizer.param_groups:
                g["lr"] = lr * epoch / warmup

        # ── Train ──
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler_type == "onecycle" and scheduler:
                scheduler.step()
            train_loss += loss.item() * len(yb)
            correct    += (out.argmax(1) == yb).sum().item()
            total      += len(yb)

        if scheduler and scheduler_type != "onecycle":
            if not (warmup > 0 and epoch <= warmup):
                scheduler.step()

        train_acc  = correct / total
        train_loss /= total

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out  = model(Xb)
                loss = criterion(out, yb)
                val_loss    += loss.item() * len(yb)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total   += len(yb)
        val_acc  = val_correct / val_total
        val_loss /= val_total

        log.info(f"Epoch {epoch:3d}/{max_epochs} | "
                 f"train={train_acc:.4f} val={val_acc:.4f} loss={val_loss:.5f}")

        epoch_log.append({
            "epoch": epoch, "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5), "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(),
                       os.path.join(output_dir, f"run_{run_id}_best.pt"))

    # ── Save result ────────────────────────────────────────────────────────────
    result = {
        "run_id": run_id, "params": p,
        "n_model_params": n_params,
        "epoch_log": epoch_log,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
    }
    result_path = os.path.join(output_dir, f"run_{run_id}_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # ── Update summary ─────────────────────────────────────────────────────────
    summary_path = os.path.join(output_dir, "summary.json")
    all_runs = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            all_runs = json.load(f).get("all_runs", [])

    all_runs.append({"run_id": run_id, "best_val_acc": best_val_acc,
                     "best_epoch": best_epoch, "params": p,
                     "n_model_params": n_params})
    all_runs.sort(key=lambda x: x["best_val_acc"], reverse=True)
    best = all_runs[0]
    with open(summary_path, "w") as f:
        json.dump({"best_run": best["run_id"], "best_epoch": best["best_epoch"],
                   "best_val_acc": best["best_val_acc"], "best_params": best["params"],
                   "all_runs": all_runs}, f, indent=2)

    log.info(f"Run {run_id} done | best_val_acc={best_val_acc:.5f} epoch={best_epoch}")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_json", required=True, help="JSON string of all hyperparameters")
    parser.add_argument("--run_id",      type=int, default=1)
    parser.add_argument("--data_path",   default="data/dataset.npz")
    parser.add_argument("--output_dir",  default="results")
    args = parser.parse_args()

    p = json.loads(args.params_json)
    train(p, args.run_id, args.data_path, args.output_dir)

if __name__ == "__main__":
    main()