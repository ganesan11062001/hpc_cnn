# 🤖 Autonomous HPC CNN Training Agent

> One command. Fully autonomous. Self-healing.

A production-grade ML pipeline that trains a CNN image classifier on HPC clusters — automatically handling environment setup, job submission, error recovery, hyperparameter optimization, and AI-driven re-optimization. No manual intervention required.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔁 **Autonomous** | Submits, monitors, heals, and re-runs without human input |
| 🩺 **Self-healing** | Detects errors in SLURM logs and patches them automatically |
| 🔬 **HPO** | Optuna finds best hyperparameters across 20 trials × 30 epochs |
| 🤖 **AI optimization** | Groq AI (Llama 3.3 70B) analyzes results and rewrites config |
| 💾 **Full logging** | Every epoch of every trial saved with provenance |
| 🔄 **Resumable** | Optuna SQLite study persists across runs |

---

## 📁 Project Structure

```
hpc_cnn/
│
├── 📂 src/                        # Core source code
│   ├── agent.py                   # Autonomous self-healing agent
│   ├── train_cnn.py               # CNN model + Optuna HPO training
│   └── generate_dataset.py        # Synthetic image dataset generator
│
├── 📂 configs/                    # Configuration files
│   └── config.yaml                # All tunable settings in one place
│
├── 📂 scripts/                    # HPC job scripts
│   ├── submit_job.sh              # SLURM training job
│   └── run_agent.sh               # SLURM agent job (recommended)
│
├── 📂 docs/                       # Documentation
│   ├── QUICKSTART.md              # Get running in 5 minutes
│   ├── HOW_IT_WORKS.md            # Architecture deep-dive
│   └── TROUBLESHOOTING.md         # Common errors and fixes
│
├── 📂 results/                    # Auto-generated training outputs
├── 📂 logs/                       # SLURM job logs
├── 📂 data/                       # Dataset (auto-generated)
│
├── .gitignore
└── README.md
```

---

## ⚡ Quickstart (5 minutes)

### 1. Clone
```bash
ssh g.murugan@login.discovery.neu.edu
git clone git@github.com:ganesan11062001/hpc_cnn.git
cd hpc_cnn
```

### 2. Get a free Groq API key
Sign up at **https://console.groq.com** (no credit card needed)
```bash
echo 'export GROQ_API_KEY="gsk_xxxxxxxxxxxx"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Run
```bash
sbatch scripts/run_agent.sh
tail -f agent_run.log
```

That's it. The agent handles everything else.

---

## 🏗️ How It Works

```
agent.py
   │
   ├── 1. Setup environment    → loads miniconda, creates conda env, installs packages
   ├── 2. Fix scripts          → auto-detects correct module names for this cluster
   ├── 3. Validate code        → patches known import errors in train_cnn.py
   ├── 4. Submit SLURM job     → sbatch submit_job.sh
   ├── 5. Monitor + heal       → polls every 30s, reads .err log, fixes and resubmits
   ├── 6. Parse results        → ranks all trials by val_acc, finds best epoch
   ├── 7. AI optimization      → Groq AI returns JSON params → auto-applied to code
   └── 8. Re-run optimized     → backs up results, reruns with new config
```

---

## 🩺 Auto-Heal Engine

The agent reads SLURM `.err` logs and applies fixes automatically:

| Error detected | Fix applied |
|---------------|-------------|
| `CondaError: Run conda init` | Rewrites activation using `source conda.sh` |
| `Unable to locate modulefile` | Scans `module avail`, picks correct version |
| `ImportError: JournalFileBackend` | Patches Optuna 4.x import in `train_cnn.py` |
| `CUDA driver mismatch` | Forces CPU fallback in `train_cnn.py` |
| `FileNotFoundError` | Creates missing directories |
| `OutOfMemoryError` | Reduces batch size, resubmits |
| Job `TIMEOUT` | Reduces `n_trials` + `max_epochs`, resubmits |

Max retries: **5**. Full audit trail in `agent_run.log`.

---

## ⚙️ Configuration

All settings live in `configs/config.yaml` — no need to edit Python files:

```yaml
training:
  n_trials: 20
  max_epochs: 30
  n_classes: 3

slurm:
  partition: gpu
  cpus: 4
  mem: 16G
  time: "04:00:00"

dataset:
  n_samples: 1500
  img_size: 32
```

---

## 📊 Output Structure

```
results/
├── trial_0_epochs.json      # Loss + accuracy for every epoch of trial 0
├── trial_N_epochs.json      # ... for all N trials
├── trial_N_best.pt          # Best model weights per trial
├── summary.json             # All trials ranked by val_acc
├── optuna.db                # Full Optuna study (resumable)
└── applied_optimizations.json  # Parameters Groq AI changed
```

### summary.json
```json
{
  "best_trial": 0,
  "best_epoch": 4,
  "best_val_acc": 1.0,
  "best_params": {
    "lr": 0.000137,
    "n_filters1": 16,
    "n_filters2": 16,
    "dropout": 0.468,
    "batch_size": 64,
    "optimizer": "SGD"
  },
  "all_trials_ranked": [...]
}
```

---

## 🖥️ SLURM Resources

| Resource | Training job | Agent job |
|----------|-------------|-----------|
| Partition | `gpu` | `short` |
| GPUs | 1 | 0 |
| CPUs | 4 | 1 |
| Memory | 16 GB | 4 GB |
| Wall time | 4 hours | 8 hours |

---

## 🔬 Adapt to Your Own Dataset

Replace the synthetic dataset with your own images:

```python
# In src/generate_dataset.py, replace generate_synthetic_images()
# with your own data loader that returns:
# X: numpy array of shape (N, H, W, 3) float32
# y: numpy array of shape (N,) int   — class labels
```

Update `configs/config.yaml`:
```yaml
dataset:
  n_samples: <your dataset size>
  img_size: <your image size>
  n_classes: <your number of classes>
```

---

## 📋 Requirements

- Northeastern Explorer HPC (or any SLURM cluster)
- Free Groq API key: https://console.groq.com
- Python 3.10+ (agent sets this up automatically)
- PyTorch, Optuna, NumPy (agent installs these automatically)

---

## 🗺️ Roadmap

- [ ] CIFAR-10 / real dataset support
- [ ] Multi-GPU training
- [ ] Weights & Biases integration
- [ ] Slack/email notifications on completion
- [ ] Support for Discovery cluster

---

## 👤 Author

**Ganesan Murugan**
MS Bioinformatics, Northeastern University
Computational Biology Co-op @ Solid Biosciences

[![GitHub](https://img.shields.io/badge/GitHub-ganesan11062001-black?logo=github)](https://github.com/ganesan11062001)
