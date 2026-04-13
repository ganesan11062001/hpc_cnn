# CNN HPC AI Agent — Explorer @ Northeastern

## What this does
1. **Generates** a synthetic 3-class image dataset (1500 samples, 32×32 RGB)
2. **Submits** a SLURM job to Explorer with GPU resources
3. **Trains** a SmallCNN with **Optuna HPO** (20 trials × 30 epochs each)
4. **Saves all results** — per-epoch logs for every trial + ranked summary
5. **Picks the best epoch** across all trials
6. **Asks Claude** for optimization recommendations based on your results

---

## Files
```
hpc_cnn/
├── generate_dataset.py   # Synthetic dataset generator
├── train_cnn.py          # CNN + Optuna training script
├── submit_job.sh         # SLURM script for Explorer
├── agent.py              # AI agent: submit → monitor → optimize
└── README.md
```

---

## Setup on Explorer

```bash
# 1. Copy files to Explorer
scp -r hpc_cnn/ murugan.g@login.discovery.neu.edu:~/cnn_hpo/
ssh murugan.g@login.discovery.neu.edu
cd ~/cnn_hpo

# 2. Activate your conda env (or create one)
conda activate af3   # reuse your existing env
pip install optuna torch torchvision

# 3. Set your Anthropic API key (for Claude optimizations)
export ANTHROPIC_API_KEY="your-key-here"
```

---

## Usage

### Option A: Full pipeline (recommended)
```bash
python agent.py --all
```
Generates data → submits SLURM → monitors until done → parses results → asks Claude for optimizations.

### Option B: Manual steps
```bash
# Step 1: Submit
python agent.py --submit
# Output: "Job submitted: 12345"

# Step 2: Monitor
python agent.py --monitor 12345

# Step 3: Get optimizations
python agent.py --optimize
```

### Option C: Just run SLURM directly
```bash
sbatch submit_job.sh
```

---

## Output Structure
```
results/
├── trial_0_epochs.json          # All epochs for trial 0
├── trial_1_epochs.json          # All epochs for trial 1
├── ...
├── trial_0_best.pt              # Best model weights for each trial
├── optuna_journal.log           # Optuna study journal
├── summary.json                 # Ranked results — best trial + epoch
└── optimization_recommendations.txt  # Claude's suggestions
```

### summary.json structure
```json
{
  "best_trial": 7,
  "best_epoch": 22,
  "best_val_acc": 0.94133,
  "best_params": {"lr": 0.00123, "n_filters1": 64, ...},
  "all_trials_ranked": [...]
}
```

---

## Tuning the search
Edit `train_cnn.py` to adjust Optuna search ranges:
- `n_trials` — how many HPO trials to run
- `max_epochs` — max epochs per trial
- `suggest_float("lr", ...)` — learning rate range
- `suggest_categorical("n_filters1", ...)` — filter sizes

---

## SLURM config (submit_job.sh)
| Resource | Default |
|----------|---------|
| GPU | 1× (any) |
| CPUs | 4 |
| Memory | 16 GB |
| Wall time | 4 hours |
| Partition | gpu |

Adjust `#SBATCH` lines for your queue limits on Explorer.
