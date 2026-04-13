#!/bin/bash
#SBATCH --job-name=cnn_hpo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/cnn_hpo_%j.out
#SBATCH --error=logs/cnn_hpo_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=murugan.g@northeastern.edu

# ── Explorer HPC CNN Training with Optuna HPO ─────────────────────────────────
echo "============================================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURMD_NODENAME"
echo "Start time    : $(date)"
echo "Working dir   : $SLURM_SUBMIT_DIR"
echo "============================================================"

# Go to project directory (edit this path)
cd $SLURM_SUBMIT_DIR

# Create output dirs
mkdir -p logs results data

# ── Module loading (Explorer-specific) ───────────────────────────────────────
module load cuda/11.8
module load python/3.10

# ── Activate conda env (adjust env name as needed) ───────────────────────────
# If you have a dedicated env:
source activate /home/murugan.g/.conda/envs/af3
# OR use: conda activate cnn_hpo

# ── Install deps if not already present ──────────────────────────────────────
pip install torch torchvision optuna --quiet 2>/dev/null || true

# ── Step 1: Generate synthetic dataset ───────────────────────────────────────
echo ""
echo ">>> Step 1: Generating synthetic dataset..."
python generate_dataset.py

# ── Step 2: Train CNN with Optuna HPO ────────────────────────────────────────
echo ""
echo ">>> Step 2: Training CNN (Optuna HPO, 20 trials, 30 epochs each)..."
python train_cnn.py \
    --data_path  data/dataset.npz \
    --output_dir results/ \
    --n_trials   20 \
    --max_epochs 30

# ── Step 3: Print summary ─────────────────────────────────────────────────────
echo ""
echo ">>> Step 3: Results Summary"
echo "------------------------------------------------------------"
python -c "
import json
with open('results/summary.json') as f:
    s = json.load(f)
print(f\"Best Trial   : {s['best_trial']}\")
print(f\"Best Epoch   : {s['best_epoch']}\")
print(f\"Best Val Acc : {s['best_val_acc']:.5f}\")
print(f\"Best Params  : {s['best_params']}\")
print()
print('All trials ranked:')
for i, t in enumerate(s['all_trials_ranked'][:5], 1):
    print(f\"  #{i}  Trial {t['trial']}  val_acc={t['best_val_acc']:.4f}  epoch={t['best_epoch']}  params={t['params']}\")
"

echo ""
echo "============================================================"
echo "Job complete: $(date)"
echo "Results in   : results/"
echo "============================================================"
