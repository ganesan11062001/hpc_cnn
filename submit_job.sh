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

echo "============================================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURMD_NODENAME"
echo "Start time    : $(date)"
echo "Working dir   : $SLURM_SUBMIT_DIR"
echo "============================================================"

cd $SLURM_SUBMIT_DIR
mkdir -p logs results data

module load cuda/12.1.1 2>/dev/null || true
source /shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh
conda activate cnn_hpo

echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"

echo ""
echo ">>> Step 1: Generating dataset..."
python generate_dataset.py

echo ""
echo ">>> Step 2: Training CNN with Optuna HPO..."
python train_cnn.py \
    --data_path  data/dataset.npz \
    --output_dir results/ \
    --n_trials 20 \
    --max_epochs 15

echo ""
echo ">>> Step 3: Summary"
python -c "
import json, sys
try:
    with open('results/summary.json') as f:
        s = json.load(f)
    print(f'Best Trial   : {s[chr(98)+chr(101)+chr(115)+chr(116)+chr(95)+chr(116)+chr(114)+chr(105)+chr(97)+chr(108)]}')
    print(f'Best Epoch   : {s[chr(98)+chr(101)+chr(115)+chr(116)+chr(95)+chr(101)+chr(112)+chr(111)+chr(99)+chr(104)]}')
    print(f'Best Val Acc : {s[chr(98)+chr(101)+chr(115)+chr(116)+chr(95)+chr(118)+chr(97)+chr(108)+chr(95)+chr(97)+chr(99)+chr(99)]:.5f}')
except Exception as e:
    print(f'Summary error: {e}', file=sys.stderr)
    sys.exit(1)
"

echo "============================================================"
echo "Job complete: $(date)"
echo "============================================================"
