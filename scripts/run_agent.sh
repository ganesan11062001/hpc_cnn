#!/bin/bash
#SBATCH --job-name=hpc_agent
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=agent_run.log
#SBATCH --error=agent_run.log

cd $SLURM_SUBMIT_DIR
source /shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh
conda activate cnn_hpo
python agent.py
