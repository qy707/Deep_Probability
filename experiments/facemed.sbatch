#!/usr/bin/env bash
# FaceMed reproduction check, one array task per (regularization, seed).
# Submit with:  sbatch experiments/facemed.sbatch
#SBATCH -J focus_facemed
#SBATCH -p cpu-gpu-v100
#SBATCH -A wang
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH --array=0-5
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail

# Cluster-specific settings. Override any of these from the submit line, e.g.
#   sbatch --export=ALL,PYTHON=/path/to/venv/bin/python experiments/facemed.sbatch
# The partition, account and node names in the #SBATCH block above are those of
# the cluster this was run on and will need changing elsewhere.
PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
DATA="$PROJECT/data/facemed"
# EPOCHS and RUNS_DIR are overridable, so the same script serves the
# 100-epoch and 200-epoch rounds.
EPOCHS="${EPOCHS:-100}"
RUNS="$PROJECT/${RUNS_DIR:-experiments/runs}"

# One entry per array index: "<regularization> <seed>".
SETTINGS=(
  "none 100" "time_dependent 100" "constant 100"
  "none 200" "time_dependent 200" "constant 200"
)
read -r REG SEED <<< "${SETTINGS[$SLURM_ARRAY_TASK_ID]}"
OUT="$RUNS/facemed_${REG}_seed${SEED}"

echo "host=$(hostname)  task=$SLURM_ARRAY_TASK_ID  reg=$REG  seed=$SEED  epochs=$EPOCHS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$PROJECT"

# split_seed is held at 42 across every task, so all realizations are trained
# and scored on the same split.
srun "$PYTHON" train.py \
    --scenario facemed --data_dir "$DATA" \
    --regularization "$REG" --seed "$SEED" --split_seed 42 \
    --epochs "$EPOCHS" --eval_interval 25 \
    --batch_size 256 --num_workers "$SLURM_CPUS_PER_TASK" \
    --n_samples 100 --max_parallel_samples 25 \
    --out_dir "$OUT"

# Final scoring on the held-out test split.
srun "$PYTHON" evaluate.py \
    --scenario facemed --data_dir "$DATA" \
    --checkpoint "$OUT/checkpoint_final.pt" --out_dir "$OUT/eval" \
    --split_seed 42 --n_samples 100 --max_parallel_samples 25 \
    --batch_size 256 --num_workers "$SLURM_CPUS_PER_TASK"

echo "done  reg=$REG  seed=$SEED"
