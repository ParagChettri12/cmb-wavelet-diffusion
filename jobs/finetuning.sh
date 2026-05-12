#!/bin/bash
# =============================================================================
# finetuning.sh
# SLURM job script — Fine-tune on real Planck data + final evaluation
#
# Prerequisite: production_training.sh must have completed successfully
#               ./outputs/production/best_model.pt must exist
#
# Customize the SBATCH directives below for your HPC cluster.
#
# Estimated cost (Penn State Roar example):
#   V100 (M=22.68): 22.68 * 1 * 6/730 = ~0.19 credits
# =============================================================================

#SBATCH --job-name=cmb_finetune
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1          # V100 sufficient; A100 also fine
#SBATCH --partition=YOUR_PARTITION  # e.g. gpu, sla-prio, ica100, etc.
#SBATCH --account=YOUR_ACCOUNT      # your HPC allocation ID

# =============================================================================
# Environment setup
# =============================================================================

module purge
module load anaconda3
conda activate YOUR_ENV_NAME

cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start      : $(date)"
echo ""

# Confirm production checkpoint exists
if [ ! -f "./outputs/production/best_model.pt" ]; then
    echo "ERROR: ./outputs/production/best_model.pt not found."
    echo "Run production_training.sh first."
    exit 1
fi

# =============================================================================
# Step 1 — Download Planck maps (skip if already done)
# =============================================================================
if [ ! -f "./data/raw/planck_143GHz.fits" ]; then
    echo "=== Downloading Planck PR3 maps (~4 GB) ==="
    python download_data.py --output_dir ./data/raw
else
    echo "=== Planck raw maps found, skipping download ==="
fi

# =============================================================================
# Step 2 — Preprocess to HDF5 patches (skip if already done)
# =============================================================================
if [ ! -f "./data/processed/dataset.h5" ]; then
    echo ""
    echo "=== Preprocessing Planck maps ==="
    python preprocess.py \
        --raw_dir    ./data/raw \
        --out_dir    ./data/processed \
        --nside      128 \
        --patch_size 64 \
        --n_patches  8000
else
    echo "=== Processed dataset found, skipping preprocessing ==="
fi

# =============================================================================
# Step 3 — Fine-tune on mixed sim + real data
# =============================================================================
echo ""
echo "=== Fine-tuning on real Planck data ==="

python train.py \
    --data          ./data/sim_50k/sim_dataset.h5 \
                    ./data/processed/dataset.h5 \
    --base_dim      256 \
    --encoder_depth 5 \
    --T_diffusion   200 \
    --epochs        50 \
    --lr            1e-4 \
    --lambda_ps     0.5 \
    --batch_size    64 \
    --patience      6 \
    --save_every    10 \
    --num_workers   4 \
    --checkpoint    ./outputs/production/best_model.pt \
    --out_dir       ./outputs/final

# =============================================================================
# Step 4 — Conformal temperature calibration
# =============================================================================
echo ""
echo "=== Conformal calibration ==="
python conformal_calibrate.py

# =============================================================================
# Step 5 — Final evaluation (update --temperature with tau* from step 4)
# =============================================================================
echo ""
echo "=== Final evaluation ==="

python evaluate.py \
    --checkpoint    ./outputs/final/best_model.pt \
    --data          ./data/processed/dataset.h5 \
    --base_dim      256 \
    --encoder_depth 5 \
    --T_diffusion   200 \
    --n_samples     100 \
    --n_patches     512 \
    --temperature   5.0 \
    --out_dir       ./outputs/eval_final_pub

echo ""
echo "=== Done: $(date) ==="
echo "Results in ./outputs/eval_final_pub/metrics.json"
