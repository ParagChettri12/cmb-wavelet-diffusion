#!/bin/bash
# =============================================================================
# production_training.sh
# SLURM job script — Production training on simulated data
#
# Customize the SBATCH directives below for your HPC cluster:
#   --partition   : your cluster's GPU partition name
#   --account     : your allocation/project account ID
#   --gres        : GPU type available on your system (a100, v100, etc.)
#   --time        : wall time limit (200 epochs ~10-20h on A100)
#
# Estimated cost (Penn State Roar example):
#   A100 full (M=50.55): 50.55 * 1 * 12/730 = ~0.83 credits
# =============================================================================

#SBATCH --job-name=cmb_production
#SBATCH --output=logs/production_%j.out
#SBATCH --error=logs/production_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1          # change to gpu:v100:1 etc. as available
#SBATCH --partition=YOUR_PARTITION  # e.g. gpu, sla-prio, ica100, etc.
#SBATCH --account=YOUR_ACCOUNT      # your HPC allocation ID

# =============================================================================
# Environment setup — adapt to your cluster's module system
# Common options shown; uncomment what applies
# =============================================================================

module purge

# Option A: conda environment
module load anaconda3             # or: module load miniconda3
conda activate YOUR_ENV_NAME      # environment with torch, h5py, astropy etc.

# Option B: pip virtualenv (comment out Option A if using this)
# source /path/to/venv/bin/activate

# Option C: containers (comment out A/B if using this)
# module load singularity
# singularity exec --nv /path/to/pytorch.sif python ...

cd $SLURM_SUBMIT_DIR
mkdir -p logs

# =============================================================================
# Diagnostics
# =============================================================================
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "VRAM       : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null)"
echo "Start      : $(date)"
echo ""

# =============================================================================
# Step 1 — Generate 50k synthetic patches (skip if already done)
# =============================================================================
if [ ! -f "./data/sim_50k/sim_dataset.h5" ]; then
    echo "=== Generating 50k synthetic patches ==="
    python simulate.py \
        --out_dir    ./data/sim_50k \
        --n_sims     50000 \
        --patch_size 64 \
        --res_arcmin 30
else
    echo "=== sim_50k dataset found, skipping generation ==="
fi

# =============================================================================
# Step 2 — Production training
# =============================================================================
echo ""
echo "=== Production training ==="

python train.py \
    --data          ./data/sim_50k/sim_dataset.h5 \
    --base_dim      256 \
    --encoder_depth 5 \
    --T_diffusion   200 \
    --epochs        200 \
    --lr            1e-3 \
    --lambda_ps     0.5 \
    --batch_size    128 \
    --patience      8 \
    --save_every    20 \
    --num_workers   8 \
    --out_dir       ./outputs/production

echo ""
echo "=== Done: $(date) ==="
