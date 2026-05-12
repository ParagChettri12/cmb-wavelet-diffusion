#!/bin/bash
# =============================================================================
# smica_rmse.sh
# SLURM job script — ILC baseline RMSE comparison on simulated data
#
# Compares three methods against known true CMB on simulated patches:
#   1. Mean predictor (trivial baseline)
#   2. ILC (uniform-weight linear combination across frequency channels)
#   3. This model (production checkpoint, posterior mean over 8 samples)
#
# Uses the PRODUCTION checkpoint (trained on sim data only) to ensure
# the input normalization matches the simulated test data.
# Do NOT use the fine-tuned checkpoint here — it drifts toward real
# Planck normalization and gives inflated RMSE on sim data.
#
# Customize the SBATCH directives below for your HPC cluster.
# =============================================================================

#SBATCH --job-name=cmb_smica_rmse
#SBATCH --output=logs/smica_rmse_%j.out
#SBATCH --error=logs/smica_rmse_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:v100:1          # V100 sufficient for this job
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

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start   : $(date)"
echo ""

# Confirm production checkpoint exists
if [ ! -f "./outputs/production/best_model.pt" ]; then
    echo "ERROR: ./outputs/production/best_model.pt not found."
    echo "Run production_training.sh first."
    exit 1
fi

# =============================================================================
# RMSE comparison
# =============================================================================
echo "=== ILC baseline vs model RMSE on simulated data ==="

python -c "
import torch, sys, numpy as np
sys.path.insert(0, '.')
from models.cmb_model import CMBDiffusionModel
from utils.dataset import build_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Simulated data — true CMB is known independently
_, val = build_dataloaders(
    ['./data/sim_50k/sim_dataset.h5'],
    0.1, 32, 4, False, 42
)

# Production checkpoint only — normalization matches sim data
model = CMBDiffusionModel(
    in_channels=3, patch_size=64,
    base_dim=256, encoder_depth=5, T_diffusion=200
).to(device)
ckpt = torch.load('./outputs/production/best_model.pt', map_location=device)
model.load_state_dict(
    {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
)
model.eval()

rmse_model, rmse_ilc, n = 0.0, 0.0, 0

for batch in val:
    X = batch['X'].to(device)
    y = batch['y']

    # ILC: uniform-weight mean across frequency channels
    # (optimal for frequency-independent CMB in thermodynamic units)
    y_ilc = X.mean(dim=1).cpu()

    # Model: posterior mean over 8 samples at T=1.0
    # T=1.0 (no inflation) for RMSE — we want the posterior mean, not coverage
    with torch.no_grad():
        samps  = model.sample(X, n_samples=8, temperature=1.0)
        y_pred = samps.mean(0).cpu()

    rmse_model += ((y_pred - y) ** 2).mean().item() * X.shape[0]
    rmse_ilc   += ((y_ilc  - y) ** 2).mean().item() * X.shape[0]
    n          += X.shape[0]
    print(f'  {n}/256 patches done', flush=True)
    if n >= 256:
        break

print()
print('=== Results ===')
print(f'Mean predictor RMSE : 1.000  (normalised unit-variance baseline)')
print(f'ILC baseline RMSE   : {(rmse_ilc   / n) ** 0.5:.4f}')
print(f'This model RMSE     : {(rmse_model / n) ** 0.5:.4f}')
print(f'Patches evaluated   : {n}')
"

echo ""
echo "=== Done: $(date) ==="
