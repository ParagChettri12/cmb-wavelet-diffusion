# conformal_calibrate.py
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.cmb_model import CMBDiffusionModel
from utils.dataset import build_dataloaders

ALPHA      = 0.32
N_SAMPLES  = 32     # reduced from 64
N_CAL      = 200    # reduced from 500
BATCH_SIZE = 16     # sample this many patches at a time
TAU_GRID   = np.linspace(1.0, 8.0, 15)  # 15 points instead of 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CMBDiffusionModel(in_channels=3, patch_size=64,
            base_dim=256, encoder_depth=5, T_diffusion=200).to(device)
ckpt  = torch.load('./outputs/final/best_model.pt', map_location=device)
model.load_state_dict({k.replace('_orig_mod.',''):v
                       for k,v in ckpt['model'].items()})
model.eval()

_, cal_loader = build_dataloaders(
    ['./data/processed/dataset.h5'],
    val_fraction=0.3, batch_size=32, num_workers=4, augment=False)

X_cal, y_cal = [], []
for b in cal_loader:
    X_cal.append(b['X']); y_cal.append(b['y'])
    if sum(x.shape[0] for x in X_cal) >= N_CAL: break
X_cal = torch.cat(X_cal)[:N_CAL]   # keep on CPU
y_cal = torch.cat(y_cal)[:N_CAL].numpy()

print(f'Calibration set: {len(X_cal)} real Planck patches')
print(f'Sweeping {len(TAU_GRID)} temperatures, {N_SAMPLES} samples each\n')

best_tau = TAU_GRID[-1]
for tau in TAU_GRID:
    # Sample in small batches to avoid OOM
    all_samps = []
    for i in range(0, len(X_cal), BATCH_SIZE):
        xb = X_cal[i:i+BATCH_SIZE].to(device)
        with torch.no_grad():
            s = model.sample(xb, n_samples=N_SAMPLES,
                             temperature=float(tau)).cpu().numpy()
        all_samps.append(s)   # (N_SAMPLES, batch, P, P)

    samps    = np.concatenate(all_samps, axis=1)  # (N_SAMPLES, N_CAL, P, P)
    lo       = np.quantile(samps, ALPHA/2,   axis=0)
    hi       = np.quantile(samps, 1-ALPHA/2, axis=0)
    coverage = float(((y_cal >= lo) & (y_cal <= hi)).mean())

    print(f'  tau={tau:.2f}  coverage@{100*(1-ALPHA):.0f}% = {coverage:.4f}',
          flush=True)

    if coverage >= (1 - ALPHA) and best_tau == TAU_GRID[-1]:
        best_tau = tau
        print(f'  *** tau* = {best_tau:.2f} ***')

print(f'\nConformal tau* = {best_tau:.2f}')
print(f'Use: --temperature {best_tau:.2f}')