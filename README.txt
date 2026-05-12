# CMB-ML: Probabilistic CMB Component Separation via Wavelet-Domain Diffusion

> **Toward Field-Level, Uncertainty-Aware CMB Reconstruction via Wavelet-Domain Conditional Diffusion Models**
> Parag Chettri — Penn State University, Department of Astronomy & Astrophysics

This repository implements a conditional denoising diffusion probabilistic model (DDPM) for Cosmic Microwave Background (CMB) component separation. Unlike deterministic methods (SMICA, NILC, SEVEM), this model produces a **distribution** over CMB realizations, enabling calibrated per-pixel uncertainty quantification.

**Key results on real Planck PR3 data (512 patches, 100 samples):**
| Metric | Value | Target |
|---|---|---|
| Coverage @ 68% | **0.699** | 0.680 |
| Coverage @ 95% | **0.917** | 0.950 |
| RMSE vs SMICA | **0.584** | — |
| RMSE vs true CMB (sim) | **0.891** | < ILC 0.921 |

---

## Requirements

```bash
pip install numpy scipy matplotlib astropy astropy-healpix torch torchvision tqdm h5py joblib
```

> **Note:** `astropy-healpix` is pure Python — no C extensions, works on all platforms including Windows and Apple Silicon.

---

## Project Structure

```
CMB-ML/
├── simulate.py              # Synthetic CMB patch generator (flat-sky FFT)
├── download_data.py         # Download Planck PR3 maps from NASA LAMBDA
├── preprocess.py            # Planck FITS -> HDF5 patch tensors
├── train.py                 # Training loop (ReduceLROnPlateau, AMP, HPC-optimised)
├── evaluate.py              # RMSE, power spectrum, calibration metrics + plots
├── conformal_calibrate.py   # Split conformal temperature calibration
├── models/
│   ├── __init__.py
│   └── cmb_model.py         # Full architecture: encoder + Haar DWT + DDPM
├── utils/
│   ├── __init__.py
│   └── dataset.py           # PyTorch Dataset / DataLoader (HPC/Lustre-safe)
└── jobs/                    # SLURM job scripts for HPC
    ├── production_training.sh
    └── finetuning.sh
```

---

## Full Pipeline

### Step 1 — Generate synthetic training data

50,000 synthetic multi-frequency CMB patches with known ground truth.
Physics calibrated to Planck 2018 best-fit LCDM (arXiv:1907.12875).

```bash
python simulate.py \
    --out_dir    ./data/sim_50k \
    --n_sims     50000 \
    --patch_size 64 \
    --res_arcmin 30
```

Output: `data/sim_50k/sim_dataset.h5` (~3.3 GB)

---

### Step 2 — Production training on simulated data

Requires GPU. For HPC: `sbatch jobs/production_training.sh`

```bash
python train.py \
    --data          ./data/sim_50k/sim_dataset.h5 \
    --base_dim      256 \
    --encoder_depth 5 \
    --T_diffusion   200 \
    --epochs        200 \
    --lr            1e-3 \
    --lambda_ps     0.5 \
    --batch_size    128 \
    --out_dir       ./outputs/production
```

Training notes:
- 15M parameters, A100-40GB recommended
- ~3-6 min/epoch on A100 with AMP float16
- Uses `ReduceLROnPlateau` (patience=8, factor=0.5)
- Checkpoints saved every 10 epochs to `outputs/production/`

---

### Step 3 — Download and preprocess real Planck data

Downloads ~4 GB of Planck PR3 FITS files from NASA LAMBDA.

```bash
python download_data.py --output_dir ./data/raw
```

Convert to HDF5 patch tensors:

```bash
python preprocess.py \
    --raw_dir    ./data/raw \
    --out_dir    ./data/processed \
    --nside      128 \
    --patch_size 64 \
    --n_patches  8000
```

Output: `data/processed/dataset.h5` (~524 MB, 8000 patches at 70/100/143 GHz)

---

### Step 4 — Fine-tune on real Planck data

Requires GPU. For HPC: `sbatch jobs/finetuning.sh`

```bash
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
    --checkpoint    ./outputs/production/best_model.pt \
    --out_dir       ./outputs/final
```

---

### Step 5 — Conformal temperature calibration

Finds the sampling temperature tau* that achieves calibrated coverage
on a held-out real Planck calibration set. Requires GPU.

```bash
python conformal_calibrate.py
```

Expected output:
```
tau=4.50  coverage@68% = 0.5683
tau=5.00  coverage@68% = 0.6706
*** tau* = 5.00 achieves target coverage ***
```

Use the reported `tau*` in all subsequent evaluation commands.

---

### Step 6 — Final evaluation

```bash
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
```

Produces in `outputs/eval_final_pub/`:
- `metrics.json` — RMSE, delta-Cl, coverage@68%, coverage@95%
- `reconstruction.png` — truth | mean prediction | uncertainty sigma | residual
- `power_spectrum.png` — |delta-Cl| vs angular scale
- `calibration.png` — reliability diagram

---

## Architecture

The model combines five differentiable components:

```
X (N, C, P, P)  -->  MultiFreqEncoder  -->  h (N, D, P/4, P/4)
                                              |
X  -->  Haar DWT  -->  SpectralAttention -->  AttnProj --> + h
                                              |
                              ForegroundLatentEncoder (z prior)
                                              |
                       Conditional DDPM (wavelet space, T=200 steps)
                                              |
                              T_CMB samples  (N_samples, N, P, P)
```

**Key design choices:**
- **Differentiable Haar DWT** via `F.conv2d` with fixed filter banks — keeps the full path in the autograd graph. PyWavelets breaks gradients via `.detach()`.
- **Normalized power spectrum loss** — divides by mean target power to prevent `L_PS` from dominating at initialization (raw MSE on Cl is O(10^4)).
- **Correct DDPM reverse step** — uses cumulative `alpha_bar` for posterior variance, not per-step `alpha` which collapses all samples to the same output.
- **ReduceLROnPlateau** — responds to actual validation plateaus; cosine annealing does not.

---

## HPC Usage (General)

Edit `YOUR_PARTITION`, `YOUR_ACCOUNT`, and `YOUR_ENV_NAME` in each job script, then:

```bash
sbatch jobs/production_training.sh   # A100, 12h
sbatch jobs/finetuning.sh            # V100, 6h (runs Steps 3-6 end-to-end)
```

**HPC notes:**
- `swmr=True` disabled in `dataset.py` — not supported on Lustre filesystems
- `persistent_workers=True` set to avoid per-epoch worker respawn overhead
- Dataset loaded entirely into RAM for zero disk I/O during training

---

## Citation

```bibtex
@article{chettri2026cmb,
  title   = {Toward Field-Level, Uncertainty-Aware CMB Reconstruction
             via Wavelet-Domain Conditional Diffusion Models},
  author  = {Chettri, Parag},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## References

- Planck Collaboration IV (2020) — SMICA/NILC/SEVEM/Commander. A&A 641, A4. arXiv:1807.06208
- Ho et al. (2020) — DDPM. NeurIPS 2020. arXiv:2006.11239
- Heurtel-Depeiges et al. (2023) — Diffusion for CMB/dust separation. arXiv:2310.16285
- Angelopoulos & Bates (2023) — Conformal Risk Control. ICLR 2023. arXiv:2208.02814
- Planck Collaboration V (2020) — CMB power spectra. A&A 641, A5. arXiv:1907.12875
- Planck Collaboration XI (2018) — Dust foregrounds. A&A 641, A11. arXiv:1801.04945