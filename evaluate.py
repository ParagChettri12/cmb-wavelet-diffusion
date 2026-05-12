"""
evaluate.py
───────────
Evaluates a trained CMBDiffusionModel using all three metrics from the paper.
No healpy / astropy-healpix.

    7.1  Field-level accuracy    RMSE(T_pred, T_true)
    7.2  Cosmological fidelity   ΔC_ℓ = |C_ℓ^pred − C_ℓ^true|
    7.3  Uncertainty calibration  coverage of credible intervals

Saves: metrics.json, reconstruction.png, power_spectrum.png, calibration.png

Usage
─────
    python evaluate.py \
        --checkpoint ./outputs/run1/best_model.pt \
        --data       ./data/simulated/sim_dataset.h5 \
        --n_samples  8 \
        --out_dir    ./outputs/eval
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from models.cmb_model import CMBDiffusionModel, power_spectrum_1d
from utils.dataset import build_dataloaders


# ─── Metrics ──────────────────────────────────────────────────────────────────

def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(((pred - true) ** 2).mean()))


def delta_cl(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Mean |ΔC_ℓ| over the batch."""
    Pk_p = power_spectrum_1d(torch.from_numpy(pred)).numpy()
    Pk_t = power_spectrum_1d(torch.from_numpy(true)).numpy()
    return np.abs(Pk_p - Pk_t).mean(axis=0)


def coverage(samples: np.ndarray, true: np.ndarray, alpha: float) -> float:
    """
    Pixel-wise credible interval coverage.
    samples : (S, N, P, P)   true : (N, P, P)
    """
    lo = np.quantile(samples, (1 - alpha) / 2, axis=0)
    hi = np.quantile(samples, (1 + alpha) / 2, axis=0)
    return float(((true >= lo) & (true <= hi)).mean())


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_reconstruction(y_mean, y_std, y_true, path: Path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    panels = [
        (y_true[0],             "Ground Truth",    "RdBu_r"),
        (y_mean[0],             "Mean Prediction", "RdBu_r"),
        (y_std[0],              "Uncertainty σ",   "inferno"),
        (y_mean[0] - y_true[0], "Residual",        "RdBu_r"),
    ]
    for ax, (img, title, cmap) in zip(axes, panels):
        vmax = np.abs(img).max()
        vmin = -vmax if cmap == "RdBu_r" else 0
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=12); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {path.name}")


def plot_power_spectrum(dcl: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(np.arange(len(dcl))[2:], dcl[2:], color="#e34a33", lw=2,
                label=r"$|\Delta C_\ell|$")
    ax.set_xlabel(r"Angular scale bin $\ell$", fontsize=13)
    ax.set_ylabel(r"$|\Delta C_\ell|$", fontsize=13)
    ax.set_title("Power Spectrum Residual", fontsize=14)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {path.name}")


def plot_calibration(samples: np.ndarray, true: np.ndarray, path: Path):
    alphas   = np.linspace(0.05, 0.99, 20)
    observed = [coverage(samples, true, a) for a in alphas]
    fig, ax  = plt.subplots(figsize=(6, 6))
    ax.plot(alphas, alphas,   "k--", lw=1.5, label="Perfect calibration")
    ax.plot(alphas, observed, "o-",  color="#2b8cbe", lw=2,
            markersize=5, label="Model")
    ax.set_xlabel("Credible interval width α", fontsize=13)
    ax.set_ylabel("Empirical coverage",        fontsize=13)
    ax.set_title("Uncertainty Calibration",    fontsize=14)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    device = (torch.device("cuda")  if torch.cuda.is_available()  else
              torch.device("mps")   if torch.backends.mps.is_available() else
              torch.device("cpu"))
    print(f"\n── Device: {device} ──")

    print(f"\n── Loading checkpoint: {args.checkpoint} ──")
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = CMBDiffusionModel(
        in_channels   = args.in_channels,
        patch_size    = args.patch_size,
        base_dim      = args.base_dim,
        encoder_depth = args.encoder_depth,
        T_diffusion   = args.T_diffusion,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}")

    _, val_loader = build_dataloaders(
        [args.data], val_fraction=0.2, batch_size=args.batch_size,
        num_workers=0, augment=False,
    )

    # Collect patches
    X_list, y_list = [], []
    for batch in val_loader:
        X_list.append(batch["X"]); y_list.append(batch["y"])
        if sum(x.shape[0] for x in X_list) >= args.n_patches:
            break
    X_all = torch.cat(X_list)[:args.n_patches].to(device)
    y_all = torch.cat(y_list)[:args.n_patches]

    print(f"\n── Sampling {args.n_samples} realisations "
          f"over {len(X_all)} patches ──")
    samps = []
    for s in range(args.n_samples):
        print(f"  sample {s+1}/{args.n_samples} ...", end=" ", flush=True)
        with torch.no_grad():
            samps.append(model.sample(X_all, n_samples=1, temperature=args.temperature)[0].cpu().numpy())
        print("done")

    samples = np.stack(samps)            # (S, N, P, P)
    y_np    = y_all.numpy()
    y_mean  = samples.mean(axis=0)
    y_std   = samples.std(axis=0)

    # Metrics
    r = rmse(y_mean, y_np)
    dc = delta_cl(y_mean, y_np)
    c68 = coverage(samples, y_np, 0.68)
    c95 = coverage(samples, y_np, 0.95)

    print(f"\n{'─'*48}")
    print(f"  RMSE (normalised)              : {r:.4f}")
    print(f"  ΔC_ℓ  mean (bins 2–10)        : {dc[2:10].mean():.4f}")
    print(f"  Coverage @ 68%  (ideal 0.68)  : {c68:.3f}")
    print(f"  Coverage @ 95%  (ideal 0.95)  : {c95:.3f}")
    print(f"{'─'*48}")

    metrics = dict(rmse=r, delta_cl_mean_low_ell=float(dc[2:10].mean()),
                   coverage_68=c68, coverage_95=c95,
                   n_patches=len(X_all), n_samples=args.n_samples)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  metrics.json saved")

    print("\n── Generating plots ──")
    plot_reconstruction(y_mean, y_std, y_np, out / "reconstruction.png")
    plot_power_spectrum(dc,                   out / "power_spectrum.png")
    plot_calibration(samples, y_np,           out / "calibration.png")

    print(f"\n✅  Evaluation complete → {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data",        required=True)
    p.add_argument("--n_samples",   type=int, default=8)
    p.add_argument("--n_patches",   type=int, default=256)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--in_channels",   type=int, default=3)
    p.add_argument("--patch_size",    type=int, default=64)
    p.add_argument("--base_dim",      type=int, default=64)
    p.add_argument("--encoder_depth", type=int, default=3)
    p.add_argument("--T_diffusion",   type=int, default=50)
    p.add_argument("--temperature",  type=float, default=1.5,
                   help="Noise temperature for sampling (>1 = more diverse)")
    p.add_argument("--out_dir",     default="./outputs/eval")
    main(p.parse_args())