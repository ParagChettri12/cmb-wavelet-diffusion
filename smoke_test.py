"""
smoke_test.py  —  CMB Pipeline end-to-end validation
No downloads required.  Run:  python smoke_test.py
"""

# ── Windows-safe path setup ───────────────────────────────────────────────────
# os.path.abspath + dirname is the most reliable way to get the script's
# own directory on every OS, including Windows.
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Verify the local package folders exist before trying to import them
for _pkg in ("utils", "models"):
    _pkg_path = os.path.join(_HERE, _pkg)
    _init     = os.path.join(_pkg_path, "__init__.py")
    if not os.path.isdir(_pkg_path):
        print(f"\n✗  Folder '{_pkg}/' not found next to smoke_test.py")
        print(f"   Expected: {_pkg_path}")
        print("   Make sure you kept the folder structure intact.")
        sys.exit(1)
    if not os.path.isfile(_init):
        print(f"\n✗  Missing {_pkg}/__init__.py")
        print(f"   Expected: {_init}")
        sys.exit(1)

# ── Standard library ──────────────────────────────────────────────────────────
import time
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch

# ── Local packages (imported AFTER sys.path is fixed) ────────────────────────
from simulate          import main as sim_main
from utils.dataset     import build_dataloaders
from models.cmb_model  import CMBDiffusionModel, power_spectrum_1d


# ─── Helper ───────────────────────────────────────────────────────────────────

def ok(label: str, cond: bool, detail: str = "") -> None:
    icon = "✓" if cond else "✗"
    msg  = f"  {icon}  {label}" + (f"  ({detail})" if detail else "")
    print(msg)
    if not cond:
        print("\n✗  Smoke test FAILED.  See error above.")
        sys.exit(1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n══════════════════════════════════════")
    print("    CMB Pipeline  ·  Smoke Test")
    print(f"    Script dir: {_HERE}")
    print("══════════════════════════════════════\n")

    tmp = Path(_HERE) / "data" / "_smoke_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    # 1 ── Synthetic data generation ──────────────────────────────────────────
    print("── 1. Synthetic data generation ──")
    t0 = time.time()
    sim_main(
        out_dir    = str(tmp),
        n_sims     = 200,
        patch_size = 64,
        res_arcmin = 60.0,
        seed       = 0,
    )
    h5_path = tmp / "sim_dataset.h5"
    ok("simulate.py", h5_path.exists(), f"{time.time()-t0:.1f}s")

    # 2 ── DataLoader ──────────────────────────────────────────────────────────
    print("\n── 2. Dataset / DataLoader ──")
    train_loader, val_loader = build_dataloaders(
        h5_paths     = [str(h5_path)],
        val_fraction = 0.1,
        batch_size   = 8,
        num_workers  = 0,      # must be 0 on Windows outside __main__ guard
        augment      = True,
    )
    batch = next(iter(train_loader))
    ok("DataLoader",
       batch["X"].shape == (8, 3, 64, 64) and batch["y"].shape == (8, 64, 64),
       f"X={tuple(batch['X'].shape)}  y={tuple(batch['y'].shape)}")

    # 3 ── Model ───────────────────────────────────────────────────────────────
    print("\n── 3. Model ──")
    model = CMBDiffusionModel(
        in_channels = 3,
        patch_size  = 64,
        base_dim    = 32,
        T_diffusion = 10,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok("instantiation", n_params > 1_000, f"{n_params:,} params")

    # 4 ── Forward pass ────────────────────────────────────────────────────────
    print("\n── 4. Forward pass ──")
    model.train()
    X, y, z = batch["X"], batch["y"], batch["z"]
    t0     = time.time()
    losses = model(X, y, z_true=z)
    lv     = losses["loss"].item()
    ok("loss finite", np.isfinite(lv), f"loss={lv:.4f}  ({time.time()-t0:.1f}s)")
    ok("loss sane",   lv < 1000.0,   # high at random init; just checks it is finite & not NaN/Inf
       f"Ld={losses['L_diffusion']:.3f}  "
       f"Lps={losses['L_ps']:.3f}  "
       f"Lf={losses['L_freq']:.3f}  "
       f"Lz={losses['L_z']:.3f}")

    # 5 ── Backward pass ───────────────────────────────────────────────────────
    print("\n── 5. Backward pass ──")
    losses["loss"].backward()
    n_grad  = sum(1 for p in model.parameters()
                  if p.requires_grad and p.grad is not None)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    ok("gradients", n_grad == n_total, f"{n_grad}/{n_total} params")

    # 6 ── Sampling ────────────────────────────────────────────────────────────
    print("\n── 6. Sampling ──")
    model.eval()
    with torch.no_grad():
        t0      = time.time()
        samples = model.sample(X[:2], n_samples=1)
    ok("shape",  samples.shape == (1, 2, 64, 64),
       f"{tuple(samples.shape)}  ({time.time()-t0:.1f}s)")
    ok("finite", samples.isfinite().all().item())

    # 7 ── Power spectrum ──────────────────────────────────────────────────────
    print("\n── 7. Power spectrum ──")
    Pk = power_spectrum_1d(y)
    ok("shape",   Pk.shape == (8, 32), f"{tuple(Pk.shape)}")
    ok("non-neg", Pk.min().item() >= 0.0)

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════")
    print("  ✅  All checks passed!")
    print("══════════════════════════════════════\n")
    print("Next steps:")
    print("  python simulate.py  --out_dir ./data/simulated --n_sims 5000")
    print("  python train.py     --data ./data/simulated/sim_dataset.h5")
    print()


if __name__ == "__main__":
    main()
