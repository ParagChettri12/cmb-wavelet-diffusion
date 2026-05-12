"""
train.py
--------
Training loop for CMBDiffusionModel.

Key upgrades over run1
----------------------
  * ReduceLROnPlateau  -- halves LR when val plateaus (beats cosine on this task)
  * torch.compile      -- free 15-30% speedup on A100 (--compile flag)
  * Proper AMP         -- float16 on CUDA, bfloat16 on CPU
  * encoder_depth arg  -- scale encoder capacity for production runs
  * num_workers auto   -- 4 on Linux/HPC, 0 on Windows

Usage
-----
    # Scale test
    python train.py --data ./data/sim_20k/sim_dataset.h5 \
        --base_dim 128 --encoder_depth 4 --T_diffusion 100 \
        --epochs 30 --lr 1e-3 --lambda_ps 0.5 --batch_size 64 \
        --compile --out_dir ./outputs/scale_test

    # Production (HPC)
    python train.py --data ./data/sim_50k/sim_dataset.h5 \
        --base_dim 256 --encoder_depth 5 --T_diffusion 200 \
        --epochs 200 --lr 1e-3 --lambda_ps 0.5 --batch_size 128 \
        --compile --out_dir ./outputs/production

    # Fine-tune on real data
    python train.py \
        --data ./data/sim_50k/sim_dataset.h5 ./data/processed/dataset.h5 \
        --base_dim 256 --encoder_depth 5 --T_diffusion 200 \
        --epochs 50 --lr 1e-4 --lambda_ps 0.5 --batch_size 64 \
        --compile --checkpoint ./outputs/production/best_model.pt \
        --out_dir ./outputs/final
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from models.cmb_model import CMBDiffusionModel
from utils.dataset import build_dataloaders


# --- AMP dtype selection ------------------------------------------------------

def amp_dtype(device):
    if device.type == "cuda":
        return torch.float16
    if device.type == "cpu":
        return torch.bfloat16
    return torch.float32


# --- Metric tracker -----------------------------------------------------------

class Meter:
    def __init__(self):
        self.total = self.n = 0.0

    def update(self, val, n=1):
        self.total += val * n
        self.n += n

    @property
    def avg(self):
        return self.total / max(self.n, 1)


def _scalar(v):
    return v.item() if hasattr(v, "item") else float(v)


# --- Train / val epoch --------------------------------------------------------

def train_epoch(model, loader, opt, device, scaler, dtype):
    model.train()
    ms = {k: Meter() for k in ["loss", "L_diffusion", "L_ps", "L_freq", "L_z"]}
    for batch in loader:
        X = batch["X"].to(device)
        y = batch["y"].to(device)
        z = batch["z"].to(device) if batch["has_z"] else None
        opt.zero_grad()
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=(dtype != torch.float32)):
            losses = model(X, y, z_true=z)
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        n = X.shape[0]
        for k in ms:
            ms[k].update(_scalar(losses[k]), n)
    return {k: m.avg for k, m in ms.items()}


@torch.no_grad()
def val_epoch(model, loader, device, dtype):
    model.eval()
    ms = {k: Meter() for k in ["loss", "L_diffusion", "L_ps"]}
    for batch in loader:
        X = batch["X"].to(device)
        y = batch["y"].to(device)
        z = batch["z"].to(device) if batch["has_z"] else None
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=(dtype != torch.float32)):
            losses = model(X, y, z_true=z)
        n = X.shape[0]
        for k in ms:
            ms[k].update(_scalar(losses[k]), n)
    return {k: m.avg for k, m in ms.items()}


# --- Main ---------------------------------------------------------------------

def main(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Device
    device = (torch.device("cuda")  if torch.cuda.is_available()  else
              torch.device("mps")   if torch.backends.mps.is_available() else
              torch.device("cpu"))
    dtype = amp_dtype(device)
    print(f"\n-- Device: {device}  AMP dtype: {dtype} --")
    if device.type == "cuda":
        print(f"   GPU : {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Data
    print("\n-- Loading dataset(s) --")
    num_workers = args.num_workers
    if num_workers < 0:
        num_workers = 4 if os.name != "nt" else 0
    print(f"   num_workers={num_workers}")
    train_loader, val_loader = build_dataloaders(
        h5_paths     = args.data,
        val_fraction = args.val_fraction,
        batch_size   = args.batch_size,
        num_workers  = num_workers,
        augment      = True,
        seed         = args.seed,
    )

    # Model
    model = CMBDiffusionModel(
        in_channels   = args.in_channels,
        patch_size    = args.patch_size,
        base_dim      = args.base_dim,
        encoder_depth = args.encoder_depth,
        T_diffusion   = args.T_diffusion,
        lambda_ps     = args.lambda_ps,
        lambda_z      = args.lambda_z,
    ).to(device)

    # torch.compile -- CUDA only, requires PyTorch >= 2.0
    if args.compile:
        if hasattr(torch, "compile") and device.type == "cuda":
            print("\n-- torch.compile enabled --")
            model = torch.compile(model)
        else:
            print("\n-- torch.compile skipped (needs CUDA + PyTorch >= 2.0) --")

    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n-- Model: {n_p:,} trainable parameters --")
    print(f"   base_dim={args.base_dim}  encoder_depth={args.encoder_depth}"
          f"  T_diffusion={args.T_diffusion}")

    # Checkpoint resume
    start_epoch = 0
    best_val    = float("inf")
    if args.checkpoint:
        ckpt  = torch.load(args.checkpoint, map_location=device)
        state = ckpt["model"]
        try:
            model.load_state_dict(state)
        except RuntimeError:
            # Strip torch.compile prefix if present
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
            model.load_state_dict(state)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val    = ckpt.get("val_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch-1}  (best val: {best_val:.4f})")

    # Optimiser + ReduceLROnPlateau
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = ReduceLROnPlateau(
        opt,
        mode     = "min",
        factor   = 0.5,
        patience = args.patience,
        min_lr   = 1e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    history = []
    print(f"\n-- Training for {args.epochs} epochs --")
    print(f"   ReduceLROnPlateau  patience={args.patience}  factor=0.5\n")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0  = time.time()
        tr  = train_epoch(model, train_loader, opt, device, scaler, dtype)
        vl  = val_epoch(model, val_loader, device, dtype)
        sched.step(vl["loss"])
        lr_now = opt.param_groups[0]["lr"]
        dt     = time.time() - t0

        print(f"E{epoch:03d} | "
              f"train {tr['loss']:.4f} "
              f"(Ld={tr['L_diffusion']:.4f} Lps={tr['L_ps']:.4f}) | "
              f"val {vl['loss']:.4f} | "
              f"lr {lr_now:.2e} | {dt:.1f}s")

        history.append(dict(
            epoch=epoch, lr=lr_now, time_s=dt,
            **{f"tr_{k}": v for k, v in tr.items()},
            **{f"vl_{k}": v for k, v in vl.items()},
        ))

        if vl["loss"] < best_val:
            best_val  = vl["loss"]
            raw_model = getattr(model, "_orig_mod", model)
            torch.save(
                {"epoch": epoch, "model": raw_model.state_dict(),
                 "val_loss": best_val, "args": vars(args)},
                out / "best_model.pt",
            )

        if (epoch + 1) % args.save_every == 0:
            raw_model = getattr(model, "_orig_mod", model)
            torch.save({"epoch": epoch, "model": raw_model.state_dict()},
                       out / f"ckpt_{epoch:03d}.pt")

    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone.  Best val loss: {best_val:.4f}")
    print(f"   Best model -> {out / 'best_model.pt'}")


# --- Args ---------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--data",          nargs="+", required=True)
    p.add_argument("--val_fraction",  type=float, default=0.10)
    p.add_argument("--num_workers",   type=int,   default=-1)
    # Model
    p.add_argument("--in_channels",   type=int,   default=3)
    p.add_argument("--patch_size",    type=int,   default=64)
    p.add_argument("--base_dim",      type=int,   default=64)
    p.add_argument("--encoder_depth", type=int,   default=3)
    p.add_argument("--T_diffusion",   type=int,   default=50)
    p.add_argument("--lambda_ps",     type=float, default=0.10)
    p.add_argument("--lambda_z",      type=float, default=0.01)
    # Training
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--patience",      type=int,   default=5)
    p.add_argument("--save_every",    type=int,   default=10)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--compile",       action="store_true")
    # I/O
    p.add_argument("--out_dir",       default="./outputs/run1")
    p.add_argument("--checkpoint",    default=None)
    main(p.parse_args())