"""
utils/dataset.py
────────────────
PyTorch Dataset + DataLoader for HDF5 patch files produced by
simulate.py or preprocess.py.  No healpy or astropy-healpix here.

Key change from original:
  Preloads entire HDF5 dataset into RAM on init instead of reading
  per-sample from disk on every __getitem__ call.  Your dataset is
  ~1.2 GB which fits easily in cluster RAM, and this eliminates the
  HDF5 I/O bottleneck that was causing ~3600s/epoch on the A100.
"""
from pathlib import Path

import h5py
import torch
from torch.utils.data import (
    ConcatDataset, DataLoader, Dataset, random_split
)


class CMBPatchDataset(Dataset):
    """
    In-memory CMB patch dataset loaded from HDF5.

    Loads all data into RAM on __init__ so __getitem__ is a pure
    tensor slice — no file I/O during training.

    Each item is a dict:
        X      : (C, P, P) float32   multi-freq observation
        y      : (P, P)    float32   CMB target
        z      : (3,)      float32   foreground params (zeros if unavailable)
        has_z  : bool
    """

    def __init__(self, h5_path: str | Path, augment: bool = False):
        self.path    = Path(h5_path)
        self.augment = augment

        print(f"  Loading {self.path} into RAM ...", flush=True)

        with h5py.File(self.path, "r") as f:
            # Load everything at once — one sequential read is fast
            self.X     = torch.from_numpy(f["X"][:])   # (N, C, P, P)
            self.y     = torch.from_numpy(f["y"][:])   # (N, P, P)
            self.has_z = "z" in f
            self.z     = (
                torch.from_numpy(f["z"][:])             # (N, 3)
                if self.has_z
                else torch.zeros(len(self.X), 3)
            )

        print(
            f"  Loaded: X={tuple(self.X.shape)}  "
            f"y={tuple(self.y.shape)}  "
            f"z={tuple(self.z.shape)}  "
            f"({(self.X.nelement() * self.X.element_size() + self.y.nelement() * self.y.element_size()) / 1e6:.0f} MB)",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict:
        X = self.X[idx]   # pure tensor slice, no I/O
        y = self.y[idx]
        z = self.z[idx]

        if self.augment:
            X, y = _flip(X, y)
            X, y = _rot90(X, y)

        return {"X": X, "y": y, "z": z, "has_z": self.has_z}


# ── Augmentations ────────────────────────────────────────────────────────────

def _flip(X: torch.Tensor, y: torch.Tensor):
    if torch.rand(1) > 0.5:
        X = X.flip(-1)
        y = y.flip(-1)
    if torch.rand(1) > 0.5:
        X = X.flip(-2)
        y = y.flip(-2)
    return X, y


def _rot90(X: torch.Tensor, y: torch.Tensor):
    k = int(torch.randint(4, (1,)))
    return torch.rot90(X, k, (-2, -1)), torch.rot90(y, k, (-2, -1))


# ── Collate ──────────────────────────────────────────────────────────────────

def _collate(batch: list[dict]) -> dict:
    return {
        "X":     torch.stack([b["X"] for b in batch]),
        "y":     torch.stack([b["y"] for b in batch]),
        "z":     torch.stack([b["z"] for b in batch]),
        "has_z": batch[0]["has_z"],
    }


# ── DataLoader builder ───────────────────────────────────────────────────────

def build_dataloaders(
    h5_paths:     list[str],
    val_fraction: float = 0.10,
    batch_size:   int   = 16,
    num_workers:  int   = 2,
    augment:      bool  = True,
    seed:         int   = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train / val DataLoaders from one or more HDF5 files.
    Simulated and real data can be freely mixed.

    With in-memory datasets, num_workers > 0 still helps because
    augmentation and collation run in parallel across workers.
    4–8 workers is a good default on HPC nodes.
    """
    datasets = [CMBPatchDataset(p, augment=augment) for p in h5_paths]
    full_ds  = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    n_val   = int(len(full_ds) * val_fraction)
    n_train = len(full_ds) - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    kw = dict(
    batch_size         = batch_size,
    num_workers        = num_workers,
    pin_memory         = torch.cuda.is_available(),
    persistent_workers = (num_workers > 0),
    collate_fn         = _collate,
    )
    if num_workers > 0:
        kw["prefetch_factor"] = 4

    print(f"  train={n_train}  val={n_val}  batch={batch_size}")

    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
    )