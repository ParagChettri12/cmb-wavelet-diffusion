"""
preprocess.py
─────────────
Converts raw Planck .fits maps → normalised patch tensors stored in HDF5.
Uses astropy + astropy-healpix exclusively (no healpy).

Steps
─────
1. Read each .fits BINARY TABLE with astropy.io.fits
2. Downsample to Nside=128 via ring→nested→average→nested→ring
3. Project sphere → flat gnomonic patches via manual tangent-plane math
   + astropy-healpix nearest-pixel lookup
4. Normalise per-channel (zero-mean, unit-variance)
5. Save as data/processed/dataset.h5

Usage
─────
    python preprocess.py \
        --raw_dir    ./data/raw \
        --out_dir    ./data/processed \
        --nside      128 \
        --patch_size 64 \
        --n_patches  8000

HDF5 schema
───────────
    /X        (N, C, P, P)   float32   multi-freq input tensor
    /y        (N, P, P)      float32   SMICA target
    /meta/channel_labels     list[str]
    /meta/norm_mean_X        (C,)
    /meta/norm_std_X         (C,)
    /meta attrs: y_mu, y_std, nside, patch_size, n_patches, res_arcmin

Map Nside notes
───────────────
    070 GHz (LFI) : native Nside=1024  → downgraded to 128
    100 GHz (HFI) : native Nside=2048  → downgraded to 128
    143 GHz (HFI) : native Nside=2048  → downgraded to 128
    SMICA         : native Nside=2048  → downgraded to 128
    ud_grade handles all cases since 1024 and 2048 are both multiples of 128.
"""

import argparse
from pathlib import Path

import astropy.units as u
import astropy_healpix as ah
import h5py
import numpy as np
from astropy.coordinates import Galactic
from astropy.io import fits
from astropy_healpix import HEALPix
from astropy_healpix.core import ring_to_nested, nested_to_ring
from tqdm import tqdm

# ── Channel config ────────────────────────────────────────────────────────────
# Filenames must match what download_data.py saves to --output_dir
CHANNEL_FILES = {
    "070GHz": "planck_070GHz.fits",
    "100GHz": "planck_100GHz.fits",
    "143GHz": "planck_143GHz.fits",
}
TARGET_FILE = "smica_cmb.fits"

PLANCK_UNSEEN = -1.6375e30   # sentinel value for masked/unobserved pixels


# ── FITS reading ──────────────────────────────────────────────────────────────

def read_healpix_fits(path: Path) -> np.ndarray:
    """
    Read temperature column from a Planck FITS binary table.

    Planck PR3 FITS files store full-sky maps as binary tables in extension 1.
    The first column is always the intensity (I_STOKES / TEMPERATURE).
    Returns a 1-D float64 array in RING ordering.

    Note: HFI maps (100, 143 GHz) have 10 columns (I/Q/U Stokes + covariances).
          LFI maps (70 GHz) have 3 columns (I/Q/U Stokes).
          We always take columns[0] which is I_STOKES in both cases.
    """
    with fits.open(str(path)) as hdul:
        tbl      = hdul[1]
        col_name = tbl.columns.names[0]
        data     = tbl.data[col_name].flatten().astype(np.float64)
    return data


# ── Resolution downgrade ──────────────────────────────────────────────────────

def ud_grade(map_ring: np.ndarray, nside_in: int, nside_out: int) -> np.ndarray:
    """
    Downsample a HEALPix RING map from nside_in → nside_out by averaging
    child pixels.  nside_in must be an integer multiple of nside_out.

    Works for both 1024→128 (70 GHz LFI) and 2048→128 (HFI/SMICA).

    ring_to_nested and nested_to_ring are imported from astropy_healpix.core
    directly — they are not exposed at the astropy_healpix top-level namespace.
    """
    if nside_in == nside_out:
        return map_ring.copy()
    assert nside_in > nside_out and nside_in % nside_out == 0, (
        f"nside_in={nside_in} must be a multiple of nside_out={nside_out}"
    )
    n_in   = len(map_ring)
    n_out  = ah.nside_to_npix(nside_out)
    factor = (nside_in // nside_out) ** 2

    # RING → NESTED
    nested_of_ring = ring_to_nested(np.arange(n_in), nside_in)
    map_nested     = np.empty(n_in, dtype=np.float64)
    map_nested[nested_of_ring] = map_ring

    # Average child groups
    map_out_nested = map_nested.reshape(n_out, factor).mean(axis=1)

    # NESTED → RING
    ring_of_nested_out = nested_to_ring(np.arange(n_out), nside_out)
    map_out_ring       = np.empty(n_out, dtype=np.float64)
    map_out_ring[ring_of_nested_out] = map_out_nested
    return map_out_ring


# ── Map loader ────────────────────────────────────────────────────────────────

def load_map(path: Path, nside_out: int) -> np.ndarray:
    """Read a Planck .fits map, extract temperature, downsample."""
    raw      = read_healpix_fits(path)
    nside_in = ah.npix_to_nside(len(raw))
    print(f"    native Nside={nside_in}", end="")
    if nside_in != nside_out:
        raw = ud_grade(raw, nside_in, nside_out)
        print(f" → downgraded to {nside_out}", end="")
    raw[raw < PLANCK_UNSEEN * 0.5] = 0.0   # mask sentinel values
    return raw.astype(np.float32)


# ── Gnomonic patch extraction ─────────────────────────────────────────────────

def gnomonic_patch(hpmap: np.ndarray,
                   lon_deg: float, lat_deg: float,
                   patch_size: int,
                   res_arcmin: float = 30.0) -> np.ndarray:
    """
    Extract a flat (patch_size × patch_size) image from a HEALPix RING map
    using gnomonic (tangent-plane) projection centred on (lon_deg, lat_deg).
    """
    nside  = ah.npix_to_nside(len(hpmap))
    hp_obj = HEALPix(nside=nside, order="ring", frame=Galactic())

    half   = (patch_size - 1) / 2.0
    step   = res_arcmin / 60.0
    offs   = np.linspace(-half, half, patch_size) * step
    ox, oy = np.meshgrid(offs, offs)

    lon0  = np.radians(lon_deg)
    lat0  = np.radians(lat_deg)
    ox_r  = np.radians(ox)
    oy_r  = np.radians(oy)

    rho   = np.sqrt(ox_r**2 + oy_r**2)
    rho   = np.where(rho == 0.0, 1e-15, rho)
    c     = np.arctan(rho)

    lat   = np.arcsin(
        np.cos(c) * np.sin(lat0)
        + oy_r * np.sin(c) * np.cos(lat0) / rho
    )
    lon   = lon0 + np.arctan2(
        ox_r * np.sin(c),
        rho * np.cos(lat0) * np.cos(c) - oy_r * np.sin(lat0) * np.sin(c),
    )

    lon_grid = np.degrees(lon).ravel() % 360.0
    lat_grid = np.degrees(lat).ravel()

    coords = Galactic(l=lon_grid * u.deg, b=lat_grid * u.deg)
    pixels = hp_obj.skycoord_to_healpix(coords)
    patch  = hpmap[pixels].reshape(patch_size, patch_size).astype(np.float32)
    patch[~np.isfinite(patch)] = 0.0
    return patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def random_patch_centres(n: int,
                         avoid_plane_deg: float = 20.0,
                         seed: int = 42) -> list[tuple[float, float]]:
    """Sample n (lon, lat) pairs uniformly on the sphere, avoiding |b| < cut."""
    rng     = np.random.default_rng(seed)
    centres = []
    while len(centres) < n:
        lon = rng.uniform(0.0, 360.0)
        lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0)))
        if abs(lat) > avoid_plane_deg:
            centres.append((float(lon), float(lat)))
    return centres


def normalise(stack: np.ndarray):
    """Per-channel zero-mean unit-std.  stack: (N, C, P, P)."""
    mu  = stack.mean(axis=(0, 2, 3), keepdims=True)
    std = stack.std (axis=(0, 2, 3), keepdims=True) + 1e-8
    return (stack - mu) / std, mu.squeeze(), std.squeeze()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(raw_dir: str, out_dir: str, nside: int,
         patch_size: int, n_patches: int,
         res_arcmin: float, seed: int) -> None:

    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load maps
    print("\n── Loading frequency maps ──")
    freq_maps: dict[str, np.ndarray] = {}
    for label, fname in CHANNEL_FILES.items():
        fpath = raw / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Missing: {fpath}\n"
                "Run  python download_data.py  first."
            )
        print(f"  {label} ({fname}) ... ", end="", flush=True)
        freq_maps[label] = load_map(fpath, nside)
        print("  ok")

    print(f"  SMICA target ... ", end="", flush=True)
    smica_path = raw / TARGET_FILE
    if not smica_path.exists():
        raise FileNotFoundError(
            f"Missing: {smica_path}\n"
            "Run  python download_data.py  first."
        )
    smica = load_map(smica_path, nside)
    print("  ok")

    channel_labels = list(freq_maps.keys())
    C, P = len(channel_labels), patch_size

    # 2. Sample patch centres
    centres = random_patch_centres(n_patches, seed=seed)

    # 3. Extract patches
    print(f"\n── Extracting {n_patches} patches ({P}×{P}, {res_arcmin}'/px) ──")
    X_list, y_list = [], []
    for lon, lat in tqdm(centres, desc="Patches"):
        chans = [gnomonic_patch(freq_maps[lbl], lon, lat, P, res_arcmin)
                 for lbl in channel_labels]
        X_list.append(np.stack(chans, axis=0))
        y_list.append(gnomonic_patch(smica, lon, lat, P, res_arcmin))

    X = np.stack(X_list, axis=0).astype(np.float32)   # (N, C, P, P)
    y = np.stack(y_list, axis=0).astype(np.float32)   # (N, P, P)

    # 4. Normalise
    print("\n── Normalising ──")
    X_norm, mu, std = normalise(X)
    y_mu   = float(y.mean())
    y_std  = float(y.std() + 1e-8)
    y_norm = ((y - y_mu) / y_std).astype(np.float32)

    # 5. Save
    h5_path = out / "dataset.h5"
    print(f"\n── Saving → {h5_path} ──")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X", data=X_norm, compression="gzip", compression_opts=4)
        f.create_dataset("y", data=y_norm, compression="gzip", compression_opts=4)
        meta = f.create_group("meta")
        meta.create_dataset("channel_labels",
                            data=np.array(channel_labels, dtype="S"))
        meta.create_dataset("norm_mean_X", data=mu)
        meta.create_dataset("norm_std_X",  data=std)
        meta.attrs.update(dict(
            y_mu       = y_mu,
            y_std      = y_std,
            nside      = nside,
            patch_size = patch_size,
            n_patches  = n_patches,
            res_arcmin = res_arcmin,
        ))

    mb = h5_path.stat().st_size / 1e6
    print(f"  X {X_norm.shape}  y {y_norm.shape}  {mb:.1f} MB")
    print(f"\n✅  Preprocessing complete → {h5_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",    default="./data/raw")
    p.add_argument("--out_dir",    default="./data/processed")
    p.add_argument("--nside",      type=int,   default=128)
    p.add_argument("--patch_size", type=int,   default=64)
    p.add_argument("--n_patches",  type=int,   default=8000)
    p.add_argument("--res_arcmin", type=float, default=30.0)
    p.add_argument("--seed",       type=int,   default=42)
    main(**vars(p.parse_args()))