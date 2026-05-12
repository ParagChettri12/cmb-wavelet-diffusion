"""
download_data.py
────────────────
Downloads Planck frequency maps and the SMICA component-separated map
from the NASA IRSA archive (Planck Public Release 3).

Usage
─────
    python download_data.py --output_dir ./data/raw

What gets downloaded
────────────────────
    planck_070GHz.fits   ~1 GB   70 GHz LFI frequency map
    planck_100GHz.fits   ~1 GB  100 GHz HFI frequency map
    planck_143GHz.fits   ~1 GB  143 GHz HFI frequency map  (your X channels)
    smica_cmb.fits       ~1 GB  SMICA cleaned CMB map       (your y target)

Total: ~ 4 GB.  preprocess.py then downgrades to Nside=128 (~ 5 MB each).

URL notes
─────────
All files are hosted under:
    https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/

The 70 GHz map is an LFI product (prefix LFI_SkyMap, Nside=1024, R2.01).
The 100/143 GHz maps are HFI products (prefix HFI_SkyMap, Nside=2048, R3.01).
The SMICA map is a component-separation product under the same release.
"""

import urllib.request
from pathlib import Path

from tqdm import tqdm

# ── Planck Public Release 3 (PR3) URLs from NASA IRSA ────────────────────────
# Verified against https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/
PLANCK_MAPS = {
    "planck_070GHz.fits": (
        "https://irsa.ipac.caltech.edu/data/Planck/release_3/"
        "all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_full.fits"
    ),
    "planck_100GHz.fits": (
        "https://irsa.ipac.caltech.edu/data/Planck/release_3/"
        "all-sky-maps/maps/HFI_SkyMap_100_2048_R3.01_full.fits"
    ),
    "planck_143GHz.fits": (
        "https://irsa.ipac.caltech.edu/data/Planck/release_3/"
        "all-sky-maps/maps/HFI_SkyMap_143_2048_R3.01_full.fits"
    ),
    "smica_cmb.fits": (
        "https://irsa.ipac.caltech.edu/data/Planck/release_3/"
        "all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits"
    ),
}


class _ProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  ✓ already present: {dest.name}  ({dest.stat().st_size / 1e9:.2f} GB)")
        return
    print(f"  ↓ {dest.name}")
    print(f"    {url}")
    try:
        with _ProgressBar(unit="B", unit_scale=True, miniters=1,
                          desc=dest.name) as t:
            urllib.request.urlretrieve(url, dest, reporthook=t.update_to)
        print(f"    saved → {dest}  ({dest.stat().st_size / 1e9:.2f} GB)")
    except Exception as e:
        # Clean up partial download so resume works correctly next run
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Download failed for {dest.name}: {e}") from e


def main(output_dir: str = "./data/raw") -> None:
    raw = Path(output_dir)
    raw.mkdir(parents=True, exist_ok=True)

    print("\n── Downloading Planck PR3 maps from NASA IRSA ──\n")
    print("  Total download: ~4 GB")
    print("  Files will be saved to:", raw.resolve())
    print()

    for fname, url in PLANCK_MAPS.items():
        download_file(url, raw / fname)

    print("\n✅  All downloads complete.")
    print(f"   Files in: {raw.resolve()}")
    print("\nNext step:")
    print("   python preprocess.py --raw_dir ./data/raw --out_dir ./data/processed")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="./data/raw")
    main(p.parse_args().output_dir)