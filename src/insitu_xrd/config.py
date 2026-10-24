"""Project-wide settings.

Values that change per experiment (detector geometry, beam energy, default image paths) are read
from [geometry] and [paths] in experiments.toml in the project folder; the defaults below apply
when the file or an entry is missing. Only code-side constants (HDF5 keys, EPICS epoch, output
folder layout) live here.

Image paths are not searched for automatically: write the real paths in experiments.toml or on
the command line.
"""

import os
import tempfile
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"  # local auxiliary data (temperature logs etc.)
OUT_DIR = PROJECT_ROOT / "out"  # all outputs (out/<sample>/...)
CACHE_DIR = OUT_DIR / "cache"  # integration cache (recomputed if deleted)
SETTINGS_FILE = PROJECT_ROOT / "experiments.toml"

DSET = "entry/data/data"
ATTR = "entry/instrument/NDAttributes"
EPICS_EPOCH_OFFSET = 631_152_000  # 1990-01-01 UTC - 1970-01-01 UTC [s]


def load_toml(path: Path) -> dict:
    """Read a TOML file; on a syntax error, exit naming the file."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"❌ {Path(path).name} 형식 오류: {e}") from e


def project_path(s: str | Path) -> Path:
    """Path from a settings file: absolute as is, relative to the project folder otherwise."""
    p = Path(s)
    return p if p.is_absolute() else PROJECT_ROOT / p


_S = load_toml(SETTINGS_FILE) if SETTINGS_FILE.exists() else {}
_geo = _S.get("geometry", {})
_paths = _S.get("paths", {})

# Integration cache (out/cache):
#   use     = reuse when the folder files (count, modification times), geometry and options match (default)
#   refresh = always integrate again and overwrite the cache
#   off     = neither read nor write the cache
# Priority: the command's --cache > environment variable INSITU_XRD_CACHE > experiments.toml [options] cache > "use"
CACHE_MODES = ("use", "refresh", "off")
CACHE_MODE = str(
    os.environ.get("INSITU_XRD_CACHE") or _S.get("options", {}).get("cache", "use")
).lower()
if CACHE_MODE not in CACHE_MODES:
    raise SystemExit(f"❌ cache = '{CACHE_MODE}' 는 잘못된 값 ({' / '.join(CACHE_MODES)})")

PONI_PX = tuple(float(v) for v in _geo.get("poni_px", (514.0, 850.0)))  # PONI (x, y) [px]
SDD_MM = float(_geo.get("sdd_mm", 861.14))  # sample to PONI distance [mm]
ENERGY_KEV = float(_geo.get("energy_kev", 13.0))  # X-ray energy [keV]
ALPHA_DEG = float(_geo.get("alpha_deg", 18.9))  # detector arm 2θ [deg]
PIXEL_UM = float(_geo.get("pixel_um", 75.0))  # pixel size [µm]

# defaults for commands run without a path (timing, center, profile, heatmap)
DEFAULT_FOLDER = project_path(_paths.get("default_folder", r"Z:\exp\hkim\261004\images\testx1"))
DEFAULT_FILE = project_path(_paths.get("default_file", DEFAULT_FOLDER / "testx1_00089.h5"))

# Saving results (PNG, CSV, HTML) is off by default: figures are only shown and the HTML for the
# browser goes to a temporary folder. Turn it on with --save (under out/) or --save-dir DIR on any
# command, the environment variable INSITU_XRD_SAVE=1, or experiments.toml [options] save = true.
VIEW_DIR = Path(tempfile.gettempdir()) / "insitu-xrd"
_save_root: Path | None = (
    OUT_DIR
    if (
        os.environ.get("INSITU_XRD_SAVE", "").lower() in ("1", "true", "yes")
        or _S.get("options", {}).get("save", False)
    )
    else None
)


def set_save(root: Path | None) -> None:
    """Set the save folder (None = do not save, temporary folder only)."""
    global _save_root
    _save_root = root


def saving() -> bool:
    return _save_root is not None


def out_dir(sub: str) -> Path:
    """Output folder: <out>/<sub> when saving is on, otherwise a temporary folder (for showing HTML)."""
    d = (_save_root or VIEW_DIR) / sub
    d.mkdir(parents=True, exist_ok=True)
    return d
