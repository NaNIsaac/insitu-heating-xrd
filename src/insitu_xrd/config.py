"""Project-wide settings.

Values that change per experiment (detector geometry, beam energy, default image paths) are read
from [geometry] and [paths] in experiments.toml in the project folder; an optional gitignored
experiments.local.toml (or <stem>.local.toml beside any settings file) overrides matching entries
when present. Defaults below apply when the file or an entry is missing. Only code-side constants
(HDF5 keys, EPICS epoch, output folder layout) live here.

Image paths are not searched for automatically: write real local paths in experiments.local.toml
or on the command line.
"""

import os
import re
import tempfile
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"  # local auxiliary data (temperature logs etc.)
OUT_DIR = PROJECT_ROOT / "out"  # all outputs (out/<sample>/...)
CACHE_DIR = OUT_DIR / "cache"  # integration cache (recomputed if deleted)
SETTINGS_FILE = PROJECT_ROOT / "experiments.toml"
OVERRIDE_FILE = PROJECT_ROOT / "experiments.local.toml"

DSET = "entry/data/data"
ATTR = "entry/instrument/NDAttributes"
EPICS_EPOCH_OFFSET = 631_152_000  # 1990-01-01 UTC - 1970-01-01 UTC [s]


def override_path(path: Path) -> Path | None:
    """Return the companion *.local.toml path unless `path` itself is already local."""
    p = Path(path)
    if p.stem.endswith(".local"):
        return None
    return p.with_name(f"{p.stem}.local{p.suffix}")


def _read_toml(path: Path) -> dict:
    """Read a TOML file; on a syntax error, exit naming the file."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"❌ {Path(path).name} 형식 오류: {e}") from e


def _merge_toml(base: dict, override: dict) -> dict:
    """Recursively deep-merge tables; replace arrays of tables and scalars."""
    merged = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_toml(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_toml(path: Path) -> dict:
    """Read a TOML file and merge its companion *.local.toml override when present."""
    p = Path(path)
    data = _read_toml(p)
    ov = override_path(p)
    if ov is not None and ov.exists():
        return _merge_toml(data, _read_toml(ov))
    return data


_ROOT_RE = re.compile(r"^<([^<>]+)>(?:[/\\](.*))?$")


def _expand_path(s: str | Path, *, roots: dict[str, str], strict: bool) -> Path:
    text = str(s)
    if m := _ROOT_RE.match(text):
        name, rest = m.group(1), m.group(2) or ""
        base = roots.get(name)
        if not base or not str(base).strip():
            if strict:
                raise SystemExit(
                    f"❌ root '<{name}>' is not set — add `{name} = '...'` under [roots] in {OVERRIDE_FILE.name}"
                )
            return Path(text)
        root = Path(base)
        root = root if root.is_absolute() else PROJECT_ROOT / root
        return root / rest if rest else root
    p = Path(s)
    return p if p.is_absolute() else PROJECT_ROOT / p


def project_path(s: str | Path, *, roots: dict[str, str] | None = None) -> Path:
    """Path from a settings file: expand leading <root>/ via [roots], keep absolute, resolve relative to PROJECT_ROOT."""
    return _expand_path(s, roots=_roots if roots is None else roots, strict=True)


def resolve_root_arg(target: str | Path) -> Path:
    """Expand a leading <root>/... placeholder if present; otherwise return Path(target) unchanged."""
    text = str(target)
    return project_path(text) if _ROOT_RE.match(text) else Path(target)


_S = load_toml(SETTINGS_FILE) if SETTINGS_FILE.exists() else {}
_geo = _S.get("geometry", {})
_paths = _S.get("paths", {})
_roots: dict[str, str] = {str(k): str(v) for k, v in _S.get("roots", {}).items()}

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
DEFAULT_FOLDER = _expand_path(
    _paths.get("default_folder", "<images>/testx1"), roots=_roots, strict=False
)
DEFAULT_FILE = _expand_path(
    _paths.get("default_file", DEFAULT_FOLDER / "testx1_00089.h5"), roots=_roots, strict=False
)

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
