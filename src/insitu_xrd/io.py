"""Read HDF5 detector images and EPICS timestamps."""

import os
from pathlib import Path

import hdf5plugin  # noqa: F401  (registers the compression filters)
import h5py
import numpy as np

from .config import ATTR, DSET, EPICS_EPOCH_OFFSET
from .geometry import FArray


def _image(f: h5py.File, dset: str) -> FArray:
    """2D image (float64). Detector gap and bad pixels (negative, saturated) become NaN."""
    img = np.asarray(f[dset][()], dtype=np.float64)
    img = img[0] if img.ndim == 3 else img
    img[(img < 0) | (img >= np.iinfo(np.int32).max)] = np.nan
    return img


def _epics(f: h5py.File, attr: str) -> tuple[int, float] | None:
    try:
        g = f[attr]
        sec = int(g["NDArrayEpicsTSSec"][0])
        nsec = int(g["NDArrayEpicsTSnSec"][0])
        uid = int(g["NDArrayUniqueId"][0])
    except KeyError:
        return None
    return uid, sec + EPICS_EPOCH_OFFSET + nsec * 1e-9


def load_image(path: Path, dset: str = DSET) -> FArray:
    with h5py.File(path, "r") as f:
        return _image(f, dset)


def read_epics(path: Path, attr: str = ATTR) -> tuple[int, float] | None:
    """(UniqueId, EPICS acquisition time in Unix seconds), or None if the attributes are missing."""
    with h5py.File(path, "r") as f:
        return _epics(f, attr)


def read_frame(path: Path, dset: str = DSET, attr: str = ATTR) -> tuple[FArray, float]:
    """(image, acquisition time) from a single open of the file; opens dominate time on a network drive."""
    with h5py.File(path, "r") as f:
        img, ep = _image(f, dset), _epics(f, attr)
    return img, ep[1] if ep else file_birth_time(path)


def file_birth_time(path: Path) -> float:
    """File creation time in Unix seconds: st_birthtime where available, else st_ctime on Windows."""
    st = path.stat()
    t = getattr(st, "st_birthtime", None)
    if t is None:
        t = st.st_ctime if os.name == "nt" else st.st_mtime
    return t


def frame_time(path: Path, attr: str = ATTR) -> float:
    """EPICS acquisition time, or the file creation time if absent."""
    ep = read_epics(path, attr)
    return ep[1] if ep else file_birth_time(path)


def list_files(folder: Path, pattern: str = "*.h5", recursive: bool = False) -> list[Path]:
    if not folder.is_dir():
        raise SystemExit(f"❌ 폴더 없음: {folder}")
    files = sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))
    if not files:
        raise SystemExit(f"❌ 파일 없음: {folder / pattern}")
    return files
