"""HDF5 디텍터 이미지 / EPICS 타임스탬프 읽기."""

import os
from pathlib import Path

import hdf5plugin  # noqa: F401  (압축 데이터 디코딩 등록)
import h5py
import numpy as np

from .config import ATTR, DSET, EPICS_EPOCH_OFFSET
from .geometry import FArray


def _image(f: h5py.File, dset: str) -> FArray:
    """2D 이미지 (float64). 디텍터 갭/불량 픽셀(음수, 포화값)은 NaN."""
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
    """(UniqueId, EPICS 촬영시각 Unix 초). 속성이 없으면 None."""
    with h5py.File(path, "r") as f:
        return _epics(f, attr)


def read_frame(path: Path, dset: str = DSET, attr: str = ATTR) -> tuple[FArray, float]:
    """(이미지, 촬영시각) 을 파일 한 번 열어서 읽음 (네트워크 드라이브에서 열기 횟수가 곧 속도)."""
    with h5py.File(path, "r") as f:
        img, ep = _image(f, dset), _epics(f, attr)
    return img, ep[1] if ep else file_birth_time(path)


def file_birth_time(path: Path) -> float:
    """파일 생성시간(Unix 초). Python 3.12+ 는 st_birthtime, Windows 구버전은 st_ctime."""
    st = path.stat()
    t = getattr(st, "st_birthtime", None)
    if t is None:
        t = st.st_ctime if os.name == "nt" else st.st_mtime
    return t


def frame_time(path: Path, attr: str = ATTR) -> float:
    """EPICS 촬영시각, 없으면 파일 생성시간."""
    ep = read_epics(path, attr)
    return ep[1] if ep else file_birth_time(path)


def list_files(folder: Path, pattern: str = "*.h5", recursive: bool = False) -> list[Path]:
    if not folder.is_dir():
        raise SystemExit(f"❌ 폴더 없음: {folder}")
    files = sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))
    if not files:
        raise SystemExit(f"❌ 파일 없음: {folder / pattern}")
    return files
