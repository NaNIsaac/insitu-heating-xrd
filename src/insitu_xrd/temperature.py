"""온도 컨트롤러 로그 CSV 병합 · 임의 시각의 온도 조회."""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .io import frame_time, list_files

KST = timezone(timedelta(hours=9))  # 로그의 Timestamp 는 KST (tz 표기 없음)
TEMP_DIR = DATA_DIR / "temperature"
DEFAULT_LOG = TEMP_DIR / "manual_data_20261004_x1.csv"

PV = "TC Temp PV [°C]"
SV = "TC Temp SV [°C]"


def _read(csvs: Sequence[Path], source: bool = False) -> pd.DataFrame:
    """여러 로그를 시간순으로 합침 (중복 Timestamp 제거). source=True 면 원본 파일명 열 추가."""
    dfs = []
    for p in csvs:
        df = pd.read_csv(p, encoding="utf-8-sig")
        dfs.append(df.assign(source=Path(p).stem) if source else df)
    df = pd.concat(dfs, ignore_index=True)
    t = pd.to_datetime(df["Timestamp"])
    order = np.argsort(t.to_numpy(), kind="stable")
    df = df.iloc[order].assign(_t=t.iloc[order].to_numpy())
    return df.drop_duplicates("Timestamp").reset_index(drop=True)


def merge(csvs: Sequence[Path], out: Path) -> pd.DataFrame:
    """여러 로그를 시간순으로 합치고 원본 파일명을 source 열에 남김."""
    df = _read(csvs, source=True).drop(columns="_t")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


class TempLog:
    """로그를 읽어 두고 Unix 시각 배열 → 온도 를 선형보간으로 조회.
    로그 샘플에서 max_gap [s] 넘게 떨어진 시각(기록이 끊긴 구간)은 NaN.
    path 에 CSV 여러 개를 주면 시간순으로 합침. 같은 파일은 load() 로 한 번만 읽음."""

    def __init__(self, path: Path | Sequence[Path] = DEFAULT_LOG, max_gap: float = 5.0):
        paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(p) for p in path]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"❌ 온도 로그 없음: {', '.join(map(str, missing))}")
        self.path, self.paths = paths[0], paths
        df = _read(paths)
        self.t = (
            (df["_t"].dt.tz_localize(KST) - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()
        )
        self.df = df.drop(columns="_t")
        self.max_gap = max_gap
        self._cols: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    @lru_cache(maxsize=16)
    def _load(paths: tuple[Path, ...], max_gap: float) -> "TempLog":
        return TempLog(paths, max_gap)

    @classmethod
    def load(cls, path: Path | Sequence[Path], max_gap: float = 5.0) -> "TempLog":
        """TempLog(...) 와 같지만 같은 파일(들)은 한 번만 읽어 재사용."""
        paths = (Path(path),) if isinstance(path, (str, Path)) else tuple(Path(p) for p in path)
        return cls._load(tuple(p.resolve() for p in paths), max_gap)

    @property
    def span(self) -> tuple[str, str]:
        return self.df["Timestamp"].iloc[0][:19], self.df["Timestamp"].iloc[-1][:19]

    def _column(self, column: str) -> tuple[np.ndarray, np.ndarray]:
        """(시각, 값) — 값이 있는 샘플만. 열마다 한 번만 변환."""
        if column not in self._cols:
            y = pd.to_numeric(self.df[column], errors="coerce").to_numpy(float)
            ok = np.isfinite(y)
            self._cols[column] = self.t[ok], y[ok]
        return self._cols[column]

    def at(self, times: float | Sequence[float] | np.ndarray, column: str = PV) -> np.ndarray:
        times = np.atleast_1d(np.asarray(times, dtype=float))
        t, y = self._column(column)
        val = np.interp(times, t, y)
        i = np.clip(np.searchsorted(t, times), 1, len(t) - 1)
        dist = np.minimum(np.abs(times - t[i - 1]), np.abs(t[i] - times))
        val[dist > self.max_gap] = np.nan
        return val


def parse_time(s: str, date: str = "2026-10-04") -> float:
    """'2026-10-04 15:55:00' 또는 '15:55:00' (KST) → Unix 초. 날짜 생략 시 date."""
    if len(s) <= 8:  # HH:MM[:SS]
        s = f"{date} {s}"
    return pd.Timestamp(s).tz_localize(KST).timestamp()


def fmt_kst(t: float, ms: bool = False) -> str:
    """Unix 초 → 'YYYY-MM-DD HH:MM:SS' (KST). ms=True 면 밀리초까지."""
    s = datetime.fromtimestamp(t, KST).strftime("%Y-%m-%d %H:%M:%S.%f")
    return s[:-3] if ms else s[:19]


def _span(path: Path) -> tuple[float, float] | None:
    """CSV 의 첫·마지막 Timestamp (Unix). 파일 전체를 읽지 않음."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            if not f.readline().startswith("Timestamp"):
                return None
            first = f.readline()
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            last = f.read().decode("utf-8", "ignore").strip().splitlines()[-1]
        return parse_time(first.split(",")[0][:19]), parse_time(last.split(",")[0][:19])
    except Exception:
        return None


def find_log(times: Sequence[float] | np.ndarray, root: Path = TEMP_DIR) -> Path | None:
    """root 아래 CSV 중 프레임 절대시각과 가장 많이 겹치는 로그 (병합본·raw 제외). 없으면 None."""
    t = np.asarray(times, float)
    best, n_best = None, 0
    for p in sorted(root.rglob("*.csv")):
        if "raw" in p.parts or p == DEFAULT_LOG or (sp := _span(p)) is None:
            continue
        n = int(((t >= sp[0] - 5) & (t <= sp[1] + 5)).sum())
        if n > n_best:
            best, n_best = p, n
    return best


def lookup(targets: Sequence[str], csv: Path = DEFAULT_LOG, max_gap: float = 5.0) -> pd.DataFrame:
    """시각 문자열 / .h5 파일 / h5 폴더 → 온도 표 출력."""
    log = TempLog(csv, max_gap)
    date = log.span[0][:10]
    names, times = [], []
    for s in targets:
        p = Path(s)
        if p.is_dir():
            files = list_files(p)
            names += [f.name for f in files]
            times += [frame_time(f) for f in files]
        elif p.suffix == ".h5" and p.exists():
            names.append(p.name)
            times.append(frame_time(p))
        else:
            names.append(s)
            times.append(parse_time(s, date))
    df = pd.DataFrame(
        {
            "target": names,
            "time_kst": [fmt_kst(t) for t in times],
            "temp_pv": log.at(times, PV),
            "temp_sv": log.at(times, SV),
        }
    )
    print(f"🌡️  {log.path.name}  ({log.span[0]} ~ {log.span[1]})")
    print(df.to_string(index=False))
    return df
