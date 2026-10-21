"""폴더 내 HDF5 파일의 생성시각 vs EPICS 촬영시각 비교 → CSV."""

from pathlib import Path

import pandas as pd

from .config import ATTR, out_dir
from .io import file_birth_time, list_files, read_epics
from .temperature import fmt_kst


def _fmt(t):
    return fmt_kst(t, ms=True) if pd.notna(t) else None


def collect(folder: Path, pattern: str = "*.h5", attr: str = ATTR) -> pd.DataFrame:
    rows = []
    for p in list_files(folder, pattern, recursive=True):
        try:
            uid, epics_t = read_epics(p, attr) or (None, None)
        except OSError as e:
            print(f"[skip] {p.name}: {e}")
            uid, epics_t = None, None
        rows.append(
            {
                "file": p.name,
                "uid": uid,
                "created": file_birth_time(p),
                "modified": p.stat().st_mtime,
                "epics": epics_t,
            }
        )

    df = pd.DataFrame(rows).sort_values("created").reset_index(drop=True)

    # 첫 파일 기준 상대시간 / 이전 파일과의 간격 (초)
    for col in ["created", "epics"]:
        df[f"{col}_rel_s"] = df[col] - df[col].iloc[0]
        df[f"{col}_dt_s"] = df[col].diff()

    # 촬영 시각 → 파일 생성까지 지연 (NAS 복사본이면 커질 수 있음)
    df["write_lag_s"] = df["created"] - df["epics"]

    # 사람이 읽는 시각 (KST)
    for col in ["created", "modified", "epics"]:
        df[f"{col}_kst"] = df[col].map(_fmt)
    return df


def run(folder: Path | str, pattern: str = "*.h5") -> pd.DataFrame:
    folder = Path(folder)
    df = collect(folder, pattern)

    show = [
        "file",
        "uid",
        "created_kst",
        "epics_kst",
        "created_rel_s",
        "created_dt_s",
        "epics_rel_s",
        "epics_dt_s",
        "write_lag_s",
    ]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    print(df[show].round(3).to_string(index=False))

    print("\n--- 요약 ---")
    print(f"파일 수: {len(df)}")
    print(f"첫 파일: {df['file'].iloc[0]}  ({df['created_kst'].iloc[0]})")
    print(f"마지막 : {df['file'].iloc[-1]}  ({df['created_kst'].iloc[-1]})")
    print(f"총 경과(생성기준): {df['created_rel_s'].iloc[-1]:.1f} s")
    print(
        f"평균 간격  생성: {df['created_dt_s'].mean():.3f} s / EPICS: {df['epics_dt_s'].mean():.3f} s"
    )
    print(f"쓰기 지연  평균 {df['write_lag_s'].mean():.3f} s, 최대 {df['write_lag_s'].max():.3f} s")

    out = out_dir(folder.name) / f"{folder.name}_time.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nCSV 저장: {out}")
    return df
