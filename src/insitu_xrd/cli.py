"""Heating in-situ 2D XRD analysis (insitu-xrd).

Write image folders and files as their real paths (they are not searched for).
Experiment names (x1, exp2.x8) are looked up in experiments.toml.

Saving results: off by default (figures and browser only). Any command, anywhere:


  --save           save PNG, CSV and HTML under out/
  --save-dir DIR   save under DIR
  Change the default: environment variable INSITU_XRD_SAVE=1 or experiments.toml [options] save = true

Integration cache (out/cache). Any command, anywhere:


  --refresh        always integrate again and overwrite the cache
  --no-cache       neither read nor write the cache
  --cache MODE     use (default) / refresh / off
  Change the default: environment variable INSITU_XRD_CACHE or experiments.toml [options] cache = "refresh"
  List / delete: uv run main.py cache [folder name] [--clear]

Per-command options: `uv run main.py <command> --help`.
"""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from .config import DEFAULT_FILE, DEFAULT_FOLDER, SETTINGS_FILE
from .geometry import Geometry
from .temperature import DEFAULT_LOG, TEMP_DIR

app = typer.Typer(
    help=__doc__,
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)

GEO = "geometry (기본값: experiments.toml [geometry])"
_G = Geometry()
Alpha = Annotated[float, typer.Option(help="디텍터 암 2θ [deg]", rich_help_panel=GEO)]
Sdd = Annotated[float, typer.Option(help="시료→PONI 거리 [mm]", rich_help_panel=GEO)]
Poni = Annotated[
    tuple[float, float], typer.Option(metavar="X Y", help="PONI (x, y) [px]", rich_help_panel=GEO)
]
Energy = Annotated[float, typer.Option(help="X-선 에너지 [keV]", rich_help_panel=GEO)]
Workers = Annotated[
    int | None, typer.Option("-j", "--workers", help="병렬 프로세스 수 (기본 min(8, CPU))")
]
NoShow = Annotated[bool, typer.Option("--no-show", help="그림 창 / 브라우저 열지 않음")]
Pattern = Annotated[str, typer.Option(help="이미지 파일 패턴")]
Bin = Annotated[float, typer.Option("--bin", help="2θ 빈 [deg]")]
Config = Annotated[
    Path | None, typer.Argument(help="설정 TOML (생략 시 프로젝트 폴더의 experiments.toml)")
]


def _geo(alpha: float, sdd: float, poni: tuple[float, float], energy: float) -> Geometry:
    return Geometry(sdd=sdd * 1e-3, xc=poni[0], yc=poni[1], alpha_deg=alpha, energy_kev=energy)


class Reduce(str, Enum):
    mean = "mean"
    sum = "sum"


class Region(str, Enum):
    full = "full"
    strip = "strip"


MAIN, MORE, TOOL = "주요 분석", "추가 분석", "도구"


@app.command(rich_help_panel=MAIN)
def overview(
    paths: Annotated[
        list[Path],
        typer.Argument(
            help="이미지 폴더 경로 (여러 개면 시간순으로 이어 붙임) [+ 온도 로그 .csv, 생략 시 자동 선택]. "
            "experiments.toml 의 실험 이름도 가능: x1, exp2.x8"
        ),
    ],
    pattern: Pattern = "*.h5",
    workers: Workers = None,
    tth_bin: Bin = 0.02,
    normalize: Annotated[
        bool, typer.Option("--normalize", help="프레임마다 평균 세기로 나눔")
    ] = False,
    log: Annotated[bool, typer.Option("--log", help="log 색 스케일")] = False,
    seconds: Annotated[bool, typer.Option("--seconds", help="시간축 단위를 초로")] = False,
    cmap: Annotated[str, typer.Option(help="컬러맵 (예: jet, viridis)")] = "inferno",
    clip: Annotated[
        tuple[float, float],
        typer.Option(metavar="LO HI", help="히트맵 색 범위 = 세기 백분위 (대비 약하게: 1 99.5)"),
    ] = (25.0, 99.7),
    no_segments: Annotated[
        bool, typer.Option("--no-segments", help="승온/유지/하온 구간 표시 끄기")
    ] = False,
    hold_rate: Annotated[
        float, typer.Option(help="|dT/dt| 가 이보다 작으면 유지 구간 [°C/min]")
    ] = 1.5,
    min_seg: Annotated[float, typer.Option(help="이보다 짧은 구간은 무시 [s]")] = 90.0,
    no_fill: Annotated[
        bool, typer.Option("--no-fill", help="로그 끊긴 구간을 외삽으로 추정하지 않음")
    ] = False,
    no_match: Annotated[
        bool, typer.Option("--no-match", help="폴더 여러 개일 때 세기 수준 맞추지 않음")
    ] = False,
    peaks: Annotated[
        list[float] | None,
        typer.Option(
            metavar="2θ", help="추적할 2θ 위치 직접 추가 (여러 개: --peaks 18.4 --peaks 21.4)"
        ),
    ] = None,
    no_track: Annotated[bool, typer.Option("--no-track", help="피크 추적 끄기")] = False,
    level: Annotated[float, typer.Option(help="전이 기준: 정규화 면적이 이 값을 넘는 시점")] = 0.5,
    peaks_config: Annotated[
        Path | None,
        typer.Option(
            metavar="TOML", help="전이 온도 상자에 쓸 피크 설정 (생략 시 experiments.toml)"
        ),
    ] = None,
    no_transition: Annotated[
        bool, typer.Option("--no-transition", help="전이 온도 상자 끄기")
    ] = False,
    iso: Annotated[
        bool,
        typer.Option(
            "--iso",
            help="등온 모드 (승온→등온→하온): 상자에 T50 대신 t50 · n · k, "
            "등온 시작·끝 표시. [[iso]] 실험 이름이면 자동",
        ),
    ] = False,
    keep_orphans: Annotated[
        bool, typer.Option("--keep-orphans", help="온도 로그와 시간이 안 겹치는 이미지 블록도 표시")
    ] = False,
    browser: Annotated[
        bool, typer.Option("--browser", help="--no-show 여도 HTML 을 브라우저로")
    ] = False,
    mpl: Annotated[bool, typer.Option("--mpl", help="브라우저 대신 matplotlib 창 두 개")] = False,
    no_html: Annotated[bool, typer.Option("--no-html", help="인터랙티브 HTML 만들지 않음")] = False,
    no_show: NoShow = False,
    alpha: Alpha = _G.alpha_deg,
    sdd: Sdd = _G.sdd * 1e3,
    poni: Poni = (_G.xc, _G.yc),
    energy: Energy = _G.energy_kev,
):
    """Folder(s) plus temperature log to one figure: temperature, 2θ heatmap, segments and peak tracking (Origin style plus HTML)."""
    from . import heatmap as hm
    from . import overview as ov
    from .integrate import ProfileOptions
    from .segments import SegmentOptions
    from .tracks import TrackOptions

    folders, csvs, auto_iso = _overview_inputs(paths)
    cfg = hm.Config(
        folder=folders[0],
        pattern=pattern,
        geometry=_geo(alpha, sdd, poni, energy),
        workers=workers,
        profile=ProfileOptions(tth_bin=tth_bin, normalize=normalize),
        plot=hm.PlotOptions(
            log_color=log,
            time_unit="s" if seconds else "min",
            cmap=cmap,
            show=not no_show,
            clip=clip,
        ),
        temp_log=csvs or None,  # without a log, overview picks one automatically
    )
    ov.run(
        cfg,
        None
        if no_segments
        else SegmentOptions(hold_rate=hold_rate, min_duration=min_seg, fill_gaps=not no_fill),
        extra=folders[1:],
        match=not no_match,
        html=not no_html,
        browser=browser,
        mpl=mpl,
        keep_orphans=keep_orphans,
        peaks_cfg=None if no_transition else (peaks_config or SETTINGS_FILE),
        track_opt=None if no_track else TrackOptions(manual=tuple(peaks or ()), level=level),
        iso=iso or auto_iso,
    )


@app.command(rich_help_panel=MAIN)
def compare(
    targets: Annotated[
        list[str] | None,
        typer.Argument(
            help="회차 이름 (예: exp2 exp3). 생략 시 [[exp]], [[exp2]] ..., [[iso]] 등 전부 차례로. "
            ".toml 을 주면 그 설정 파일 사용"
        ),
    ] = None,
    group: Annotated[
        list[str] | None,
        typer.Option("-g", "--group", help="회차 이름 (위치 인자와 같음, 예전 방식)"),
    ] = None,
    workers: Workers = None,
    no_show: NoShow = False,
    alpha: Alpha = _G.alpha_deg,
    sdd: Sdd = _G.sdd * 1e3,
    poni: Poni = (_G.xc, _G.yc),
    energy: Energy = _G.energy_kev,
):
    """Per run: crystallized fraction X vs temperature for each peak, with T50 (one browser window per run)."""
    from . import compare as cmp

    targets = targets or []
    tomls = [Path(t) for t in targets if t.lower().endswith(".toml")]
    groups = [t for t in targets if not t.lower().endswith(".toml")] + (group or [])
    if len(tomls) > 1:
        raise SystemExit("❌ 설정 TOML 은 하나만")
    cmp.run(
        tomls[0] if tomls else SETTINGS_FILE,
        _geo(alpha, sdd, poni, energy),
        workers,
        show=not no_show,
        groups=groups or None,
    )


@app.command(rich_help_panel=MAIN, context_settings={"ignore_unknown_options": True})  # -1 = last
def image(
    target: Annotated[
        str,
        typer.Argument(help="이미지 폴더 · .h5 파일 · experiments.toml 실험 이름 (x1, exp2.x8)"),
    ],
    index: Annotated[
        int,
        typer.Argument(
            help="폴더 안 몇 번째 파일 (0 부터, 음수는 뒤에서: -1 = 마지막). .h5 를 주면 무시"
        ),
    ] = 0,
    tth_bin: Annotated[float, typer.Option("--bin", help="펼친 이미지 2θ 칸 [deg]")] = 0.01,
    chi_bin: Annotated[float, typer.Option(help="펼친 이미지 χ 칸 [deg]")] = 0.1,
    log: Annotated[bool, typer.Option("--log", help="log 색 스케일")] = False,
    clip: Annotated[
        tuple[float, float], typer.Option(metavar="LO HI", help="색 범위 = 세기 백분위")
    ] = (1.0, 99.7),
    cmap: Annotated[str, typer.Option(help="컬러맵")] = "inferno",
    no_show: NoShow = False,
    alpha: Alpha = _G.alpha_deg,
    sdd: Sdd = _G.sdd * 1e3,
    poni: Poni = (_G.xc, _G.yc),
    energy: Energy = _G.energy_kev,
):
    """One h5 image: raw detector image | image unwrapped to 2θ–χ plus the 2θ profile (hover shows 2θ and χ)."""
    from . import image as im

    im.run(
        target,
        index,
        _geo(alpha, sdd, poni, energy),
        tth_bin=tth_bin,
        chi_bin=chi_bin,
        log=log,
        clip=clip,
        cmap=cmap,
        show=not no_show,
    )


@app.command(rich_help_panel=MAIN, context_settings={"ignore_unknown_options": True})  # -1 = last
def profile(
    target: Annotated[
        str,
        typer.Argument(help="이미지 폴더 · .h5 파일 · experiments.toml 실험 이름 (x1, exp2.x8)"),
    ] = str(DEFAULT_FILE),
    index: Annotated[
        int,
        typer.Argument(
            help="폴더 안 몇 번째 파일 (0 부터, 음수는 뒤에서: -1 = 마지막). .h5 를 주면 무시"
        ),
    ] = 0,
    width: Annotated[int, typer.Option(help="스트립 폭 [px]")] = 10,
    tth_bin: Bin = 0.02,
    smooth: Annotated[int, typer.Option(help="Savitzky–Golay 창 (0=끔)")] = 7,
    prominence: Annotated[float | None, typer.Option(help="피크 기준 (생략 시 자동)")] = None,
    min_distance: Annotated[float, typer.Option(help="피크 간 최소 간격 [deg]")] = 0.15,
    no_show: NoShow = False,
    alpha: Alpha = _G.alpha_deg,
    sdd: Sdd = _G.sdd * 1e3,
    poni: Poni = (_G.xc, _G.yc),
    energy: Energy = _G.energy_kev,
):
    """One h5 image to a 2θ profile (center strip and full-ring average) with peak fits (position, FWHM, d)."""
    from . import tth_profile
    from .image import resolve

    geo = _geo(alpha, sdd, poni, energy)
    file, i, n, _ = resolve(target, index)
    print(f"🖼️  {file}  [{i} / {n - 1}]")
    tth_profile.run(
        file,
        geo,
        strip_width=width,
        tth_bin=tth_bin,
        smooth=smooth,
        prominence=prominence,
        min_distance=min_distance,
        show=not no_show,
        sub=f"[{i} / {n - 1}]",
    )


@app.command(rich_help_panel=MAIN)
def heatmap(
    target: Annotated[
        str,
        typer.Argument(help="이미지 폴더 또는 experiments.toml 실험 이름 (폴더 여러 개면 마지막)"),
    ] = str(DEFAULT_FOLDER),
    watch: Annotated[
        float,
        typer.Option(metavar="SEC", help="SEC 초마다 새 파일 반영 (측정 중 확인). 0 = 한 번만"),
    ] = 0,
    recent: Annotated[
        float,
        typer.Option(
            metavar="MIN", help="최근 MIN 분 동안의 면적 변화 % 표시 (0 근처 = 포화 = 상변이 끝)"
        ),
    ] = 5.0,
    cmap: Annotated[str, typer.Option(help="컬러맵")] = "inferno",
    clip: Annotated[
        tuple[float, float], typer.Option(metavar="LO HI", help="히트맵 색 범위 = 세기 백분위")
    ] = (25.0, 99.7),
    workers: Workers = None,
    no_show: NoShow = False,
    alpha: Alpha = _G.alpha_deg,
    sdd: Sdd = _G.sdd * 1e3,
    poni: Poni = (_G.xc, _G.yc),
    energy: Energy = _G.energy_kev,
):
    """Folder to a 2θ time heatmap plus phase-transition graphs (peak area, FWHM, current value and recent change %); needs no temperature and works during a measurement."""
    from . import live

    folder = Path(target)
    if not folder.is_dir():
        from .compare import exp_names, find_exp

        e = find_exp(target)
        if e is None:
            raise SystemExit(
                f"❌ 없음: {target}  (이미지 폴더 · {SETTINGS_FILE.name} 의 실험 이름)\n"
                f"   있는 이름: {exp_names()}"
            )
        folder = e.images[-1]
    live.run(
        folder,
        _geo(alpha, sdd, poni, energy),
        workers,
        watch=watch,
        recent=recent,
        show=not no_show,
        clip=clip,
        cmap=cmap,
    )


@app.command(rich_help_panel=MAIN)
def overlay(
    names: Annotated[
        list[str],
        typer.Argument(
            help="experiments.toml 의 실험 이름 여러 개, 회차 상관없이 (예: exp.x1 exp2.x1 x8)"
        ),
    ],
    config: Annotated[
        Path | None, typer.Option(help="설정 TOML (생략 시 experiments.toml)")
    ] = None,
    workers: Workers = None,
    no_show: NoShow = False,
    alpha: Alpha = _G.alpha_deg,
    sdd: Sdd = _G.sdd * 1e3,
    poni: Poni = (_G.xc, _G.yc),
    energy: Energy = _G.energy_kev,
):
    """Overlay the transitions of several scans: temperature, X vs time (from the start of heating), X vs temperature, transition points marked with diamonds."""
    from . import overlay as ov

    ov.run(
        names, config or SETTINGS_FILE, _geo(alpha, sdd, poni, energy), workers, show=not no_show
    )


@app.command(rich_help_panel=MORE)
def avrami(
    targets: Annotated[
        list[str] | None,
        typer.Argument(
            help="분석할 회차 (여러 개 가능: iso iso2). 생략 시 iso 로 시작하는 회차 전부 한꺼번에 "
            "(온도가 다른 회차끼리도 Ea). exp2 처럼 승온 회차도 가능 (시험용). .toml 을 주면 그 설정 파일"
        ),
    ] = None,
    group: Annotated[
        str | None, typer.Option("-g", "--group", help="회차 이름 (위치 인자와 같음, 예전 방식)")
    ] = None,
    only: Annotated[
        list[str] | None,
        typer.Option(metavar="NAME", help="이 이름(또는 sample)만 (여러 개: --only x1 --only x2)"),
    ] = None,
    workers: Workers = None,
    no_show: NoShow = False,
    alpha: Alpha = _G.alpha_deg,
    sdd: Sdd = _G.sdd * 1e3,
    poni: Poni = (_G.xc, _G.yc),
    energy: Energy = _G.energy_kev,
):
    """From the change of X(t): Avrami exponent n, rate constant k and, with two or more isothermal temperatures, the activation energy Ea."""
    from . import avrami as av

    targets = targets or []
    tomls = [Path(t) for t in targets if t.lower().endswith(".toml")]
    groups = [t for t in targets if not t.lower().endswith(".toml")] + ([group] if group else [])
    if len(tomls) > 1:
        raise SystemExit("❌ 설정 TOML · 회차는 하나씩 (예: avrami iso)")
    av.run(
        tomls[0] if tomls else SETTINGS_FILE,
        _geo(alpha, sdd, poni, energy),
        workers,
        only=only,
        show=not no_show,
        section=groups or None,
    )


@app.command(rich_help_panel=MORE)
def profiles(
    names: Annotated[
        list[str],
        typer.Argument(help="experiments.toml 의 실험 이름 여러 개 (예: exp2.x1 exp2.x2 x8)"),
    ],
    T: Annotated[
        list[float] | None,
        typer.Option(
            "-T", "--temp", help="볼 온도 [°C] (여러 개: -T 350 -T 450). 생략 시 300 400 500 700"
        ),
    ] = None,
    no_end: Annotated[
        bool, typer.Option("--no-end", help="냉각 후 마지막 프레임(end) 줄 빼기")
    ] = False,
    tol: Annotated[float, typer.Option(help="온도 ± 이 범위의 프레임 평균 [°C]")] = 3.0,
    config: Annotated[
        Path | None, typer.Option(help="설정 TOML (생략 시 experiments.toml)")
    ] = None,
    workers: Workers = None,
    no_show: NoShow = False,
    alpha: Alpha = _G.alpha_deg,
    sdd: Sdd = _G.sdd * 1e3,
    poni: Poni = (_G.xc, _G.yc),
    energy: Energy = _G.energy_kev,
):
    """Overlay the 2θ profiles of several scans at the same temperature (intensity I and amorphous-normalized R)."""
    from . import profiles as pf

    pf.run(
        names,
        T or [300, 400, 500, 700],
        end=not no_end,
        config=config or SETTINGS_FILE,
        geo=_geo(alpha, sdd, poni, energy),
        workers=workers,
        tol=tol,
        show=not no_show,
    )


# legacy single-folder / single-image commands: hidden from help, still working
@app.command(hidden=True)
def center(
    file: Annotated[Path, typer.Argument(help=".h5 이미지")] = DEFAULT_FILE,
    width: Annotated[int, typer.Option(help="스트립 폭 [px]")] = 100,
    reduce: Reduce = Reduce.mean,
    prominence: Annotated[float, typer.Option(help="음수면 자동 (3σ)")] = 1.0,
    min_distance: Annotated[int, typer.Option(help="피크 간 최소 간격 [px]")] = 10,
    no_show: NoShow = False,
):
    """Horizontal center strip of one image to a vertical pixel profile with peaks."""
    from . import center_profile

    center_profile.run(
        file,
        width,
        reduce.value,
        prominence=None if prominence < 0 else prominence,
        min_distance=min_distance,
        show=not no_show,
    )


@app.command(hidden=True)
def timing(
    folder: Annotated[Path, typer.Argument(help="이미지 폴더")] = DEFAULT_FOLDER,
    pattern: Pattern = "*.h5",
):
    """File creation time vs EPICS acquisition time, written to CSV."""
    from . import timing as tm

    tm.run(folder, pattern)


@app.command(rich_help_panel=TOOL)
def cache(
    match: Annotated[
        str | None,
        typer.Argument(help="폴더 이름에 이 글자가 들어간 캐시만 (예: IGOx8). 생략 시 전부"),
    ] = None,
    clear: Annotated[
        bool, typer.Option("--clear", help="해당 캐시 파일 삭제 (다음 실행 때 다시 적분)")
    ] = False,
):
    """List or delete the integration cache (out/cache). Every command also accepts --refresh, --no-cache and --cache MODE."""
    from datetime import datetime

    from .config import CACHE_DIR
    from .heatmap import _cache_mode, cache_files

    fs = cache_files(match)
    print(
        f"🗂️  {CACHE_DIR}  — 현재 방식: {_cache_mode}  ({len(fs)} 파일, "
        f"{sum(f.stat().st_size for f in fs) / 1e6:.1f} MB)"
    )
    for f in fs:
        st = f.stat()
        print(
            f"   {f.name:40s} {st.st_size / 1e6:7.1f} MB  {datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M}"
        )
    if clear and fs:
        for f in fs:
            f.unlink()
        print(f"🧹 {len(fs)} 개 삭제")


@app.command(rich_help_panel=TOOL)
def temp(
    targets: Annotated[
        list[str],
        typer.Argument(help="'HH:MM:SS' 또는 'YYYY-MM-DD HH:MM:SS' (KST), .h5 파일, h5 폴더"),
    ],
    csv: Annotated[Path, typer.Option(help="온도 로그 CSV")] = DEFAULT_LOG,
    max_gap: Annotated[float, typer.Option(help="로그 샘플과 이보다 멀면 NaN [s]")] = 5.0,
):
    """Look up the temperature at a given time, h5 frame or folder."""
    from . import temperature

    temperature.lookup(targets, csv, max_gap)


@app.command("merge-temp", rich_help_panel=TOOL)
def merge_temp(
    csvs: Annotated[
        list[Path] | None, typer.Argument(help=f"합칠 CSV (생략 시 {TEMP_DIR / 'raw'}\\*.csv 전부)")
    ] = None,
    out: Annotated[Path, typer.Option("-o", "--out", help="저장할 파일")] = DEFAULT_LOG,
):
    """Merge several temperature log CSVs into one."""
    from . import temperature

    csvs = csvs or sorted((TEMP_DIR / "raw").glob("*.csv"))
    df = temperature.merge(csvs, out)
    print(
        f"{len(csvs)} files → {out}  ({len(df)} rows, "
        f"{df['Timestamp'].iloc[0][:19]} ~ {df['Timestamp'].iloc[-1][:19]})"
    )


def _overview_inputs(paths: list[Path]) -> tuple[list[Path], list[Path], bool]:
    """overview arguments to (image folders, temperature CSVs). A path that does not exist but is an
    experiments.toml experiment name (x1, exp2.x8) uses that experiment's images and recipe as is."""
    from .compare import exp_names, find_exp

    folders, csvs, iso = [], [], False
    for p in paths:
        if p.suffix.lower() == ".csv":
            csvs.append(p)
        elif p.exists():
            folders.append(p)
        elif (e := find_exp(str(p))) is not None:
            print(f"📋 {SETTINGS_FILE.name} {p}: {', '.join(map(str, e.images))}")
            folders += e.images
            iso = iso or e.group.startswith("iso")
            csvs += e.recipe
        else:
            raise SystemExit(
                f"❌ 폴더 없음: {p}  (전체 경로 또는 {SETTINGS_FILE.name} 의 실험 이름)\n"
                f"   있는 이름: {exp_names()}"
            )
    if not folders:
        raise SystemExit("이미지 폴더를 하나 이상 지정하세요")
    return folders, csvs, iso


def _num_or_str(v: str) -> float | str:
    try:
        return float(v)
    except ValueError:
        return v


CACHE_FLAGS = {"--refresh": "refresh", "--no-cache": "off"}


def _pop_global_args(argv: list[str]) -> tuple[list[str], str | None, Path | None]:
    """Pull the options common to all commands out of the arguments, wherever they appear.
    Cache: --refresh / --no-cache / --cache use|refresh|off.  Saving: --save / --save-dir DIR."""
    from .config import OUT_DIR

    out, mode, save, it = [], None, None, iter(argv)
    for a in it:
        if a in CACHE_FLAGS:
            mode = CACHE_FLAGS[a]
        elif a == "--cache":
            mode = next(it, None)
        elif a.startswith("--cache="):
            mode = a.split("=", 1)[1]
        elif a == "--save":
            save = OUT_DIR
        elif a == "--save-dir":
            save = Path(next(it, OUT_DIR))
        elif a.startswith("--save-dir="):
            save = Path(a.split("=", 1)[1])
        else:
            out.append(a)
    return out, mode, save


def main(argv: list[str] | None = None) -> None:
    import sys

    from .config import set_save
    from .heatmap import set_cache_mode

    argv, mode, save = _pop_global_args(list(sys.argv[1:] if argv is None else argv))
    set_cache_mode(mode)
    if save is not None:
        set_save(save)
    app(args=argv, prog_name="insitu-xrd")


if __name__ == "__main__":
    main()
