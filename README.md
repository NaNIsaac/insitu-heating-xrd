# insitu-heating-xrd — 승온 in-situ 2D XRD 분석

Eiger HDF5 디텍터 이미지 → 2θ 프로파일 / 시간 히트맵 / 피크 추적.

## 구조

```
main.py                  진입점 (서브커맨드 CLI)
src/insitu_xrd/
  config.py              HDF5 키 등 코드 상수 + experiments.toml [geometry]·[paths] 읽기
  style.py               그림 공통 스타일 (Origin rc, 구간·피크 색)
  geometry.py            Geometry (캘리브레이션 값, 2θ/q/d 변환)
  io.py                  이미지 / EPICS 타임스탬프 읽기
  integrate.py           RadialIntegrator (이미지 → 2θ 프로파일)
  peaks.py               피크 검출 · 가우시안 피팅
  timing.py              [timing]  파일 생성시각 vs EPICS 촬영시각
  center_profile.py      [center]  중앙 스트립 픽셀 프로파일
  tth_profile.py         [profile] 단일 이미지 2θ 프로파일
  heatmap.py             [heatmap] 폴더 전체 시간 히트맵 (+ 온도)
  temperature.py         [merge-temp, temp] 온도 로그 병합 / 시각별 온도 조회
  segments.py            온도 프로파일 → 승온/유지/하온 구간 자동 분할
  tracks.py              피크 추적 · 전이(기준선 넘나듦) 판단
  overview.py            [overview] 온도 + 히트맵 + 구간별 프로파일 한 장
  interactive.py         overview 의 인터랙티브 HTML (Plotly)
  compare.py             [compare] 실험별 X vs 온도 · T50
data/temperature/
  raw/                   온도 컨트롤러 원본 로그
  manual_data_20261004_x1.csv   병합본 (온도 조회 기본값)
out/<샘플>/              모든 결과물 (git 제외)
out/cache/               적분 결과 캐시 (지워도 다시 계산됨)
```

## 사용

```powershell
uv sync
uv run main.py                        # 인자 없이 (VS Code ▶ 포함) = compare (experiments.toml)
uv run main.py --no-show              # 위와 같되 브라우저 안 염 (명령 없이 옵션만 → 기본 명령에)
uv run main.py -h                     # 명령 목록
uv run main.py overview x1 --save     # 결과물(PNG · CSV · HTML)은 --save 일 때만 out/ 에 저장
uv run main.py image exp2.x8 200      # h5 한 장: 원본 | 2θ–χ 로 펼친 이미지 (+ 2θ 프로파일)
uv run main.py image Z:\...\IGOx8_3rd -1          # 폴더의 마지막 파일 (.h5 파일 경로도 가능)
```

저장: 기본은 그림 · 브라우저만 (HTML 은 임시 폴더). `--save` (out/) 또는 `--save-dir DIR` 을
명령 어디에나, 또는 환경변수 `INSITU_XRD_SAVE=1` / experiments.toml `[options] save = true`.

```powershell
uv run main.py profile Z:\...\IGO_x1_5th 125      # h5 한 장의 2θ 프로파일 + 피크 피팅 (번호 생략 = 0, -1 = 마지막)
uv run main.py heatmap Z:\...\IGO_x1_5th          # 폴더 → 2θ 히트맵 + 피크 면적 · FWHM (온도 없이)
uv run main.py heatmap Z:\...\IGO_x1_5th --watch 30   # 측정 중: 30 초마다 새 파일 반영
```

heatmap 맨 위 값: 피크마다 지금 면적 · 최근 5 분 변화 (전체 증가분 대비 %, `--recent`) · FWHM · 2θ.
최근 변화가 0 % 근처면 포화 = 상변이 끝. 적분 캐시는 새 파일만 이어 붙이므로 다시 돌려도 빠름.

`center` · `timing` 은 도움말에서 숨겼지만 그대로 동작.

### 온도 + 히트맵 + 온도 구간 한 장 (Origin 스타일)

```powershell
uv run main.py overview x1                                    # experiments.toml 의 [[exp]] x1 (images · recipe 그대로)
uv run main.py overview Z:\exp\hkim\261004\images\test2x1
uv run main.py overview Z:\...\testx1 Z:\...\test2x1          # 폴더 여러 개 → 시간순으로 이어 붙임
uv run main.py overview Z:\...\test2x1 data\temperature\other.csv   # 온도 로그 지정 (생략 시 자동 선택)
```

가로축 = 시간: 위에 구간 라벨 → 온도 그래프 → 2θ 히트맵(세로=2θ) 이 시간축 공유, 오른쪽에 구간 경계 프로파일 → `out/<샘플>/<샘플>_overview.png` (300 dpi).
폴더 여러 개면 결과 이름은 `testx1+test2x1`, 이미지 없는 시간은 회색 "no images".

승온 / 유지 / 하온 구간 자동 감지 (`segments.py`, 온도 로그 SV 기준, 없으면 PV):
- 구간 경계를 온도·히트맵에 세로 점선으로, 구간 라벨(`Heat 10 °C/min · 400 → 650 °C · 25 min · 16:11–16:36`)은 그래프 위 바깥에
- 오른쪽: 각 경계 시점의 2θ 프로파일을 시간순으로 쌓고 시간·온도 표시 (빔 꺼진 어두운 프레임, 이미지 없는 경계는 건너뜀)
- 온도 로그가 끊긴 시간은 이웃 램프를 연장해 추정 → 회색 점선 `est.`, 온도 앞에 `~`, CSV `estimated=True`
  (예: 15:23–15:51 은 로그 없음 → 284 °C 부터 10 °C/min 역외삽으로 ~15:25 에 승온 시작 추정)
- 폴더 여러 개면 폴더별 세기 수준을 마지막 폴더에 맞춤 (배율 출력, testx1 ≈ ×8.4)
- 출력: `<샘플>_segments.csv`, `_frames.csv` 에 `segment`, `temp_profile`, `temp_estimated` 열

**피크 추적 (상변이 추정)** (`tracks.py`):
- 프레임마다 피크를 찾고 이웃 프레임끼리 이어 붙여 추적 (열팽창으로 위치가 조금씩 움직여도 따라감)
- 히트맵 위에 피크 위치(P1, P2 …) 점선, 아래 패널에 **피크 면적 vs 시간**, 오른쪽 아래 **피크 면적 vs 온도**
  (승온 실선 / 하온 점선 / 유지 점선 → 승온·하온 차이(히스테리시스) 확인)
- 전이 시점 = 정규화 면적(최댓값=1)이 0.5 를 위로/아래로 넘는 프레임 (▲/▼ + 온도, `--level` 로 변경). 잠깐 튀는 건 무시, 이미지 공백 너머는 판단 안 함
  (예: testx1 18.40° · 21.38° 가 374–376 °C 에서 나타남, IGOx1 19.0° · 21.86° 가 395–405 °C)
- `--peaks 18.4 --peaks 21.4` 로 위치 직접 추가, `--no-track` 으로 끄기
- 출력: `<샘플>_peaks.csv` (프레임별 위치·면적·검출 여부), `<샘플>_peak_events.csv`

**FWHM 패널** (피크 면적 아래): `[peaks]` 피크마다 프레임별 FWHM vs 시간, 오른쪽에 FWHM vs 온도 (승온 실선 · 하온 점선).
좁아짐 = 결정립 성장, 넓어짐 = 변형·결함. 승온 후 냉각에서 되돌아오면 열적(가역) 변화. `_frames.csv` 에 `center_*` · `FWHM_*` 열.

**전이 온도 상자** (왼쪽 맨 아래 피크 범례 옆, 시료·시작 시각 포함 — 왼쪽 아래만 캡처해도 됨): `experiments.toml` 의 `[peaks]`·`[options]` 로 compare 와 같은 방식의 T50
(승온 중 X = 0.5) 을 피크(hkl)별로 크게 표시. 추적된 피크도 같은 hkl 이름 · T50 으로 표시됨. 결과 `<샘플>_transition.csv`.
다른 설정: `--peaks-config 파일.toml`, 끄기: `--no-transition`.

**온도 로그 자동 선택**: overview 에 CSV 를 안 주면 `data/temperature` 아래(하위 폴더 포함) 로그 중 프레임 절대시각과 가장 많이 겹치는 것을 고름.

**덮어쓰기 잔여 블록 자동 제외**: 온도 매칭은 절대시각(EPICS) 기준. 이미지 공백으로 나뉜 블록 중 온도 로그와 한 프레임도 안 겹치는 블록
(예: IGOx4 의 03:12–03:18 이전 측정 35 프레임)은 다른 측정으로 보고 그림·분석에서 뺌 (콘솔에 표시, `--keep-orphans` 로 유지).

**보기**: `--no-show` 없이 실행하면 인터랙티브 HTML 이 브라우저로 열림 (화면 폭에 맞춰져 잘리지 않음).
`--mpl` 이면 대신 matplotlib 창 두 개 — ① 시간축 그림(온도·히트맵·피크 면적 vs 시간) ② 2θ·온도축 그림(경계 프로파일·피크 면적 vs 온도) — 를 화면 크기에 맞춰 나란히 띄움.
PNG 는 항상 한 장짜리로 저장.

**인터랙티브 HTML** (`<샘플>_overview.html`, 자동 생성, 인터넷 없이 열림; `--browser` 로 `--no-show` 여도 열기, `--no-html` 로 끄기):
- 마우스 휠/드래그로 확대, 더블클릭 원래대로. 온도·히트맵 시간축 연동
- 확대하면 온도 그래프 아래 온도 눈금이 촘촘해지고(최소 3 s 간격까지), 온도 세로축도 보이는 범위에 맞춰짐
- 히트맵 위 마우스: 시간 / KST / 온도 / 2θ / 세기. 클릭하면 그 시점 프로파일을 오른쪽에 (Shift+클릭 누적)
- 히트맵 **Alt+클릭**: 그 2θ 의 세기 vs 시간을 피크 패널에 추가
- 오른쪽 범례 클릭으로 프로파일 켜고 끄기, 카메라 아이콘으로 PNG 저장
- matplotlib 창(`--mpl`)에서도 확대하면 온도 눈금이 촘촘해짐

옵션: `--clip 25 99.7` (히트맵 색 범위 = 세기 백분위, 기본값이 고대비; 예전처럼 `--clip 1 99.5`), `--cmap jet`, `--log`, `--normalize`, `--seconds`, `--bin`, `--no-show`,
`--hold-rate 1.5` (이보다 느리면 유지 [°C/min]), `--min-seg 90` (더 짧은 구간 무시 [s]),
`--no-segments`, `--no-fill` (로그 공백 추정 끄기), `--no-match` (폴더별 세기 맞춤 끄기)

### 디텍터 기하 / 빔

`experiments.toml` 의 `[geometry]` 만 고치면 모든 명령 기본값이 바뀜 (`config.py` 는 이 값을 읽기만 함, 형식 오류면 모든 명령이 멈추고 알려 줌):

```toml
[geometry]
poni_px    = [514.0, 850.0]   # PONI (x, y) [px]
sdd_mm     = 861.14           # 시료 → PONI 거리 [mm]
energy_kev = 13.0             # X-선 에너지 [keV]
alpha_deg  = 18.9             # 디텍터 암 2θ [deg]
pixel_um   = 75.0

[paths]                       # 경로 생략한 명령(timing, center, profile, heatmap)의 기본값
default_folder = 'Z:\exp\hkim\261004\images\testx1'
default_file   = 'Z:\exp\hkim\261004\images\testx1\testx1_00089.h5'
```

항목이나 파일이 없으면 기존 기본값 사용. HDF5 키·EPICS 기준시각 같은 코드 상수는 `config.py` 에 그대로.

일회성으로는 `--poni 514 850 --sdd 861.14 --energy 13 --alpha 18.9`.

### 실험 비교: 피크별 결정화 분율 X vs 온도 + T50

`experiments.toml` 에 실험마다 이미지 폴더 · 온도 로그(recipe) 의 **전체 경로**를 적고
(자동으로 찾지 않음 — 2nd 측정 · 로컬 백업 · Z: 원본이 섞여 있어서 적힌 경로만 씀):

```toml
[[exp]]
name   = "x1"
images = ['D:\USER_DATA\hkim\261004\data\IGOx1']                 # 여러 개면 시간순으로 합침
recipe = ['D:\USER_DATA\hkim\261004\data\temperature\recipe_data_20261004_x1.csv']

[[exp2]]                 # 2회차 실험 → [[exp2]], 3회차 → [[exp3]] ...
name   = "x8"
images = ['Z:\exp\hkim\261004\images\IGOx8_3rd']
recipe = ['D:\...\recipe_data_20261005_111220.csv']
```

```powershell
uv run main.py compare                  # 회차([[exp]], [[exp2]], ...) 전부 차례로, 회차마다 브라우저 창 하나
uv run main.py compare exp2             # 이 회차만 (여러 개: compare exp exp2, 예전 -g exp2 도 됨)
uv run main.py compare 다른설정.toml
```

- 아래 줄: **FWHM vs T** — 피크가 있는 프레임(면적 ≥ 최종의 20 %)마다 가우시안 + 직선 배경 피팅 (점 = 프레임, 선 = 이동 중앙값).
  프레임별 `center_<피크>` · `FWHM_<피크>` 는 `_frames.csv` 에. `_T50.csv` 의 FWHM 도 같은 피팅의 결정화 직후 값
  (피크가 비대칭·두 성분이면(예: x8) 가우시안 하나로는 정확한 폭이 아니라 '퍼진 정도' 로 봐야 함)
- 회차별 결과: `experiments.*` ([[exp]]), `experiments_exp2.*` ([[exp2]]) …
- 같은 이름(예: x8)은 모든 회차에서 같은 색
- overview 에서 회차 지정: `overview exp2.x8` (그냥 `x8` 이면 첫 회차, 여러 회차에 있으면 알려 줌)

- 피크마다 패널 하나, 가로축 = PV 온도, 범례 `x2  T50 397 °C`
- `[peaks]` 의 2θ 는 참고값: 실험마다 결정화 끝난 상태에서 그 근처(±`search`) 피크를 가우시안 피팅 →
  중심·반치폭(FWHM) → **중심 ± `fwhm_k`(1.5) × FWHM** 을 적분 (중심·FWHM 은 `_T50.csv` 에 기록)
- 각 프레임을 비정질(전이 전) 프로파일로 나눈 R = I/I₀ − 1 에서 적분 → 디텍터 가장자리 세기 감소·비정질 배경 상쇄
- **T50 (전이점)** — 데이터가 정하는 창만 써서, 범위(`t_range`) 끝 · 먼 아웃라이어 · 전이 뒤 드리프트에 둔감 (`tracks.transition_point`):
  1. 대략 정규화 → isotonic(단조) 회귀 = 누적 곡선 → 0.5 지점 = 중심 c, 0.25–0.75 지점 사이 = 폭 w
  2. 창 c ± 3w 의 바깥 꼬리(±1.5w … ±3w)에 Theil–Sen 직선 = 전·후 배경 → 국소 정규화
  3. 코어 |T − c| ≤ w 를 직선 회귀 → 0.5 교차 = T50 (± = 회귀 x절편 표준오차), c · w 갱신해 수렴까지 반복
  - X 곡선도 이 배경 직선으로 정규화 (창 밖은 창 끝 값으로 고정) → 그래프의 0.5 교차 = T50
  - `_T50.csv`: `T50`, `±`(오차), `ΔT25-75`(전이 폭). 범례 `x1  T50 400 °C  ΔT 8`
  - 창을 못 잡으면(전이 전후 데이터 부족) 예전 방식(양 끝 n_norm 프레임 정규화 → 첫 교차)으로 하고 표시
  - 검토 결과 (예전 → 지금): t_range 를 280–460 / 250–650 으로 바꿨을 때 T50 변동 최대 3.5 °C → (222) 0.3 °C 이내,
    전이 뒤 15 % 드리프트 −0.6 ~ −1.5 °C → 0 °C
- 이미지가 전이 이후부터만 있으면(전이 전 기준 없음) 그 실험은 T50 계산 안 하고 범례에 표시
- 없는 경로가 있으면 계산 전에 한꺼번에 알려 주고 멈춤
- 결과: `out/compare/<설정이름>.png / .html / _T50.csv / _frames.csv`
- 적분 결과는 `out/cache/` 에 캐시 (overview · heatmap 과 공용; 폴더 파일·기하·적분 옵션이 그대로면 재사용)
- 13 keV 에서 (222) ≈ 18.8°, (400) ≈ 21.7°

### 적분 캐시 (out/cache)

폴더의 파일 수 · 수정시각 합 · 기하 · 적분 옵션이 같으면 이전 적분 결과를 재사용 (파일이 추가·수정되면 자동으로 다시 적분).
강제로 바꾸려면 — 모든 명령, 인자 어느 위치에나:

```powershell
uv run main.py compare --refresh          # 무조건 다시 적분 + 캐시 덮어씀
uv run main.py overview x8 --no-cache     # 캐시 읽지도 쓰지도 않음
uv run main.py --cache refresh            # = --cache use | refresh | off
uv run main.py cache                      # 캐시 목록 (크기 · 날짜)
uv run main.py cache IGOx8 --clear        # 폴더 이름에 IGOx8 들어간 캐시 삭제 (생략 시 전부)
```

기본값 바꾸기: `experiments.toml` `[options] cache = "refresh"` 또는 환경변수 `INSITU_XRD_CACHE=refresh`
(우선순위: 명령 옵션 > 환경변수 > toml > use)

### 여러 스캔 전이 겹쳐 보기 (회차 상관없이)

```powershell
uv run main.py overlay exp.x1 exp2.x1 exp.x8 exp2.x8     # 재현성 · 시료 간 전이점 차이
```

- 위: 온도 vs 시간 (승온 시작 = 0 으로 맞춤) / 가운데: X vs 시간 (피크별, 숫자 = 승온 시작부터 T50 까지 분) /
  아래: X vs 온도 (숫자 = T50 °C). 전이점마다 ◆ · 점선 · 숫자 (가까우면 위아래로 엇갈림)
- X · T50 은 compare 와 같은 계산. 콘솔에 T50 · 도달 시간 표. 결과 `out/overlay/<이름들>.png / .html`

### 여러 스캔 2θ 프로파일 겹쳐 보기

```powershell
uv run main.py profiles exp2.x1 exp2.x2 x8          # experiments.toml 의 실험 이름 여러 개
uv run main.py profiles x1 x8 -T 350 -T 450 --no-end
```

- 줄 = 상태: 승온 중 각 온도 ± `--tol`(3 °C) 프레임 평균 (기본 300 · 400 · 500 · 700 °C) + `end` (냉각 후 마지막 10 프레임)
- 열 = 세기 I | 비정질로 나눈 R = I/I_비정질 − 1 (compare 와 같은 양 → 결정 피크만, 스캔 간 세기 차이 상쇄)
- 온도 로그 밖 프레임(덮어쓰기 잔여)은 제외, 디텍터 끝(거의 안 닿는 2θ)은 R 에서 뺌. `[peaks]` 참고 2θ 는 점선
- 결과: `out/profiles/<이름들>.png / .html` (HTML 은 마우스 올리면 같은 2θ 의 모든 스캔 값)

### 등온 실험 → Avrami 지수

온도 구간을 판별하지 않고 **X 변화만** 봄 → 등온·승온 어떤 측정에도 돌아감.
`experiments.toml` 에 측정마다 `[[iso]]` 하나 (측정 하나 = 비정질 새 조각):

```toml
[[iso]]
sample = "x1"            # 같은 sample 끼리 묶어 Arrhenius (등온 온도 2개 이상이면 Ea)
T_iso  = 370             # (선택) 이름 붙이기용. 실제 온도는 로그에서 계산
images = ['Z:\exp\hkim\26xxxx\images\IGOx1_iso370']
recipe = ['D:\...\recipe_data_xxxx.csv']   # (선택) 없으면 n, k 만
# start / end = "10:05" (선택, 프레임 범위),  t0 = "10:15:02" (선택, 변화 시작 직접 지정)
```

```powershell
uv run main.py avrami                       # [[iso]] 전부
uv run main.py avrami --only x1             # 이름 또는 sample 로 일부만 (여러 개: --only x1 --only x2)
uv run main.py avrami exp2                  # [[iso]] 대신 [[exp2]] — 기존 승온 데이터로 시험 (예전 -g exp2 도 됨)
```

`[[iso]]` 도 다른 회차와 똑같이 쓸 수 있음: `compare iso`, `overview iso.x1_370`, `overlay exp2.x1 iso.x1_370`,
`image iso.x1_370 -1` (name 이 없으면 이름 = sample_T_iso). 측정이 하나여도 n, k 는 나오고 Ea 만 빠짐.

측정 = 승온 → 등온 → 하온. 경고 · 판정 없이 전체 결과만 냄.

- 등온 구간 = 온도 로그에서 PV 가 등온 온도 − 3 °C 이상인 구간 (자동). 하온 시작 전에서 자름
- X(t): compare 와 같은 방식 (비정질로 나눈 R 의 FWHM 창 면적). X = 0 은 처음 프레임 (승온 초기), X = 1 은 등온 끝
- **t0** (t = 0): `t0` 지정 > 등온 시작 (설정 온도 도달) > (온도 로그 없으면) JMAK 피팅.
  JMAK X = 1 − exp(−(k(t − t0 − τ))ⁿ) 의 τ = 등온 도달 뒤 잠복시간
- **Avrami 플롯** ln[−ln(1−X)] vs ln(t − t0) 를 `fit_range`(0.15–0.85) 에서 직선 피팅 → n, k
- T_iso = 등온 구간 PV 평균
- 같은 sample 온도 2개 이상 → ln k vs 1/T → Ea (Avrami k 기준 `Ea_eV`, JMAK k 기준 `Ea_jmak_eV`)
- (400) 은 적분 창이 디텍터 끝에서 잘리므로 (222) 값을 우선
- 결과 (`--save`): `out/avrami/<설정이름>_<iso|exp>_avrami.png / .csv / _arrhenius.csv / _frames.csv`
- 합성 데이터 검증 (정답 n 2.7 · Ea 2.40 eV / 잠복 1.5 min · n 2.0 · Ea 1.80 eV): n 2.60–2.73, Ea 2.35 eV / n 1.98–2.02, Ea 1.78–1.81 eV
- 기존 승온 데이터(`-g exp`, 10 °C/min): 변화 중심 온도 x1 399 · x2 396 · x4 376 · x8 319 °C (compare T50 과 일치),
  겉보기 n 6–10, X 15–85 % 구간이 7–10 프레임뿐 → 등온 측정이 필요한 이유

### 온도

```powershell
uv run main.py merge-temp                        # raw/*.csv → 병합본 (새 로그 추가 시 재실행)
uv run main.py temp 16:55:00                     # 특정 시각(KST)의 온도
uv run main.py temp Z:\...\test2x1_00086.h5      # 특정 프레임의 온도
uv run main.py temp Z:\...\test2x1               # 폴더 전체 프레임별 온도
uv run main.py heatmap Z:\...\test2x1 --temp data\temperature\manual_data_20261004_x1.csv --at 0 --at 10 --at 16:55:00
```

- `--temp CSV` : 히트맵 옆에 온도(PV/SV) 패널
- `--at T`     : 여러 개면 `--at` 반복. 숫자 = 첫 프레임 기준 상대시간(분, `--seconds` 면 초), `HH:MM:SS` = KST 시각.
  해당 프레임을 가로선으로 표시하고, 그 시점 프로파일을 온도와 함께 오른쪽 패널에 겹쳐 그림
- 결과: `out/<샘플>/<샘플>_frames.csv` (프레임별 시각·온도), `<샘플>_tth_heatmap.png/.npz`
- 로그가 끊긴 구간(가장 가까운 기록과 5초 넘게 차이)은 온도 NaN

이미지 폴더/파일은 **실제 경로**를 그대로 적음 (자동으로 찾지 않음). overview 는 `experiments.toml` 의 실험 이름(`x1`, `exp2.x8`)도 받음.
CLI 는 [Typer](https://typer.tiangolo.com) (`src/insitu_xrd/cli.py`: 명령 = 함수 하나, 옵션 = 함수 인자). 여러 값 옵션은 반복해서 씀 (`--at 5 --at 10`).
경로를 생략하면 `experiments.toml` `[paths]` 의 `default_folder` / `default_file` 사용.
`uv run insitu-xrd <명령>` 또는 `python -m insitu_xrd <명령>` 도 동일.

파이썬에서 직접:

```python
from pathlib import Path
from insitu_xrd import heatmap, tth_profile, Geometry

geo = Geometry(alpha_deg=20.0)
tth_profile.run(Path(r"Z:\...\testx1_00089.h5"), geo, show=False)
from insitu_xrd.temperature import DEFAULT_LOG, TempLog
res = heatmap.run(heatmap.Config(folder=Path(r"Z:\...\test2x1"), geometry=geo,
                                 temp_log=DEFAULT_LOG, at=(10, "16:55:00")))
res.frames                                  # 프레임별 file / time_kst / time_rel / temp_pv / temp_sv
TempLog().at([1791100537.0])                # Unix 시각 → 온도
```
