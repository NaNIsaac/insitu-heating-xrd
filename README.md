# insitu-heating-xrd — In-situ heating 2D XRD analysis

English · [한국어](README.ko.md)

In-situ heating 2D XRD analysis: Eiger HDF5 detector images to 2θ profiles, time heatmaps and peak tracking.

An analysis tool that combines HDF5 images taken continuously with an Eiger 2D detector while a sample is heated
with the temperature controller log, to follow the amorphous → crystalline transition on the time and temperature axes.

- Image folder → 2θ profiles · time heatmap (updated during a measurement with `--watch`)
- Automatic split into heating / hold / cooling segments from the temperature log, peak tracking, peak area · FWHM vs temperature
- Per-experiment crystallized fraction X vs temperature and comparison of the transition temperature T50, Avrami exponent for isothermal measurements

```powershell
uv run main.py heatmap Z:\...\IGO_x1_5th      # one folder → 2θ heatmap + peak area · FWHM
uv run main.py overview x1                    # experiment x1 in experiments.toml → temperature + heatmap + segments on one figure
```

## Installation

Python 3.14 or later, [uv](https://docs.astral.sh/uv/).

```powershell
git clone https://github.com/NaNIsaac/insitu-heating-xrd.git
cd insitu-heating-xrd
uv sync
```

The 2θ · q · d conversion and the per-pixel 2θ map use [QoraGeom](https://github.com/NaNIsaac/QoraGeom) (`qorageom.geometry`),
fetched from GitHub (`uv sync` installs it together).

Tests:

```powershell
uv run pytest
```

## Usage

```powershell
uv run main.py                        # no arguments = compare (experiments.toml)
uv run main.py --no-show              # same as above, but does not open the browser
uv run main.py -h                     # list of commands
uv run main.py overview x1 --save     # outputs (PNG · CSV · HTML) are written to out/ only with --save
uv run main.py image exp2.x8 200      # one h5 frame: raw | image unrolled to 2θ–χ (+ 2θ profile)
uv run main.py image Z:\...\IGOx8_3rd -1          # last file in a folder (an .h5 file path also works)
```

Saving: by default only figures and the browser (HTML goes to a temporary folder). Use `--save` (out/) or `--save-dir DIR`
anywhere in a command, or the environment variable `INSITU_XRD_SAVE=1` / `[options] save = true` in experiments.toml.

```powershell
uv run main.py profile Z:\...\IGO_x1_5th 125      # 2θ profile of one h5 frame + peak fit (no number = 0, -1 = last)
uv run main.py heatmap Z:\...\IGO_x1_5th          # folder (or experiment name) → 2θ heatmap + peak area · FWHM (no temperature)
uv run main.py heatmap Z:\...\IGO_x1_5th --watch 30   # during a measurement: pick up new files every 30 s
```

Values at the top of the heatmap: for each peak, the current area · the change over the last 5 min (% of the total increase, `--recent`) · FWHM · 2θ.
A recent change near 0 % means saturation = phase transition finished. The integration cache only appends new files, so rerunning is fast.

### Temperature + heatmap + temperature segments on one figure (Origin style)

```powershell
uv run main.py overview x1                                    # [[exp]] x1 in experiments.toml (images · recipe as written)
uv run main.py overview Z:\exp\hkim\261004\images\test2x1
uv run main.py overview Z:\...\testx1 Z:\...\test2x1          # several folders → joined in time order
uv run main.py overview Z:\...\test2x1 data\temperature\other.csv   # specify the temperature log (chosen automatically if omitted)
```

Horizontal axis = time: segment labels on top → temperature graph → 2θ heatmap (vertical = 2θ) share the time axis, with the segment-boundary profiles on the right → `out/<sample>/<sample>_overview.png` (300 dpi).
With several folders the result name is `testx1+test2x1`, and time without images is shown as a gray "no images".

Automatic detection of heating / hold / cooling segments (`segments.py`, based on the SV of the temperature log, PV if absent):
- Segment boundaries are drawn as vertical dotted lines on the temperature graph and the heatmap, and the segment label (`Heat 10 °C/min · 400 → 650 °C · 25 min · 16:11–16:36`) is placed outside, above the graph
- Right: the 2θ profiles at each boundary time are stacked in time order with time and temperature shown (dark frames with the beam off; boundaries without images are skipped)
- Time where the temperature log is interrupted is estimated by extending the neighboring ramp → gray dotted line `est.`, `~` before the temperature, `estimated=True` in the CSV
  (e.g. 15:23–15:51 has no log → heating start estimated at ~15:25 by extrapolating backward at 10 °C/min from 284 °C)
- With several folders, the intensity level of each folder is matched to the last folder (the scale factor is printed, testx1 ≈ ×8.4)
- Output: `<sample>_segments.csv`, and columns `segment`, `temp_profile`, `temp_estimated` in `_frames.csv`

**Peak tracking (phase transition estimate)** (`tracks.py`):
- Peaks are found in every frame and linked between neighboring frames to form tracks (followed even when thermal expansion shifts the position gradually)
- Dotted lines for the peak positions (P1, P2 …) over the heatmap, **peak area vs time** in the panel below, **peak area vs temperature** at the lower right
  (heating solid / cooling dotted / hold dotted → check the heating–cooling difference (hysteresis))
- Transition time = the frame where the normalized area (maximum = 1) crosses 0.5 upward/downward (▲/▼ + temperature, change with `--level`). Brief spikes are ignored, and no decision is made across an image gap
  (e.g. testx1 18.40° · 21.38° appear at 374–376 °C, IGOx1 19.0° · 21.86° at 395–405 °C)
- Add positions manually with `--peaks 18.4 --peaks 21.4`, turn tracking off with `--no-track`
- Output: `<sample>_peaks.csv` (per-frame position · area · detected or not), `<sample>_peak_events.csv`

**FWHM panel** (below the peak area): for each `[peaks]` peak, per-frame FWHM vs time, and FWHM vs temperature on the right (heating solid · cooling dotted).
Narrowing = grain growth, broadening = strain · defects. If it returns on cooling after heating, it is a thermal (reversible) change. `_frames.csv` gets `center_*` · `FWHM_*` columns.

**Transition temperature box** (to the right of the peak legend at the lower left, includes the sample · start time): using `[peaks]` and `[options]` of `experiments.toml`, T50
(X = 0.5 during heating) is shown large for each peak (hkl), computed the same way as in compare. Tracked peaks are also shown with the same hkl name · T50. Result: `<sample>_transition.csv`.
Other settings: `--peaks-config file.toml`, turn off: `--no-transition`.

**Automatic temperature log selection**: if no CSV is given to overview, it picks, among the logs under `data/temperature` (including subfolders), the one that overlaps the absolute frame times the most.

**Automatic exclusion of leftover overwritten blocks**: temperature matching is based on absolute time (EPICS). Among the blocks separated by image gaps, a block that does not overlap the temperature log by even one frame
(e.g. the 35 frames of an earlier measurement before 03:12–03:18 in IGOx4) is regarded as a different measurement and left out of the figure and the analysis (printed on the console, keep with `--keep-orphans`).

**Viewing**: running without `--no-show` opens an interactive HTML in the browser (fitted to the window width, not cropped).
With `--mpl`, two matplotlib windows are shown instead — ① the time-axis figure (temperature · heatmap · peak area vs time) ② the 2θ · temperature-axis figure (boundary profiles · peak area vs temperature) — placed side by side to fit the screen size.
The PNG is always saved as a single figure.

**Interactive HTML** (`<sample>_overview.html`, generated automatically, opens without internet; `--browser` opens it even with `--no-show`, `--no-html` turns it off):
- Zoom with the mouse wheel/drag, double-click to reset. The temperature and heatmap time axes are linked
- When zoomed in, the temperature ticks below the temperature graph become denser (down to a 3 s interval), and the temperature vertical axis also fits the visible range
- Mouse over the heatmap: time / KST / temperature / 2θ / intensity. Click to show the profile at that time on the right (Shift+click accumulates)
- Heatmap **Alt+click**: adds the intensity vs time of that 2θ to the peak panel
- Click the legend on the right to toggle profiles, camera icon saves a PNG
- In the matplotlib window (`--mpl`) too, the temperature ticks become denser when zoomed in

Options: `--clip 25 99.7` (heatmap color range = intensity percentiles, the default is high contrast), `--cmap jet`, `--log`, `--normalize`, `--seconds`, `--bin`, `--no-show`,
`--hold-rate 1.5` (hold if slower than this [°C/min]), `--min-seg 90` (ignore segments shorter than this [s]),
`--no-segments`, `--no-fill` (turn off estimation of log gaps), `--no-match` (turn off per-folder intensity matching)

### Detector geometry / beam

Editing only `[geometry]` in `experiments.toml` changes the defaults of every command (a malformed entry stops all commands and reports it):

```toml
[geometry]
poni_px    = [514.0, 850.0]   # PONI (x, y) [px]
sdd_mm     = 861.14           # sample → PONI distance [mm]
energy_kev = 13.0             # X-ray energy [keV]
alpha_deg  = 18.9             # detector arm 2θ [deg]
pixel_um   = 75.0

[paths]                       # defaults for commands with the path omitted (timing, center, profile, heatmap)
default_folder = 'Z:\exp\hkim\261004\images\testx1'
default_file   = 'Z:\exp\hkim\261004\images\testx1\testx1_00089.h5'
```

If an entry or the file is missing, the previous defaults are used.

For a one-off use, `--poni 514 850 --sdd 861.14 --energy 13 --alpha 18.9`.

### Experiment comparison: crystallized fraction X vs temperature per peak + T50

In `experiments.toml`, write the **full paths** of the image folder · temperature log (recipe) for each experiment:

```toml
[[exp]]
name   = "x1"
images = ['D:\USER_DATA\hkim\261004\data\IGOx1']                 # several entries are joined in time order
recipe = ['D:\USER_DATA\hkim\261004\data\temperature\recipe_data_20261004_x1.csv']

[[exp2]]                 # second round of experiments → [[exp2]], third → [[exp3]] ...
name   = "x8"
images = ['Z:\exp\hkim\261004\images\IGOx8_3rd']
recipe = ['D:\...\recipe_data_20261005_111220.csv']
```

```powershell
uv run main.py compare                  # all rounds ([[exp]], [[exp2]], ...) in turn, one browser window per round
uv run main.py compare exp2             # only this round (several: compare exp exp2)
uv run main.py compare other_config.toml
```

- Bottom row: **FWHM vs T** — for each frame that has the peak (area ≥ 20 % of the final), a Gaussian + linear background fit (point = frame, line = moving median).
  The per-frame `center_<peak>` · `FWHM_<peak>` are in `_frames.csv`. The FWHM in `_T50.csv` is also the value from the same fit right after crystallization
  (if a peak is asymmetric or has two components (e.g. x8), a single Gaussian gives not an exact width but should be read as a measure of 'spread')
- Per-round results: `experiments.*` ([[exp]]), `experiments_exp2.*` ([[exp2]]) …
- The same name (e.g. x8) has the same color in all rounds
- To pick a round in overview: `overview exp2.x8` (plain `x8` means the first round; if it exists in several rounds, it tells you)

- One panel per peak, horizontal axis = PV temperature, legend `x2  T50 397 °C`
- The 2θ in `[peaks]` is a reference value: for each experiment, the peak near it (±`search`) in the fully crystallized state is Gaussian-fitted →
  center · full width at half maximum (FWHM) → **center ± `fwhm_k`(1.5) × FWHM** is integrated (the center · FWHM are recorded in `_T50.csv`)
- R = I/I₀ − 1, with each frame divided by the amorphous (pre-transition) profile, is integrated → the intensity drop at the detector edge and the amorphous background cancel
- **T50 (transition point)** — uses only the window determined by the data, so it is insensitive to the ends of the range (`t_range`) · distant outliers · drift after the transition (`tracks.transition_point`):
  1. Rough normalization → isotonic (monotonic) regression = cumulative curve → the 0.5 point = center c, the distance between the 0.25–0.75 points = width w
  2. A Theil–Sen line on the outer tails (±1.5w … ±3w) of the window c ± 3w = pre/post background → local normalization
  3. Linear regression on the core |T − c| ≤ w → the 0.5 crossing = T50 (± = standard error of the regression x-intercept), update c · w and repeat until convergence
  - The X curve is also normalized with this background line (outside the window it is held at the window-edge value) → the 0.5 crossing in the graph = T50
  - `_T50.csv`: `T50`, `±` (error), `ΔT25-75` (transition width). Legend `x1  T50 400 °C  ΔT 8`
  - If no window can be found (too little data before/after the transition), the old method is used (normalize with n_norm frames at both ends → first crossing) and this is indicated
- If the images exist only after the transition (no pre-transition reference), T50 is not computed for that experiment and this is indicated in the legend
- If some paths do not exist, all of them are reported before the calculation and it stops
- Result: `out/compare/<config name>.png / .html / _T50.csv / _frames.csv`
- Integration results are cached in `out/cache/` (shared with overview · heatmap; reused if the folder files · geometry · integration options are unchanged)
- At 13 keV, (222) ≈ 18.8°, (400) ≈ 21.7°

### Integration cache (out/cache)

If the number of files in the folder · the sum of modification times · geometry · integration options are the same, the previous integration result is reused (when a file is added or modified, it is integrated again automatically).
To force a change — for every command, anywhere among the arguments:

```powershell
uv run main.py compare --refresh          # always integrate again + overwrite the cache
uv run main.py overview x8 --no-cache     # neither read nor write the cache
uv run main.py --cache refresh            # = --cache use | refresh | off
uv run main.py cache                      # list caches (size · date)
uv run main.py cache IGOx8 --clear        # delete caches whose folder name contains IGOx8 (all if omitted)
```

To change the default: `[options] cache = "refresh"` in `experiments.toml` or the environment variable `INSITU_XRD_CACHE=refresh`
(priority: command option > environment variable > toml > use)

### Overlay transitions of several scans (regardless of round)

```powershell
uv run main.py overlay exp.x1 exp2.x1 exp.x8 exp2.x8     # reproducibility · difference in transition point between samples
```

- Top: temperature vs time (aligned so that heating start = 0) / middle: X vs time (per peak, number = minutes from heating start to T50) /
  bottom: X vs temperature (number = T50 °C). Each transition point gets a ◆ · dotted line · number (staggered up and down when close)
- X · T50 are computed the same way as in compare. A table of T50 · time to reach it is printed on the console. Result `out/overlay/<names>.png / .html`

### Overlay 2θ profiles of several scans

```powershell
uv run main.py profiles exp2.x1 exp2.x2 x8          # several experiment names from experiments.toml
uv run main.py profiles x1 x8 -T 350 -T 450 --no-end
```

- Rows = state: mean of the frames within ± `--tol` (3 °C) of each temperature during heating (default 300 · 400 · 500 · 700 °C) + `end` (last 10 frames after cooling)
- Columns = intensity I | R = I/I_amorphous − 1, divided by the amorphous profile (the same quantity as in compare → only crystalline peaks, intensity differences between scans cancel)
- Frames outside the temperature log (leftover overwritten ones) are excluded, and the detector end (2θ that is barely reached) is left out of R. The `[peaks]` reference 2θ is shown as a dotted line
- Result: `out/profiles/<names>.png / .html` (in the HTML, hovering shows the values of all scans at the same 2θ)

### Isothermal experiment → Avrami exponent

It looks **only at the change in X** without identifying temperature segments → works on any measurement, isothermal or heating.
In `experiments.toml`, one `[[iso]]` per measurement (one measurement = a fresh piece of amorphous sample):

```toml
[[iso]]
sample = "x1"            # entries with the same sample are grouped for Arrhenius (Ea if there are 2 or more isothermal temperatures)
T_iso  = 370             # (optional) for naming only. The actual temperature is computed from the log
images = ['Z:\exp\hkim\26xxxx\images\IGOx1_iso370']
recipe = ['D:\...\recipe_data_xxxx.csv']   # (optional) without it, only n, k
# start / end = "10:05" (optional, frame range),  t0 = "10:15:02" (optional, specify the start of the change directly)
```

```powershell
uv run main.py avrami                       # all [[iso]]
uv run main.py avrami --only x1             # only some, by name or sample (several: --only x1 --only x2)
uv run main.py avrami exp2                  # [[exp2]] instead of [[iso]] — try it on existing heating data
```

`[[iso]]` can be used just like the other rounds: `compare iso`, `overview iso.x1_370`, `overlay exp2.x1 iso.x1_370`,
`image iso.x1_370 -1` (without a name, name = sample_T_iso). Even with a single measurement n, k are obtained and only Ea is missing.

Measurement = heating → isothermal → cooling. It outputs only the full results, without warnings or judgments.

- Isothermal segment = the part of the temperature log where PV is at least the isothermal temperature − 3 °C (automatic). Cut before cooling starts
- X(t): the same way as compare (area of R, divided by the amorphous profile, in the FWHM window). X = 0 is the first frame (early heating), X = 1 is the end of the isothermal hold
- **t0** (t = 0): given `t0` > isothermal start (set temperature reached) > (if there is no temperature log) JMAK fit.
  In JMAK X = 1 − exp(−(k(t − t0 − τ))ⁿ), τ = incubation time after reaching the isothermal temperature
- **Avrami plot** ln[−ln(1−X)] vs ln(t − t0) is fitted with a line over `fit_range` (0.15–0.85) → n, k
- T_iso = mean PV over the isothermal segment
- Two or more temperatures for the same sample → ln k vs 1/T → Ea (`Ea_eV` from the Avrami k, `Ea_jmak_eV` from the JMAK k)
- For (400) the integration window is cut off at the detector edge, so the (222) value takes priority
- Result (`--save`): `out/avrami/<config name>_<iso|exp>_avrami.png / .csv / _arrhenius.csv / _frames.csv`
- Check on synthetic data (true n 2.7 · Ea 2.40 eV / incubation 1.5 min · n 2.0 · Ea 1.80 eV): n 2.60–2.73, Ea 2.35 eV / n 1.98–2.02, Ea 1.78–1.81 eV
- Existing heating data (`-g exp`, 10 °C/min): temperature at the center of the change x1 399 · x2 396 · x4 376 · x8 319 °C (agrees with compare T50),
  apparent n 6–10, only 7–10 frames in the X 15–85 % range → why isothermal measurements are needed

### Temperature

```powershell
uv run main.py merge-temp                        # raw/*.csv → merged file (rerun when a new log is added)
uv run main.py temp 16:55:00                     # temperature at a given time (KST)
uv run main.py temp Z:\...\test2x1_00086.h5      # temperature of a given frame
uv run main.py temp Z:\...\test2x1               # temperature of every frame in a folder (--csv to specify the log, --max-gap)
```

- Where the log is interrupted (more than 5 s from the nearest record), the temperature is NaN
- A heatmap with a temperature panel · the `at` time marker is made from Python with `heatmap.run(heatmap.Config(temp_log=..., at=...))`, not from the CLI (below)

Write the image folder/file as the **actual path** (it is not searched for automatically). overview also accepts experiment names from `experiments.toml` (`x1`, `exp2.x8`).
Options with several values are repeated (`--peaks 18.4 --peaks 21.4`).
If the path is omitted, `default_folder` / `default_file` in `[paths]` of `experiments.toml` are used.
`uv run insitu-xrd <command>` or `python -m insitu_xrd <command>` are equivalent.

Directly from Python:

```python
from pathlib import Path
from insitu_xrd import heatmap, tth_profile, Geometry

geo = Geometry(alpha_deg=20.0)
tth_profile.run(Path(r"Z:\...\testx1_00089.h5"), geo, show=False)
from insitu_xrd.temperature import DEFAULT_LOG, TempLog

res = heatmap.run(
    heatmap.Config(
        folder=Path(r"Z:\...\test2x1"), geometry=geo, temp_log=DEFAULT_LOG, at=(10, "16:55:00")
    )
)
res.frames  # per-frame file / time_kst / time_rel / temp_pv / temp_sv
TempLog().at([1791100537.0])  # Unix time → temperature
```

`at`: a number = relative time (min) from the first frame, `"HH:MM:SS"` = KST time. The corresponding frame is marked with a horizontal line and its profile is overlaid in the right panel.

## Structure

```
main.py                  entry point (compare if no arguments)
src/insitu_xrd/
  cli.py                 Typer subcommand definitions (insitu-xrd script · python -m insitu_xrd)
  config.py              code constants such as HDF5 keys + reads experiments.toml [geometry]·[paths]
  style.py               common figure style (Origin rc, segment · peak colors)
  geometry.py            Geometry (calibration values, 2θ/q/d conversion uses qorageom)
  io.py                  reads images / EPICS timestamps
  integrate.py           RadialIntegrator (image → 2θ profile)
  peaks.py               peak detection · Gaussian fitting
  timing.py              [timing]  file creation time vs EPICS acquisition time
  center_profile.py      [center]  pixel profile of the central strip
  tth_profile.py         [profile] 2θ profile of a single image
  heatmap.py             folder integration · cache · time heatmap (+ temperature), shared by overview · compare
  live.py                [heatmap] heatmap without temperature + peak area · FWHM (updated during a measurement with --watch)
  image.py               [image]   one h5 frame raw | image unrolled to 2θ–χ
  temperature.py         [merge-temp, temp] merges temperature logs / looks up temperature by time
  segments.py            temperature profile → automatic split into heating/hold/cooling segments
  tracks.py              peak tracking · decision of a transition (baseline crossing)
  overview.py            [overview] temperature + heatmap + per-segment profiles on one figure
  interactive.py         interactive HTML of overview (Plotly)
  compare.py             [compare] per-experiment X vs temperature · T50
  overlay.py             [overlay] overlay transitions of several scans
  profiles.py            [profiles] overlay 2θ profiles of several scans
  avrami.py              [avrami]  isothermal X(t) → Avrami n · k · Ea
data/temperature/        (excluded from git, local only)
  raw/                   raw logs of the temperature controller
  manual_data_20261004_x1.csv   merged file (default for temperature lookup)
out/<sample>/            all outputs (excluded from git)
out/cache/               integration result cache (recomputed if deleted)
tests/                   pytest (synthetic images · curves only, no real data needed)
```
