import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from insitu_xrd import config
from insitu_xrd.avrami import load_iso
from insitu_xrd.cli import app
from insitu_xrd.compare import exp_names, find_exp, load_settings


def test_project_path_expands_roots_keeps_absolute_and_resolves_relative(tmp_path: Path):
    abs_root = tmp_path / "srv_images"
    roots = {"images": str(abs_root), "data": "local_data"}

    assert config.project_path("<images>/IGOx4", roots=roots) == abs_root / "IGOx4"
    assert config.project_path("<images>", roots=roots) == abs_root
    assert config.project_path("<data>/temperature/a.csv", roots=roots) == (
        config.PROJECT_ROOT / "local_data" / "temperature" / "a.csv"
    )

    abs_file = tmp_path / "direct" / "scan.h5"
    assert config.project_path(abs_file, roots=roots) == abs_file
    assert config.project_path("rel/folder", roots=roots) == config.PROJECT_ROOT / "rel/folder"


def test_project_path_missing_root_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "_roots", {})

    with pytest.raises(SystemExit) as exc:
        config.project_path("<images>/IGOx4", roots={})
    msg = str(exc.value)
    assert "images" in msg
    assert config.OVERRIDE_FILE.name in msg

    cfg = tmp_path / "experiments.toml"
    cfg.write_text(
        '[peaks]\n"(222)" = 18.9\n[[exp]]\nname = "x1"\nimages = ["<images>/x1"]\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match=r"images.*experiments\.local\.toml"):
        load_settings(cfg)
    with pytest.raises(SystemExit, match=r"images.*experiments\.local\.toml"):
        find_exp("x1", cfg)


def test_load_toml_deep_merges_tables_and_replaces_arrays(tmp_path: Path):
    base = tmp_path / "experiments.toml"
    base.write_text(
        """
[geometry]
sdd_mm = 861.14
alpha_deg = 18.9

[paths]
default_folder = "<images>/testx1"
default_file = "<images>/testx1/testx1_00089.h5"

[peaks]
"(222)" = 18.9
"(400)" = 21.8

[[exp]]
name = "x1"
images = ["<images>/IGOx1"]
recipe = ["<data>/recipe_x1.csv"]

[[iso]]
sample = "x1"
T_iso = 380
images = ["<images>/IGOx1_iso"]
""",
        encoding="utf-8",
    )
    img_root = tmp_path / "imgs"
    data_root = tmp_path / "dat"
    local = tmp_path / "experiments.local.toml"
    local.write_text(
        f"""
[roots]
images = '{img_root}'
data = '{data_root}'

[geometry]
alpha_deg = 20.0

[[iso]]
sample = "x2"
T_iso = 370
images = ["<images>/IGOx2_iso"]
""",
        encoding="utf-8",
    )

    merged = config.load_toml(base)
    assert merged["geometry"] == {"sdd_mm": 861.14, "alpha_deg": 20.0}
    assert merged["peaks"] == {"(222)": 18.9, "(400)": 21.8}
    assert [e["name"] for e in merged["exp"]] == ["x1"]
    assert [e["sample"] for e in merged["iso"]] == ["x2"]

    st = load_settings(base)
    assert [e.name for e in st.exps] == ["x1"]
    assert st.exps[0].images == [img_root / "IGOx1"]
    assert st.exps[0].recipe == [data_root / "recipe_x1.csv"]

    _, iso_runs, _ = load_iso(base)
    assert [r.exp.name for r in iso_runs] == ["x2_370"]
    assert iso_runs[0].exp.images == [img_root / "IGOx2_iso"]


def test_override_path_skips_local_and_reports_syntax_error(tmp_path: Path):
    local = tmp_path / "experiments.local.toml"
    assert config.override_path(local) is None

    base = tmp_path / "experiments.toml"
    base.write_text("[paths]\ndefault_folder = '<images>/testx1'\n", encoding="utf-8")
    local.write_text("[roots\nbroken", encoding="utf-8")

    with pytest.raises(SystemExit, match=r"experiments\.local\.toml"):
        config.load_toml(base)


def test_no_local_file_loads_non_data_commands_and_stops_on_data_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "_roots", {})
    cfg = tmp_path / "experiments.toml"
    cfg.write_text(config.SETTINGS_FILE.read_text(encoding="utf-8"), encoding="utf-8")

    raw = config.load_toml(cfg)
    assert "(222)" in raw["peaks"]
    assert not re.search(r"[A-Za-z]:[/\\]", cfg.read_text(encoding="utf-8"))
    assert "exp.x1" in exp_names(cfg)

    st = load_settings(cfg, need_exps=False)
    assert st.exps == []
    assert st.peaks == {"(222)": 18.9, "(400)": 21.8}

    runner = CliRunner()
    for args in (["-h"], ["heatmap", "-h"], ["compare", "-h"], ["cache"]):
        res = runner.invoke(app, args)
        assert res.exit_code == 0, res.output

    res = runner.invoke(app, ["heatmap"])
    assert res.exit_code != 0
    assert "images" in res.output
    assert config.OVERRIDE_FILE.name in res.output
