import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from insitu_xrd import config
from insitu_xrd.avrami import load_iso
from insitu_xrd.cli import app
from insitu_xrd.compare import load_settings


def test_load_toml_deep_merges_tables_and_replaces_arrays(tmp_path: Path):
    base = tmp_path / "experiments.toml"
    base.write_text(
        """
[geometry]
sdd_mm = 861.14
alpha_deg = 18.9

[paths]
default_folder = "<data>/images/sample1"
default_file = "<data>/images/sample1/sample1_00001.h5"

[peaks]
"(222)" = 18.9
"(400)" = 21.8

[[exp]]
name = "placeholder"
images = ["<data>/images/sample1"]
""",
        encoding="utf-8",
    )
    local = tmp_path / "experiments.local.toml"
    local.write_text(
        """
[geometry]
alpha_deg = 20.0

[paths]
default_folder = "local/images/run1"

[[exp]]
name = "x1"
images = ["local/images/x1"]
recipe = ["local/temp/x1.csv"]

[[iso]]
sample = "x1"
T_iso = 380
images = ["local/images/x1_iso"]
""",
        encoding="utf-8",
    )

    merged = config.load_toml(base)
    assert merged["geometry"] == {"sdd_mm": 861.14, "alpha_deg": 20.0}
    assert merged["paths"] == {
        "default_folder": "local/images/run1",
        "default_file": "<data>/images/sample1/sample1_00001.h5",
    }
    assert merged["peaks"] == {"(222)": 18.9, "(400)": 21.8}
    assert [e["name"] for e in merged["exp"]] == ["x1"]
    assert merged["exp"][0]["images"] == ["local/images/x1"]

    st = load_settings(base)
    assert [e.name for e in st.exps] == ["x1"]
    assert st.exps[0].images == [config.PROJECT_ROOT / "local/images/x1"]

    _, iso_runs, _ = load_iso(base)
    assert [r.exp.name for r in iso_runs] == ["x1_380"]


def test_override_path_skips_local_and_reports_syntax_error(tmp_path: Path):
    local = tmp_path / "experiments.local.toml"
    assert config.override_path(local) is None

    base = tmp_path / "experiments.toml"
    base.write_text("[paths]\ndefault_folder = '<data>/images/sample1'\n", encoding="utf-8")
    local.write_text("[paths\nbroken", encoding="utf-8")

    with pytest.raises(SystemExit, match=r"experiments\.local\.toml"):
        config.load_toml(base)


def test_placeholders_load_without_local_file(tmp_path: Path):
    cfg = tmp_path / "experiments.toml"
    cfg.write_text(config.SETTINGS_FILE.read_text(encoding="utf-8"), encoding="utf-8")

    raw = config.load_toml(cfg)
    assert "(222)" in raw["peaks"]
    assert not re.search(r"[A-Za-z]:[/\\]", cfg.read_text(encoding="utf-8"))
    assert config.DEFAULT_FOLDER == config.PROJECT_ROOT / "<data>/images/sample1"
    assert config.DEFAULT_FILE == config.PROJECT_ROOT / "<data>/images/sample1/sample1_00001.h5"

    st = load_settings(cfg, need_exps=False)
    assert st.exps == []
    assert st.peaks == raw["peaks"]

    runner = CliRunner()
    for args in (["-h"], ["heatmap", "-h"], ["compare", "-h"], ["cache"]):
        res = runner.invoke(app, args)
        assert res.exit_code == 0, res.output
