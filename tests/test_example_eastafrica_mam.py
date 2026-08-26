"""Smoke tests for the §8 end-to-end reference example (issue #24).

The dry-run test is fast (no compute). The tiny-pipeline test runs the real
AfricaS2S pipeline (seasonal_mme -> flex_forecast) on small synthetic data with
no network, so it's marked `integration` to stay out of the fast unit gate.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "examples" / "seasonal_forecast_eastafrica_mam.py"


def _run(args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=str(REPO),
    )


def test_dry_run_prints_plan_and_completes():
    r = _run(["--phase", "0", "1", "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "East Africa" in r.stdout and "MAM" in r.stdout
    assert "DRY RUN" in r.stdout
    assert "Pipeline complete" in r.stdout
    # SPEAR/CanSIPS exclusion is surfaced to the user.
    assert "SPEAR" in r.stdout


@pytest.mark.integration
def test_tiny_pipeline_runs_end_to_end(tmp_path):
    r = _run(["--tiny", "--phase", "0", "1", "2", "5", "6",
              "--output-dir", str(tmp_path)])
    assert r.returncode == 0, r.stderr
    assert "Pipeline complete" in r.stdout
    # Phase 2 produced an MME; phase 6 produced a flex forecast.
    assert "MME complete" in r.stdout
    assert "flex forecast" in r.stdout
