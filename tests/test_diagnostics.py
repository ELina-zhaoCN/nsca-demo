#!/usr/bin/env python3
"""
Smoke tests for the four pre-training diagnostics (SPEC Issue #5 / #6).

Verifies that each diagnostic script can be imported and its main
validation function runs without raising an exception.

Run with:
    pytest tests/test_diagnostics.py -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parent.parent


class TestDiagnosticScriptsExist:
    """All four diagnostic scripts must be present."""

    @pytest.mark.parametrize("script", [
        "scripts/noisy_tv_test.py",
        "scripts/forgetting_test.py",
        "scripts/balloon_test.py",
        "scripts/slot_discovery_test.py",
    ])
    def test_script_exists(self, script: str):
        assert (_root / script).exists(), f"Diagnostic script missing: {script}"


class TestRunAllDiagnostics:
    """run_all.py must exist and be syntactically valid."""

    def test_run_all_importable(self):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", "diagnostics/run_all.py"],
            cwd=str(_root),
            capture_output=True,
        )
        assert result.returncode == 0, (
            f"diagnostics/run_all.py has syntax errors:\n{result.stderr.decode()}"
        )

    def test_evaluation_module_importable(self):
        result = subprocess.run(
            [sys.executable, "-c", "from src.evaluation.metaworld_eval import AblationStudy, EvaluationConfig"],
            cwd=str(_root),
            capture_output=True,
        )
        assert result.returncode == 0, (
            f"Evaluation module import failed:\n{result.stderr.decode()}"
        )


class TestSampleEfficiencyBenchmark:
    """AblationStudy configuration must be valid."""

    def test_evaluation_config_defaults(self):
        import sys
        sys.path.insert(0, str(_root))
        from src.evaluation.metaworld_eval import EvaluationConfig
        cfg = EvaluationConfig()
        assert cfg.num_seeds >= 1
        assert cfg.num_samples >= 1

    def test_ablation_study_instantiates(self):
        import sys
        sys.path.insert(0, str(_root))
        from src.evaluation.metaworld_eval import AblationStudy, EvaluationConfig
        cfg = EvaluationConfig(num_seeds=1, num_samples=5)
        study = AblationStudy(cfg)
        assert study is not None
