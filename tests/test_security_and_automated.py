#!/usr/bin/env python3
"""
Developer-04 — Task 2: Automated tests + Security review.

Tests:
  1. test_no_hardcoded_secrets   — scan source for leaked API keys / tokens
  2. test_env_example_exists     — .env.example present and covers required vars
  3. test_model_output_deterministic — same input → same output (no random side-effects)
  4. test_intrinsic_reward_bounded   — reward signal stays in expected range
  5. test_category_probs_sum_to_one  — classifier output is a valid probability dist

Run:
    pytest tests/test_security_and_automated.py -v
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# ── 1. Security: no hardcoded secrets ───────────────────────────────────────

# Patterns that indicate a real secret (not a placeholder like sk-... or YOUR_KEY)
_SECRET_PATTERNS = [
    re.compile(r'sk-[A-Za-z0-9]{20,}'),          # OpenAI key
    re.compile(r'ghp_[A-Za-z0-9]{36,}'),          # GitHub PAT
    re.compile(r'AKIA[A-Z0-9]{16}'),               # AWS access key
    re.compile(r'AIza[0-9A-Za-z\-_]{35}'),         # Google API key
    re.compile(r'(?i)wandb.*=\s*[a-f0-9]{40}'),   # W&B token (40-char hex)
]

# Directories / files to skip
_SKIP = {'.git', '__pycache__', 'venv', '.venv', 'node_modules', '.env.example'}
_EXTENSIONS = {'.py', '.yaml', '.yml', '.json', '.toml', '.txt', '.env', '.sh'}


def _scan_secrets(root: Path):
    """Return list of (file, line_no, match) for any detected secrets."""
    hits = []
    for path in root.rglob('*'):
        if any(part in _SKIP for part in path.parts):
            continue
        if path.suffix not in _EXTENSIONS:
            continue
        try:
            text = path.read_text(errors='ignore')
        except (OSError, PermissionError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for pattern in _SECRET_PATTERNS:
                if pattern.search(line):
                    hits.append((path.relative_to(root), i, line.strip()[:80]))
    return hits


class TestSecurity:
    def test_no_hardcoded_secrets(self):
        """No real API keys or tokens should be committed to the repository."""
        hits = _scan_secrets(_root)
        if hits:
            report = '\n'.join(f"  {f}:{n}  {snippet}" for f, n, snippet in hits)
            pytest.fail(f"Hardcoded secrets found:\n{report}")

    def test_env_example_exists(self):
        """.env.example must exist and document the required environment variables."""
        env_example = _root / '.env.example'
        assert env_example.exists(), (
            ".env.example not found — add it so contributors know which env vars are needed"
        )
        content = env_example.read_text()
        required_vars = ['OPENAI_API_KEY', 'WANDB_API_KEY']
        for var in required_vars:
            assert var in content, (
                f"{var} is not documented in .env.example"
            )

    def test_dotenv_not_committed(self):
        """.env file (with real secrets) must not be tracked by git."""
        dot_env = _root / '.env'
        gitignore = _root / '.gitignore'
        if dot_env.exists():
            # If .env exists it must be in .gitignore
            assert gitignore.exists(), ".gitignore missing"
            content = gitignore.read_text()
            assert '.env' in content, ".env exists but is not in .gitignore — risk of secret leak"


# ── 2. Model output is deterministic ────────────────────────────────────────

class TestDeterminism:
    def test_vision_encoder_deterministic(self):
        """VisionEncoder must return the same output for the same input (eval mode)."""
        from src.encoders.vision_encoder import VisionEncoder
        model = VisionEncoder(embed_dim=64)
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2), "VisionEncoder output is not deterministic"

    def test_causal_layer_deterministic(self):
        """CausalLayer must return consistent outputs in eval mode."""
        from src.reasoning.causal_layer import CausalLayer
        model = CausalLayer(state_dim=32, num_variables=4)
        model.eval()
        state = torch.randn(3, 32)
        with torch.no_grad():
            out1 = model(state)
            out2 = model(state)
        assert torch.allclose(out1, out2), "CausalLayer output is not deterministic"


# ── 3. Intrinsic reward stays bounded ───────────────────────────────────────

class TestRewardBounds:
    def test_intrinsic_reward_non_negative(self):
        """IntrinsicRewardModule must never return negative rewards."""
        from src.motivation.intrinsic_reward import IntrinsicRewardModule
        module = IntrinsicRewardModule(state_dim=64)
        for _ in range(10):
            s  = torch.randn(8, 64)
            ns = torch.randn(8, 64)
            reward = module(s, ns)
            assert (reward >= 0).all(), f"Negative reward detected: {reward.min().item()}"

    def test_drive_system_bounded(self):
        """DriveSystem reward must stay in [0, 1] — learnability is a probability."""
        from src.motivation.drive_system import DriveSystem
        drive = DriveSystem(state_dim=64)
        for _ in range(10):
            s  = torch.randn(4, 64)
            ns = torch.randn(4, 64)
            r  = drive(s, ns)
            assert (r >= 0).all() and (r <= 1).all(), (
                f"Drive reward out of [0,1]: min={r.min():.3f} max={r.max():.3f}"
            )


# ── 4. Classifier produces valid probability distribution ────────────────────

class TestClassifierValidity:
    def test_category_probs_sum_to_one(self):
        """CategoryClassifier output must be a valid probability distribution."""
        from src.semantics.categories import CategoryClassifier
        clf = CategoryClassifier(feature_dim=64, num_categories=10)
        features = torch.randn(4, 64)
        probs = clf(features)
        assert probs.shape == (4, 10), f"Unexpected shape: {probs.shape}"
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5), \
            "Category probabilities do not sum to 1"
        assert (probs >= 0).all(), "Negative probabilities detected"

    def test_affordance_extractor_bounded(self):
        """AffordanceExtractor must return values in [0, 1] (sigmoid output)."""
        from src.semantics.affordances import AffordanceExtractor
        model = AffordanceExtractor(feature_dim=64, num_affordances=8)
        features = torch.randn(4, 64)
        out = model(features)
        assert out.shape == (4, 8)
        assert (out >= 0).all() and (out <= 1).all(), \
            "Affordance values must be in [0, 1]"
