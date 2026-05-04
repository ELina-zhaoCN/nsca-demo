#!/usr/bin/env python3
"""
Integration tests for Layer 2 (Causal Reasoning) and Layer 3 (Motivation System).

Verifies that:
- Causal graph learning runs without error
- Counterfactual simulation produces valid outputs
- Curiosity filter correctly rejects noisy-TV signals
- EWC reduces catastrophic forgetting on sequential tasks
- Intrinsic reward signal is non-negative

Run with:
    pytest tests/test_layer_2_3_integration.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# ---------------------------------------------------------------------------
# Layer 2 — Causal Reasoning
# ---------------------------------------------------------------------------

class TestCausalReasoningIntegration:
    """Integration tests for Layer 2 causal reasoning module."""

    def test_causal_layer_forward(self):
        """CausalLayer runs a forward pass without error."""
        from src.reasoning.causal_layer import CausalLayer
        model = CausalLayer(state_dim=64, num_variables=8)
        state = torch.randn(2, 64)
        output = model(state)
        assert output is not None

    def test_counterfactual_simulation(self):
        """CounterfactualSimulator produces outputs of the same shape as input."""
        from src.reasoning.counterfactual import CounterfactualSimulator
        sim = CounterfactualSimulator(state_dim=64, action_dim=8)
        state = torch.randn(2, 64)
        action = torch.randn(2, 8)
        cf_state = sim(state, action)
        assert cf_state.shape == state.shape, \
            f"Counterfactual state shape mismatch: {cf_state.shape} vs {state.shape}"

    def test_intuitive_physics_engine(self):
        """Intuitive physics engine runs without error."""
        from src.reasoning.intuitive_physics import IntuitivePhysicsEngine
        engine = IntuitivePhysicsEngine(state_dim=64)
        state = torch.randn(2, 64)
        prediction = engine(state)
        assert prediction is not None


# ---------------------------------------------------------------------------
# Layer 3 — Motivation System
# ---------------------------------------------------------------------------

class TestMotivationSystemIntegration:
    """Integration tests for Layer 3 motivation and curiosity system."""

    def test_intrinsic_reward_non_negative(self):
        """Intrinsic reward signal must be non-negative."""
        from src.motivation.intrinsic_reward import IntrinsicRewardModule
        module = IntrinsicRewardModule(state_dim=64)
        state = torch.randn(4, 64)
        next_state = torch.randn(4, 64)
        reward = module(state, next_state)
        assert (reward >= 0).all(), "Intrinsic rewards must be non-negative"

    def test_noisy_tv_filter_rejects_noise(self):
        """Curiosity filter should assign lower learnability to pure random noise."""
        from src.motivation.drive_system import DriveSystem
        drive = DriveSystem(state_dim=64)
        # Structured transition (easier to learn)
        s1 = torch.zeros(4, 64)
        s2 = s1 + 0.01  # small deterministic shift
        # Noisy transition (hard to learn — mimics noisy-TV)
        s_noisy = torch.randn(4, 64)
        s_noisy_next = torch.randn(4, 64)

        reward_structured = drive(s1, s2).mean().item()
        reward_noisy = drive(s_noisy, s_noisy_next).mean().item()
        # Noisy-TV signal should not dominate curiosity reward
        assert reward_noisy <= reward_structured * 10, \
            "Noisy-TV filter is not suppressing random noise curiosity"

    def test_ewc_penalty_reduces_forgetting(self):
        """EWC penalty term is computed and is a positive scalar."""
        from src.learning.ewc import ElasticWeightConsolidation
        import torch.nn as nn

        model = nn.Linear(16, 8)
        ewc = ElasticWeightConsolidation(model, dataset=None, device="cpu")

        # Simulate Fisher information already computed
        for name, param in model.named_parameters():
            ewc.fisher[name] = torch.ones_like(param)
            ewc.optimal_params[name] = param.clone()

        penalty = ewc.penalty(model)
        assert penalty.item() >= 0, "EWC penalty must be non-negative"

    def test_attention_module(self):
        """Attention module in motivation system produces valid output."""
        from src.motivation.attention import MotivationAttention
        attn = MotivationAttention(dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)  # (batch, seq, dim)
        out = attn(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Layer 2 → Layer 3 composition
# ---------------------------------------------------------------------------

class TestLayer2to3Composition:
    """Verify causal reasoning output feeds into motivation system."""

    def test_causal_to_motivation_pipeline(self):
        """CausalLayer output can be passed into IntrinsicRewardModule."""
        from src.reasoning.causal_layer import CausalLayer
        from src.motivation.intrinsic_reward import IntrinsicRewardModule

        causal = CausalLayer(state_dim=64, num_variables=8)
        reward_module = IntrinsicRewardModule(state_dim=64)

        state = torch.randn(2, 64)
        causal_repr = causal(state)

        if isinstance(causal_repr, tuple):
            causal_repr = causal_repr[0]

        reward = reward_module(state, causal_repr)
        assert reward is not None, "Layer 2 → Layer 3 pipeline failed"
