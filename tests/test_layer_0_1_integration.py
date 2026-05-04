#!/usr/bin/env python3
"""
Integration tests for Layer 0 (World Model) and Layer 1 (Semantic Properties).

Verifies that:
- Multi-modal encoders produce embeddings of the expected shape
- Slot attention discovers at least 2 object properties
- World model dynamics prediction runs without error
- Layer 0 and Layer 1 can be composed end-to-end

Run with:
    pytest tests/test_layer_0_1_integration.py -v
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
# Layer 0 — World Model
# ---------------------------------------------------------------------------

class TestWorldModelIntegration:
    """End-to-end tests for Layer 0 multi-modal world model."""

    def test_vision_encoder_output_shape(self):
        """Vision encoder produces a fixed-size embedding."""
        from src.encoders.vision_encoder import VisionEncoder
        enc = VisionEncoder(embed_dim=128)
        x = torch.randn(2, 3, 64, 64)
        z = enc(x)
        assert z.shape == (2, 128), f"Expected (2, 128), got {z.shape}"

    def test_audio_encoder_output_shape(self):
        """Audio encoder produces a fixed-size embedding."""
        from src.encoders.audio_encoder import AudioEncoder
        enc = AudioEncoder(embed_dim=64)
        x = torch.randn(2, 1, 64, 64)
        z = enc(x)
        assert z.ndim == 2 and z.shape[1] == 64, f"Unexpected shape: {z.shape}"

    def test_proprio_encoder_output_shape(self):
        """Proprioception encoder produces a fixed-size embedding."""
        from src.encoders.proprio_encoder import ProprioEncoder
        enc = ProprioEncoder(input_dim=16, embed_dim=32)
        x = torch.randn(2, 16)
        z = enc(x)
        assert z.shape == (2, 32), f"Expected (2, 32), got {z.shape}"

    def test_dynamics_prediction_runs(self):
        """World model dynamics predictor runs a forward pass without error."""
        from src.world_model.dynamics import DynamicsPredictor
        model = DynamicsPredictor(state_dim=64, action_dim=8, hidden_dim=128)
        state = torch.randn(2, 64)
        action = torch.randn(2, 8)
        next_state = model(state, action)
        assert next_state.shape == (2, 64), f"Unexpected shape: {next_state.shape}"

    def test_temporal_world_model_sequence(self):
        """Temporal world model processes a sequence of observations."""
        from src.world_model.temporal_world_model import TemporalWorldModel
        model = TemporalWorldModel(obs_dim=64, hidden_dim=128)
        obs_seq = torch.randn(4, 2, 64)  # (T, B, obs_dim)
        outputs = model(obs_seq)
        assert outputs is not None


# ---------------------------------------------------------------------------
# Layer 1 — Semantic Properties
# ---------------------------------------------------------------------------

class TestSemanticLayerIntegration:
    """Integration tests for Layer 1 semantic property extraction."""

    def test_property_layer_discovers_properties(self):
        """PropertyLayer extracts at least 2 object properties from a visual input."""
        from src.semantics.property_layer import PropertyLayer, PropertyConfig
        cfg = PropertyConfig(num_slots=4, slot_dim=32, input_dim=64)
        layer = PropertyLayer(cfg)
        features = torch.randn(2, 64)
        props = layer(features)
        # Must return a dict or tensor with multiple property dimensions
        if isinstance(props, dict):
            assert len(props) >= 2, "Expected at least 2 property types"
        else:
            assert props.shape[-1] >= 2, "Expected at least 2 property dimensions"

    def test_affordance_extraction(self):
        """Affordance module runs without error."""
        from src.semantics.affordances import AffordanceExtractor
        model = AffordanceExtractor(feature_dim=64, num_affordances=8)
        features = torch.randn(2, 64)
        affordances = model(features)
        assert affordances.shape[-1] == 8

    def test_object_categories(self):
        """Category classifier produces valid probability distribution."""
        from src.semantics.categories import CategoryClassifier
        clf = CategoryClassifier(feature_dim=64, num_categories=10)
        features = torch.randn(2, 64)
        probs = clf(features)
        assert probs.shape == (2, 10)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5), \
            "Category probabilities must sum to 1"


# ---------------------------------------------------------------------------
# End-to-end: Layer 0 → Layer 1 composition
# ---------------------------------------------------------------------------

class TestLayer0to1Composition:
    """Verify Layer 0 embeddings feed cleanly into Layer 1."""

    def test_encoder_to_property_pipeline(self):
        """Vision encoder output can be passed directly into PropertyLayer."""
        from src.encoders.vision_encoder import VisionEncoder
        from src.semantics.property_layer import PropertyLayer, PropertyConfig

        enc = VisionEncoder(embed_dim=64)
        cfg = PropertyConfig(num_slots=4, slot_dim=32, input_dim=64)
        layer = PropertyLayer(cfg)

        img = torch.randn(2, 3, 64, 64)
        z = enc(img)
        props = layer(z)
        assert props is not None, "End-to-end Layer 0 → Layer 1 pipeline failed"
