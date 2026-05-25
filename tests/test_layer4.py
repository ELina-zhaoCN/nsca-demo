#!/usr/bin/env python3
"""
Tests for Issue #5 — Layer 4: Learned Language Grounding.

Covers:
  1. test_concept_registry          — ≥5 distinct concepts are registered
  2. test_language_to_percept       — L→P direction returns correct-shape tensor
  3. test_percept_to_language       — P→L direction returns top-k ranked concepts
  4. test_bidirectional_roundtrip   — L→P→L round-trip recovers source concept
  5. test_contrastive_loss_finite   — training loss is a finite scalar
  6. test_balloon_diagnostic        — balloon percept matches ≥1 expected concept
  7. test_llm_graceful_degradation  — LLM path works without any API key
  8. test_layer4_unified_forward    — Layer4LanguageGrounding.forward() output shape
  9. test_five_concepts_grounded    — at least 5 concepts ground distinct percepts

Run:
    pytest tests/test_layer4.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.language.grounding import (
    LearnedGrounding,
    Layer4LanguageGrounding,
    BalloonDiagnostic,
    LLMReasoner,
    _CONCEPT_SEEDS,
    _PROP_DIM,
)


# ---------------------------------------------------------------------------
# 1. Concept registry
# ---------------------------------------------------------------------------

class TestConceptRegistry:
    def test_concept_registry_at_least_five(self):
        """At least 5 language concepts must be registered (spec requirement)."""
        grounder = LearnedGrounding()
        assert len(grounder.CONCEPTS) >= 5, (
            f"Only {len(grounder.CONCEPTS)} concepts registered; need ≥5"
        )

    def test_required_concepts_present(self):
        """All seed concepts listed in the spec must be available."""
        required = {"heavy", "fast", "round", "above", "push",
                    "soft", "light", "rigid", "animate", "smooth"}
        grounder = LearnedGrounding()
        present = set(grounder.CONCEPTS)
        missing = required - present
        assert not missing, f"Missing required concepts: {missing}"


# ---------------------------------------------------------------------------
# 2. Language → Perception direction
# ---------------------------------------------------------------------------

class TestLanguageToPercept:
    def test_output_shape(self):
        """language_to_percept returns a 1-D tensor of length prop_dim."""
        grounder = LearnedGrounding()
        for concept in grounder.CONCEPTS:
            vec = grounder.language_to_percept(concept)
            assert vec.shape == (grounder.prop_dim,), (
                f"concept '{concept}': expected shape ({grounder.prop_dim},), got {vec.shape}"
            )

    def test_output_is_finite(self):
        """Predicted property vectors must not contain NaN or Inf."""
        grounder = LearnedGrounding()
        for concept in grounder.CONCEPTS:
            vec = grounder.language_to_percept(concept)
            assert torch.isfinite(vec).all(), (
                f"concept '{concept}' produced non-finite property prediction"
            )

    def test_unknown_concept_raises(self):
        """KeyError raised for concepts not in the registry."""
        grounder = LearnedGrounding()
        with pytest.raises(KeyError):
            grounder.language_to_percept("nonsense_concept_xyz")


# ---------------------------------------------------------------------------
# 3. Perception → Language direction
# ---------------------------------------------------------------------------

class TestPerceptToLanguage:
    def test_returns_top_k_results(self):
        """percept_to_language returns exactly top_k (concept, score) pairs."""
        grounder = LearnedGrounding()
        props = torch.randn(_PROP_DIM)
        results = grounder.percept_to_language(props, top_k=3)
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    def test_results_are_sorted_descending(self):
        """Concepts are returned in descending similarity order."""
        grounder = LearnedGrounding()
        props = torch.randn(_PROP_DIM)
        results = grounder.percept_to_language(props, top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True), \
            "percept_to_language results not sorted by descending score"

    def test_scores_are_cosine_similarities(self):
        """Similarity scores must be in [-1, 1] (cosine range)."""
        grounder = LearnedGrounding()
        props = torch.randn(_PROP_DIM)
        results = grounder.percept_to_language(props, top_k=len(grounder.CONCEPTS))
        for name, score in results:
            assert -1.01 <= score <= 1.01, (
                f"Score for '{name}' out of cosine range: {score:.4f}"
            )

    def test_all_results_are_known_concepts(self):
        """Every returned concept name must be in the registry."""
        grounder = LearnedGrounding()
        props = torch.randn(_PROP_DIM)
        results = grounder.percept_to_language(props, top_k=5)
        for name, _ in results:
            assert name in grounder.CONCEPTS, f"Unknown concept returned: '{name}'"


# ---------------------------------------------------------------------------
# 4. Bidirectional round-trip
# ---------------------------------------------------------------------------

class TestBidirectionalRoundtrip:
    @staticmethod
    def _quick_train(grounder: LearnedGrounding, steps: int = 80) -> None:
        """Brief contrastive training to align percept-encoder ↔ decoder spaces."""
        optimizer = torch.optim.Adam(grounder.parameters(), lr=3e-3)
        n = len(grounder.CONCEPTS)
        for _ in range(steps):
            optimizer.zero_grad()
            # One sample per concept: property vector = decoder output (noisy)
            with torch.no_grad():
                all_ids = torch.arange(n)
                props = grounder.concept_decoder(
                    torch.nn.functional.normalize(
                        grounder.concept_embeddings(all_ids), dim=-1
                    )
                ) + 0.05 * torch.randn(n, grounder.prop_dim)
            loss = grounder.contrastive_loss(props, all_ids)
            loss.backward()
            optimizer.step()

    def test_roundtrip_recovers_source_concept(self):
        """
        L→P→L round-trip: concept → predicted percept → query back.
        After brief alignment training, the source concept should appear in top-5.
        """
        grounder = LearnedGrounding()
        self._quick_train(grounder, steps=80)

        # Use semantically distinct concepts
        pivot_concepts = ["heavy", "soft", "animate", "smooth", "rigid"]
        for concept in pivot_concepts:
            percept_vec = grounder.language_to_percept(concept)
            top5 = grounder.percept_to_language(percept_vec, top_k=5)
            top5_names = [n for n, _ in top5]
            assert concept in top5_names, (
                f"Round-trip failed for '{concept}': top-5 = {top5_names}"
            )


# ---------------------------------------------------------------------------
# 5. Contrastive training loss
# ---------------------------------------------------------------------------

class TestContrastiveLoss:
    def test_loss_is_finite_scalar(self):
        """contrastive_loss returns a finite scalar tensor."""
        grounder = LearnedGrounding()
        B = 4
        props = torch.randn(B, _PROP_DIM)
        concept_ids = torch.randint(0, len(grounder.CONCEPTS), (B,))
        loss = grounder.contrastive_loss(props, concept_ids)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_loss_is_non_negative(self):
        """Cross-entropy loss is always non-negative."""
        grounder = LearnedGrounding()
        B = 6
        props = torch.randn(B, _PROP_DIM)
        concept_ids = torch.randint(0, len(grounder.CONCEPTS), (B,))
        loss = grounder.contrastive_loss(props, concept_ids)
        assert loss.item() >= 0, f"Negative loss: {loss.item()}"

    def test_loss_decreases_with_training(self):
        """Loss should decrease when training on perfectly matched pairs."""
        grounder = LearnedGrounding()
        optimizer = torch.optim.Adam(grounder.parameters(), lr=1e-3)

        # Perfectly matched (same concept repeated) — easy to overfit
        concept_idx = 0
        B = 8
        # Use language_to_percept to get the "expected" percept for this concept
        with torch.no_grad():
            target_percept = grounder.language_to_percept(
                grounder.CONCEPTS[concept_idx]
            ).unsqueeze(0).expand(B, -1).clone() + 0.05 * torch.randn(B, _PROP_DIM)
        concept_ids = torch.full((B,), concept_idx, dtype=torch.long)

        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            loss = grounder.contrastive_loss(target_percept, concept_ids)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease during training: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# 6. Balloon diagnostic
# ---------------------------------------------------------------------------

class TestBalloonDiagnostic:
    def test_balloon_diagnostic_runs(self):
        """BalloonDiagnostic.run() completes without error."""
        grounder = LearnedGrounding()
        result = BalloonDiagnostic.run(grounder, top_k=5)
        assert "passed" in result
        assert "predictions" in result
        assert "matched" in result

    def test_balloon_diagnostic_structure(self):
        """Diagnostic result contains required keys with correct types."""
        grounder = LearnedGrounding()
        result = BalloonDiagnostic.run(grounder, top_k=5)
        assert isinstance(result["passed"], bool)
        assert isinstance(result["predictions"], list)
        assert len(result["predictions"]) == 5

    def test_balloon_diagnostic_passes_after_fine_tuning(self):
        """
        After a few gradient steps on balloon-like percepts, at least one
        of {light, soft, smooth} should appear in the top-5.
        We accept the un-trained result too — seed prototypes may already work.
        """
        grounder = LearnedGrounding()
        # Quick fine-tune: push balloon percept toward "light"
        optimizer = torch.optim.Adam(grounder.parameters(), lr=5e-3)
        balloon = BalloonDiagnostic.BALLOON_PROPS.unsqueeze(0)  # [1, 9]
        light_id = torch.tensor([grounder.concept_id("light")])
        for _ in range(50):
            optimizer.zero_grad()
            loss = grounder.contrastive_loss(balloon, light_id)
            loss.backward()
            optimizer.step()
        result = BalloonDiagnostic.run(grounder, top_k=5)
        assert result["passed"], (
            f"Balloon diagnostic failed after fine-tuning.\n"
            f"Expected ≥1 of {result['expected']}, got {result['predictions']}"
        )


# ---------------------------------------------------------------------------
# 7. LLM graceful degradation
# ---------------------------------------------------------------------------

class TestLLMGracefulDegradation:
    def test_llm_disabled_without_key(self):
        """LLMReasoner must be disabled when no API key is provided."""
        llm = LLMReasoner(api_key="")
        assert not llm.enabled, "LLMReasoner should be disabled with empty key"

    def test_llm_fallback_returns_string(self):
        """reason() must return a non-empty string even when LLM is disabled."""
        llm = LLMReasoner(api_key="")
        result = llm.reason(["light", "smooth"], "What is this object?")
        assert isinstance(result, str) and len(result) > 0

    def test_llm_placeholder_key_disabled(self):
        """Placeholder key 'sk-...' must not enable LLM."""
        llm = LLMReasoner(api_key="sk-...")
        assert not llm.enabled

    def test_layer4_works_without_llm(self):
        """Layer4LanguageGrounding works with use_llm=False (default)."""
        layer4 = Layer4LanguageGrounding(use_llm=False)
        props = torch.randn(_PROP_DIM)
        result = layer4(props)
        assert "concepts" in result
        assert "answer" in result
        assert isinstance(result["answer"], str)


# ---------------------------------------------------------------------------
# 8. Unified Layer4 forward
# ---------------------------------------------------------------------------

class TestLayer4UnifiedForward:
    def test_single_input(self):
        """Single property vector returns a single result dict."""
        layer4 = Layer4LanguageGrounding(top_k=3)
        props = torch.randn(_PROP_DIM)
        result = layer4(props)
        assert isinstance(result, dict), "Expected dict for single input"
        assert "concepts" in result
        assert len(result["concepts"]) == 3

    def test_batch_input(self):
        """Batch of property vectors returns a list of result dicts."""
        layer4 = Layer4LanguageGrounding(top_k=3)
        props = torch.randn(4, _PROP_DIM)
        results = layer4(props)
        assert isinstance(results, list) and len(results) == 4
        for r in results:
            assert "concepts" in r and len(r["concepts"]) == 3

    def test_grounded_props_included(self):
        """grounded_props key must map each concept to a property tensor."""
        layer4 = Layer4LanguageGrounding(top_k=2)
        props = torch.randn(_PROP_DIM)
        result = layer4(props)
        assert "grounded_props" in result
        for concept, vec in result["grounded_props"].items():
            assert isinstance(concept, str)
            assert vec.shape == (layer4.grounder.prop_dim,)


# ---------------------------------------------------------------------------
# 9. Five concepts ground distinct percepts
# ---------------------------------------------------------------------------

class TestFiveConceptsGrounded:
    def test_five_concepts_produce_distinct_percepts(self):
        """
        Language→Perception for ≥5 concepts must produce distinct vectors.
        (Identical vectors would mean the grounding collapsed.)
        """
        grounder = LearnedGrounding()
        # Pick 5 semantically distinct concepts
        pivot = ["heavy", "soft", "animate", "smooth", "rigid"]
        vecs = [grounder.language_to_percept(c) for c in pivot]

        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                cos = torch.nn.functional.cosine_similarity(
                    vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)
                ).item()
                # Should not be identical (collapsed) — allow < 0.99
                assert cos < 0.99, (
                    f"Concepts '{pivot[i]}' and '{pivot[j]}' produce nearly "
                    f"identical percept vectors (cos={cos:.4f}) — grounding collapsed"
                )
