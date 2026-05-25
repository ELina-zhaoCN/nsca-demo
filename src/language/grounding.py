"""
Layer 4: Learned Language Grounding (Issue #5)

Grounds language concepts in sensorimotor representations without manual
dictionaries. Uses co-occurrence learning to map word embeddings <-> perceptual
features bidirectionally.

Concepts grounded (≥5 required):
    "heavy", "fast", "round", "above", "push",
    "soft", "light", "rigid", "animate", "smooth"
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Concept registry — no manual dictionaries, learned from co-occurrence
# ---------------------------------------------------------------------------

# Seed prototypes: which *perceptual dimensions* each concept is associated with.
# Indices match PropertyLayer output: [hardness, weight, size, animacy,
# rigidity, transparency, roughness, temperature, containment]
# These are SOFT initializations — the network learns to refine them.
_CONCEPT_SEEDS: Dict[str, Dict[str, float]] = {
    "heavy":   {"weight": 1.0, "size": 0.5},
    "fast":    {"animacy": 0.8, "temperature": 0.3},
    "round":   {"roughness": -0.5, "size": 0.3},
    "above":   {"size": -0.2, "temperature": -0.1},
    "push":    {"hardness": 0.6, "rigidity": 0.8},
    "soft":    {"hardness": -1.0, "rigidity": -0.7},
    "light":   {"weight": -1.0, "size": -0.3},
    "rigid":   {"rigidity": 1.0, "hardness": 0.5},
    "animate": {"animacy": 1.0, "temperature": 0.4},
    "smooth":  {"roughness": -1.0},
}

_PROPERTY_NAMES = [
    "hardness", "weight", "size", "animacy",
    "rigidity", "transparency", "roughness", "temperature", "containment",
]
_PROP_INDEX = {name: i for i, name in enumerate(_PROPERTY_NAMES)}
_PROP_DIM = len(_PROPERTY_NAMES)  # 9


def _seed_vector(concept: str) -> torch.Tensor:
    """Build a soft prototype vector from the seed dictionary."""
    vec = torch.zeros(_PROP_DIM)
    for prop, val in _CONCEPT_SEEDS.get(concept, {}).items():
        if prop in _PROP_INDEX:
            vec[_PROP_INDEX[prop]] = val
    return vec


# ---------------------------------------------------------------------------
# 2. Bidirectional grounding network
# ---------------------------------------------------------------------------

class LearnedGrounding(nn.Module):
    """
    Bidirectional grounding: concept embeddings <-> perceptual property vectors.

    Perception → Language  : given property vector, retrieve closest concept
    Language   → Perception: given concept name, produce expected property vector

    Trained via contrastive co-occurrence: aligned (word, percept) pairs get
    pulled together; misaligned pairs are pushed apart.
    """

    CONCEPTS: List[str] = list(_CONCEPT_SEEDS.keys())

    def __init__(
        self,
        prop_dim: int = _PROP_DIM,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.prop_dim  = prop_dim
        self.embed_dim = embed_dim

        # Perception encoder: property vector → shared embedding space
        self.percept_encoder = nn.Sequential(
            nn.Linear(prop_dim,  128), nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

        # Concept embeddings (one per concept, learned)
        n = len(self.CONCEPTS)
        self.concept_embeddings = nn.Embedding(n, embed_dim)

        # Initialize concept embeddings from seed prototypes
        with torch.no_grad():
            for i, name in enumerate(self.CONCEPTS):
                seed = _seed_vector(name)           # [prop_dim]
                # Project seed through a linear init to embed_dim
                proj = nn.Linear(prop_dim, embed_dim, bias=False)
                nn.init.xavier_uniform_(proj.weight)
                self.concept_embeddings.weight[i] = proj(seed)

        # Language → Perception decoder
        self.concept_decoder = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Linear(128, prop_dim),
        )

        self._concept_index = {name: i for i, name in enumerate(self.CONCEPTS)}

    # ── forward helpers ────────────────────────────────────────────────────

    def encode_percept(self, props: torch.Tensor) -> torch.Tensor:
        """Property vector → normalized embedding.  [B, prop_dim] → [B, embed_dim]"""
        return F.normalize(self.percept_encoder(props), dim=-1)

    def concept_id(self, name: str) -> int:
        if name not in self._concept_index:
            raise KeyError(f"Unknown concept '{name}'. Known: {self.CONCEPTS}")
        return self._concept_index[name]

    def concept_vector(self, name: str) -> torch.Tensor:
        """Return the normalized embedding for a concept name.  → [embed_dim]"""
        idx = torch.tensor([self.concept_id(name)])
        return F.normalize(self.concept_embeddings(idx), dim=-1).squeeze(0)

    # ── Language → Perception ──────────────────────────────────────────────

    def language_to_percept(self, concept_name: str) -> torch.Tensor:
        """
        Given a concept name, predict the expected property vector.
        Language → Perception grounding.
        Returns: [prop_dim]
        """
        emb = self.concept_vector(concept_name).unsqueeze(0)   # [1, embed_dim]
        return self.concept_decoder(emb).squeeze(0)             # [prop_dim]

    # ── Perception → Language ──────────────────────────────────────────────

    def percept_to_language(
        self, props: torch.Tensor, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Given a property vector, return top-k grounded concepts with scores.
        Perception → Language grounding.

        Args:
            props: [prop_dim]  (single sample)
        Returns:
            List of (concept_name, cosine_similarity) sorted descending
        """
        percept_emb = self.encode_percept(props.unsqueeze(0))   # [1, embed_dim]
        all_ids     = torch.arange(len(self.CONCEPTS))
        all_embs    = F.normalize(
            self.concept_embeddings(all_ids), dim=-1
        )                                                        # [N, embed_dim]
        sims = (percept_emb @ all_embs.T).squeeze(0)            # [N]
        top  = sims.topk(min(top_k, len(self.CONCEPTS)))
        return [
            (self.CONCEPTS[i.item()], sims[i].item())
            for i in top.indices
        ]

    # ── Contrastive training loss ──────────────────────────────────────────

    def contrastive_loss(
        self,
        props: torch.Tensor,
        concept_ids: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss for co-occurrence grounding.

        Args:
            props:       [B, prop_dim]  perceptual observations
            concept_ids: [B]            matching concept indices
        """
        percept_embs = self.encode_percept(props)                # [B, embed_dim]
        concept_embs = F.normalize(
            self.concept_embeddings(concept_ids), dim=-1
        )                                                         # [B, embed_dim]
        logits  = (percept_embs @ concept_embs.T) / temperature  # [B, B]
        labels  = torch.arange(len(props), device=props.device)
        loss_p2c = F.cross_entropy(logits,   labels)
        loss_c2p = F.cross_entropy(logits.T, labels)
        return (loss_p2c + loss_c2p) / 2


# ---------------------------------------------------------------------------
# 3. Balloon diagnostic
# ---------------------------------------------------------------------------

class BalloonDiagnostic:
    """
    Diagnostic: can the grounding system correctly predict balloon behavior?

    A balloon has properties: light (low weight), round (low roughness),
    and floats above (spatial relation). The grounding should associate
    these perceptual features with correct language concepts.
    """

    # Simulated perceptual signature of a balloon
    # [hardness, weight, size, animacy, rigidity, transparency, roughness, temperature, containment]
    BALLOON_PROPS = torch.tensor(
        [0.1, 0.05, 0.4, 0.0, 0.05, 0.8, 0.05, 0.2, 0.7],
        dtype=torch.float32,
    )

    # Concepts we expect the system to ground for a balloon
    EXPECTED_CONCEPTS = {"light", "soft", "smooth"}

    @staticmethod
    def run(grounder: LearnedGrounding, top_k: int = 5) -> Dict:
        """
        Run the balloon diagnostic.

        Returns:
            dict with keys: passed (bool), matched (set), predictions (list)
        """
        preds = grounder.percept_to_language(
            BalloonDiagnostic.BALLOON_PROPS, top_k=top_k
        )
        pred_names = {name for name, _ in preds}
        matched    = BalloonDiagnostic.EXPECTED_CONCEPTS & pred_names
        passed     = len(matched) >= 1   # at least 1 expected concept in top-k

        return {
            "passed":      passed,
            "matched":     matched,
            "predictions": preds,
            "expected":    BalloonDiagnostic.EXPECTED_CONCEPTS,
        }


# ---------------------------------------------------------------------------
# 4. Optional LLM integration (gracefully disabled when key is absent)
# ---------------------------------------------------------------------------

class LLMReasoner:
    """
    Optional LLM integration for high-level language reasoning.
    Gracefully disabled when OPENAI_API_KEY is not set.
    """

    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.model   = model
        self.enabled = False
        self._client = None

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if key and not key.startswith("sk-..."):
            try:
                import openai
                self._client = openai.OpenAI(api_key=key)
                self.enabled = True
            except ImportError:
                pass   # openai package not installed → disabled

    def reason(self, concepts: List[str], question: str) -> str:
        """
        Ask the LLM a question about grounded concepts.
        Falls back to rule-based answer when LLM is disabled.
        """
        if not self.enabled or self._client is None:
            return self._fallback(concepts, question)

        prompt = (
            f"The object has these grounded properties: {', '.join(concepts)}.\n"
            f"Question: {question}\nAnswer briefly:"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return self._fallback(concepts, question)

    @staticmethod
    def _fallback(concepts: List[str], question: str) -> str:
        return f"[LLM disabled] Grounded concepts: {', '.join(concepts)}."


# ---------------------------------------------------------------------------
# 5. Unified Layer 4 interface
# ---------------------------------------------------------------------------

class Layer4LanguageGrounding(nn.Module):
    """
    Unified Layer 4: Language Grounding + optional LLM reasoning.

    Usage:
        layer4 = Layer4LanguageGrounding(use_llm=False)
        result = layer4(property_vector)
        # result['concepts']  → top-k grounded concept names
        # result['answer']    → LLM answer (or fallback)
    """

    def __init__(
        self,
        prop_dim: int = _PROP_DIM,
        embed_dim: int = 64,
        use_llm: bool = False,
        llm_model: str = "gpt-3.5-turbo",
        llm_api_key: Optional[str] = None,
        top_k: int = 3,
    ) -> None:
        super().__init__()
        self.grounder = LearnedGrounding(prop_dim=prop_dim, embed_dim=embed_dim)
        self.llm      = LLMReasoner(model=llm_model, api_key=llm_api_key) if use_llm else None
        self.top_k    = top_k

    def forward(
        self,
        props: torch.Tensor,
        question: str = "What is this object?",
    ) -> Dict:
        """
        Args:
            props: [prop_dim] or [B, prop_dim] property vector(s)
        Returns:
            dict with 'concepts', 'answer', 'grounded_props'
        """
        single = props.dim() == 1
        if single:
            props = props.unsqueeze(0)

        results = []
        for i in range(props.shape[0]):
            preds    = self.grounder.percept_to_language(props[i], top_k=self.top_k)
            concepts = [name for name, _ in preds]
            answer   = (
                self.llm.reason(concepts, question)
                if self.llm is not None
                else f"Grounded concepts: {', '.join(concepts)}"
            )
            grounded = {
                name: self.grounder.language_to_percept(name)
                for name in concepts
            }
            results.append({
                "concepts":       concepts,
                "scores":         [score for _, score in preds],
                "answer":         answer,
                "grounded_props": grounded,
            })

        return results[0] if single else results
