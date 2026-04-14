"""
NSCA ``layers`` package (GitHub Issue #2 — Setup).

The full implementation lives under ``src/``; this package re-exports stable
symbols so the acceptance command succeeds::

    python -c "from layers import world_model, semantic, causal, motivation, language"
"""

__all__ = ["world_model", "semantic", "causal", "motivation", "language"]
