#!/usr/bin/env python3
"""
Interactive NSCA demo (SPEC Issue #6).

Run from repository root:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import yaml

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.checkpoint_io import load_cognitive_agent
from src.evaluation.metaworld_eval import AblationStudy, EvaluationConfig


def _load_demo_yaml() -> dict:
    path = _project_root / "configs" / "default.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@st.cache_resource
def _get_agent(checkpoint_path: str | None, use_llm: bool):
    return load_cognitive_agent(checkpoint_path or None, use_llm=use_llm)


def main() -> None:
    st.set_page_config(page_title="NSCA Demo", layout="wide")
    st.title("NSCA: Neuro-Symbolic Cognitive Architecture")
    st.caption("Five-layer demo: world model → semantics → causality → motivation → language.")

    cfg = _load_demo_yaml()
    demo_cfg = cfg.get("demo") or {}
    mod_cfg = cfg.get("modules") or {}

    default_ckpt = demo_cfg.get("cognitive_agent_checkpoint", "checkpoints/cognitive_agent_full.pth")
    ckpt_path = st.sidebar.text_input(
        "Checkpoint path",
        value=str(_project_root / default_ckpt) if not Path(default_ckpt).is_absolute() else default_ckpt,
        help="Download weights to ./checkpoints (see README). Leave path as-is to run untrained smoke test.",
    )
    use_llm = st.sidebar.checkbox(
        "Enable external LLM (Layer 4)",
        value=bool(mod_cfg.get("use_llm", False)),
        help="Requires OPENAI_API_KEY if the language module calls the API.",
    )
    st.sidebar.markdown("**Module toggles (display)**")
    st.sidebar.checkbox(
        "Physics priors (conceptual, display only)",
        value=bool(mod_cfg.get("use_physics_priors", True)),
        disabled=True,
    )
    ewc_lambda = st.sidebar.slider("EWC λ (display only)", 0.0, 1.0, float(mod_cfg.get("ewc_lambda", 0.4)), 0.05)

    path_obj = Path(ckpt_path)
    if not path_obj.exists():
        st.warning(
            f"Checkpoint not found at `{ckpt_path}`. "
            "Place `cognitive_agent_full.pth` or `world_model_final.pth` from your Google Drive under `checkpoints/`, "
            "or run `python scripts/download_checkpoints.py --folder-id <ID>`."
        )

    agent = _get_agent(str(path_obj) if path_obj.exists() else None, use_llm)

    st.subheader("Sample efficiency (synthetic Meta-World ablation)")
    st.caption(
        "Fast statistical simulation from `src/evaluation/metaworld_eval.py` — "
        "not a replacement for full MuJoCo runs. Published headline: **+7.2%** at N=20."
    )
    if st.button("Run quick ablation (few seeds)", type="primary"):
        with st.spinner("Running ablation…"):
            study_cfg = EvaluationConfig(
                tasks=["pick-place-v2"],
                demo_counts=[1, 5, 10, 20],
                num_seeds=5,
            )
            study = AblationStudy(study_cfg)
            summary = study.run()
        rows = []
        for n in study_cfg.demo_counts:
            block = summary["by_demo_count"].get(n, {})
            rows.append(
                {
                    "demos": n,
                    "NSCA (priors)": f"{block.get('priors_mean', 0):.1%}",
                    "Baseline": f"{block.get('random_mean', 0):.1%}",
                    "Cohen's d": f"{summary['effect_sizes'].get(n, {}).get('cohens_d', 0):.2f}",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.json({"ewc_lambda_shown": ewc_lambda, "effect_at_20": summary["effect_sizes"].get(20, {})})

    st.subheader("Layer forward pass (random probe)")
    b, t, c, h, w = 1, 2, 3, 64, 64
    if st.button("Perceive random multi-modal batch"):
        vision = torch.randn(b, t, c, h, w)
        audio = torch.randn(b, 8000)
        proprio = torch.randn(b, t, 12)
        with torch.no_grad():
            out = agent.forward(vision, audio, proprio)
        st.success("Forward pass completed.")

        def _summarize(v):  # noqa: ANN001
            if torch.is_tensor(v):
                return f"Tensor{tuple(v.shape)} {v.dtype}"
            if hasattr(v, "to_tensor") and callable(v.to_tensor):
                t = v.to_tensor()
                return f"{type(v).__name__} → Tensor{tuple(t.shape)}"
            if isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
                return f"list[{len(v)}] of tensors"
            if isinstance(v, dict):
                return {k: _summarize(x) for k, x in list(v.items())[:12]}
            if isinstance(v, (float, int, str, bool)) or v is None:
                return v
            return type(v).__name__

        for key in sorted(out.keys()):
            with st.expander(f"`{key}`", expanded=(key == "world_state")):
                st.code(_summarize(out[key]))

    st.subheader("Reference: reported sample efficiency")
    ref = pd.DataFrame(
        {
            "Samples": [20, 50, 100, 500],
            "Baseline": ["58.1%", "65.0%", "70.7%", "95.6%"],
            "NSCA (priors)": ["65.3%", "70.5%", "75.1%", "94.9%"],
            "Δ": ["+7.2%", "+5.5%", "+4.4%", "-0.7%"],
        }
    )
    st.table(ref)


if __name__ == "__main__":
    main()
