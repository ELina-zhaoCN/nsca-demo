# NSCA: Neuro-Symbolic Cognitive Architecture Demo System

## Project Description

NSCA is a biologically-inspired cognitive architecture that achieves **sample-efficient machine learning** by combining innate perceptual priors with learned representations. Inspired by how humans develop understanding through structured prior knowledge and curiosity-driven exploration, NSCA integrates five layers: a physics-grounded world model, semantic property extraction, causal reasoning, intrinsic motivation, and sensorimotor-grounded language understanding.

This repository contains the implementation and interactive demo for the NSCA system, developed as the TECHIN 510 Final Project.

**Key result:** Physics priors improve sample efficiency by **+7.2%** at 20 samples compared to baseline approaches.

---

## Team

| Role | Name |
|------|------|
| Proposer (Client) | Zhuxirui |
| Developer | Elina Zhao |

---

## Architecture Overview

```
Layer 4 — Language Integration        (Bidirectional sensorimotor-language grounding)
Layer 3 — Motivation System           (Curiosity, competence, continual learning via EWC)
Layer 2 — Causal Reasoning            (Causal graphs, intuitive physics, counterfactuals)
Layer 1 — Semantic Properties         (Slot attention, affordances, object categories)
Layer 0 — World Model                 (Multi-modal encoders, dynamics prediction, physics priors)
```

---

## Features

- Adaptive physics priors with learnable correction networks
- Dynamic property discovery via slot attention
- Learnability-filtered curiosity (noisy-TV defense)
- Elastic Weight Consolidation (EWC) for continual learning
- Multi-modal encoders: vision, audio, proprioception with cross-modal fusion
- Interactive web demo for exploring NSCA behavior

---

## Quick Start

```bash
git clone https://github.com/GIX-Luyao/final-project-codebase-zhuxirui677-1.git
cd final-project-codebase-zhuxirui677-1
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Issue #2 / Week 4 (Checkpoint 1) acceptance (import smoke test)

The ``layers/`` package **re-exports** the implementation under ``src/`` so the
import path matches the issue text. If you need **strict empty stubs** for
submission, copies are kept under ``layers/_archive_week4_empty_stubs/`` (see
that folder’s ``README.md``).

Run from the repository root (same directory as `layers/` and `src/`):

```bash
python -c "from layers import world_model, semantic, causal, motivation, language"
```

This should print nothing and exit with code 0.

If `pip install -r requirements.txt` fails on **MuJoCo**, check [MuJoCo install](https://mujoco.readthedocs.io/en/stable/python.html#install-mujoco) for your OS; you can temporarily comment out the `mujoco` line for CPU-only demos that do not load physics envs.

### Run the Demo

```bash
streamlit run app.py
```

Or: `bash demo/run_demo.sh` (see `demo/README.md`).

### Run Training

```bash
python train.py --config configs/default.yaml
```

### Run Diagnostics

```bash
python diagnostics/run_all.py
```

### Verify installation

```bash
python verify_world_model.py
```

### Full stack training (after Layer 0)

Uses weights from `checkpoints/world_model_final.pth` (see training repo / Drive).

```bash
python scripts/train_all_layers.py --world-model checkpoints/world_model_final.pth --config configs/training_config_local.yaml
```

### API reference

Public modules are documented in [docs/API.md](docs/API.md).

---

## GitHub issues (from `SPEC.md`)

| # | Issue | What was delivered |
|---|--------|-------------------|
| 1 | **[Setup]** Environment and repo structure | `requirements.txt`, `configs/`, `src/` package layout, `train.py`, `verify_world_model.py` |
| 2 | **[Layer 0–1]** World model and semantics | `src/world_model/`, `src/encoders/`, `src/semantics/` |
| 3 | **[Layer 2–3]** Causal reasoning and motivation | `src/reasoning/`, `src/motivation/`, `src/learning/ewc.py` |
| 4 | **[Layer 4]** Language and optional LLM | `src/language/llm_integration.py` |
| 5 | **[Eval]** Diagnostics and sample-efficiency harness | `diagnostics/run_all.py`, `scripts/*_test.py`, `src/evaluation/metaworld_eval.py` |
| 6 | **[Demo]** Interactive web UI | `app.py` (Streamlit), `configs/default.yaml` demo section |

---

## Trained checkpoints (Google Drive)

Weights from your Colab / GPU runs should be copied into `checkpoints/` (ignored by git). Expected names match the training pipeline in [ELina-zhaoCN/Neuro-Symbolic-Grounding-in-Low-Resource-Regimes](https://github.com/ELina-zhaoCN/Neuro-Symbolic-Grounding-in-Low-Resource-Regimes), for example:

- `world_model_final.pth` — Layer 0 world model
- `cognitive_agent_full.pth` — full five-layer agent (after `scripts/train_all_layers.py`)

If the folder or file is shared on Google Drive (anyone with the link can view), you can pull it locally with:

```bash
pip install gdown
python scripts/download_checkpoints.py --folder-id "<YOUR_DRIVE_FOLDER_ID>" -o checkpoints
```

Then point Streamlit to `checkpoints/cognitive_agent_full.pth` (default in `configs/default.yaml`) or set the path in the sidebar of `app.py`.

---

## Development Timeline

| Milestone | Due Date | Description |
|-----------|----------|-------------|
| **Checkpoint 1** | Week 4 (May 5, 2026) | Initial architecture PR: repo structure, module stubs, environment setup. All layers stubbed and importable. |
| **Checkpoint 2** | Week 7 (May 26, 2026) | Core implementation complete: Layers 0–3 functional. Pre-training diagnostics passing. Sample efficiency benchmark running. |
| **Checkpoint 3** | Week 9 (Jun 9, 2026) | Full system integration: Layer 4 language grounding, interactive web demo operational, documentation complete. |
| **Final Demo Day** | Week 11 (Jun 23, 2026) | Accepted deliverable: all acceptance criteria met, live demo presented. |

---

## Tech Stack

- **ML Framework:** PyTorch 2.0+
- **Physics Simulation:** MuJoCo
- **Demo Interface:** Streamlit / Gradio
- **Language:** Python 3.9+
- **Optional:** LLM API integration

---

## License

MIT
