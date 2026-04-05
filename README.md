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
| Developer | Wei Chang |

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

### Run the Demo

```bash
streamlit run app.py
```

### Run Training

```bash
python train.py --config configs/default.yaml
```

### Run Diagnostics

```bash
python diagnostics/run_all.py
```

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
