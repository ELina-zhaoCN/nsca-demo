# SPEC.md — NSCA: Neuro-Symbolic Cognitive Architecture Demo System

## Project Overview

This project proposes the design and implementation of a working demo system for the **Neuro-Symbolic Cognitive Architecture (NSCA)** — a biologically-inspired cognitive system that combines adaptive physics priors, causal reasoning, intrinsic motivation, and sensorimotor-grounded language understanding. The goal is to build a functional, interactive application that showcases the core capabilities of NSCA in a low-resource learning regime.

---

## Problem Statement

Modern deep learning models require massive amounts of labeled data to perform well. This project addresses the question: *Can a cognitive architecture inspired by human development achieve sample-efficient learning by combining innate priors with learned representations?* The developer is tasked with implementing a runnable demo pipeline and interactive visualization of the 5-layer NSCA system, enabling users to observe and evaluate how physics priors, causal reasoning, and curiosity-driven exploration contribute to learning efficiency.

---

## Developer

- **Name:** Elina Zhao
- **Agreed Development Fee:** 30$ GIX Bucks

---

## User Stories

| # | As a… | I want to… | So that… |
|---|--------|------------|---------|
| 1 | Researcher | Run the NSCA training pipeline end-to-end | I can reproduce sample efficiency results |
| 2 | User | View an interactive visualization of all 5 architecture layers | I can understand how information flows through the system |
| 3 | Researcher | Configure and toggle individual modules (e.g., physics priors, EWC) | I can perform ablation studies without modifying source code |
| 4 | Student / Demo Viewer | See a side-by-side comparison of NSCA vs. baseline on sample efficiency | I can immediately grasp the benefit of neuro-symbolic grounding |
| 5 | Developer | Access clear API documentation and modular code | I can extend or integrate NSCA components into other systems |
| 6 | User | Run pre-training diagnostic tests (Noisy TV, Forgetting, Balloon, Slot Discovery) | I can validate that each module works correctly before full training |
| 7 | Instructor / Demo Day Attendee | Interact with a live web-based demo | I can see results without setting up the full environment |

---

## Technical Specifications

### Architecture Requirements

The implementation must include the following 5 layers:

1. **Layer 0 — World Model:** Multi-modal encoders (vision, audio, proprioception), temporal processing, and dynamics prediction with learnable physics correction networks.
2. **Layer 1 — Semantic Properties:** Slot attention for dynamic property discovery; extraction of physical attributes, affordances, and object categories.
3. **Layer 2 — Causal Reasoning:** Causal graph learning, intuitive physics engine, counterfactual simulation.
4. **Layer 3 — Motivation System:** Learnability-filtered curiosity (noisy-TV defense), competence-driven reward, Elastic Weight Consolidation (EWC) for continual learning.
5. **Layer 4 — Language Integration:** Bidirectional sensorimotor-language grounding without manual dictionaries; optional LLM integration.

### Training Protocol

- Phase 1: Random exploration (1,000 steps)
- Phase 2: Competence-driven learning (9,000 steps)
- Evaluation: Meta-World ablation studies (N=20 seeds), Cohen's d effect sizes

### Tech Stack

- Python 3.9+
- PyTorch 2.0+
- MuJoCo (physics simulation)
- Web frontend: Streamlit or Gradio (for interactive demo)
- Optional: LLM API integration (OpenAI / Anthropic)

---

## Acceptance Criteria

Each deliverable must pass the following criteria before acceptance:

| Deliverable | Acceptance Criteria |
|-------------|---------------------|
| Initial Architecture PR | Repo structure, module stubs, environment setup (requirements.txt, README setup instructions) are present and runnable |
| Layer 0–2 Implementation | World model encodes multi-modal inputs; slot attention discovers ≥2 object properties; causal reasoning module runs without error |
| Layer 3–4 Implementation | Curiosity filter rejects noisy-TV signals; EWC reduces forgetting on sequential tasks; language grounding maps ≥5 concepts |
| Pre-training Diagnostics | All 4 diagnostic tests (Noisy TV, Forgetting, Balloon, Slot Discovery) pass with documented results |
| Sample Efficiency Benchmark | NSCA achieves ≥5% accuracy improvement over baseline at N=20 samples (replicate reported 7.2% result) |
| Interactive Demo | Web demo launches locally (`streamlit run app.py` or equivalent); user can adjust at least 2 hyperparameters and see results update |
| Documentation | API reference covers all public modules; README includes setup, training, and demo instructions |

---

## Out of Scope

- Full-scale cloud GPU training (>$200 budget) is not required; a scaled-down version sufficient for demo is acceptable
- Production deployment / hosting is not required
- Custom dataset collection is not required

---

## GitHub Issues

The project is decomposed into the following 6 issues (to be opened on GitHub):

1. **[Setup] Project environment and repository structure**
2. **[Layer 0-1] World model and semantic property extraction**
3. **[Layer 2-3] Causal reasoning and motivation/curiosity system**
4. **[Layer 4] Language grounding and LLM integration**
5. **[Eval] Pre-training diagnostics and sample efficiency benchmark**
6. **[Demo] Interactive web demo interface**
