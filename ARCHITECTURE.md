# ARCHITECTURE.md — NSCA Demo System

**Developer:** Elina Zhao  
**Client:** Zhuxirui  
**Project:** Neuro-Symbolic Cognitive Architecture (NSCA) Interactive Demo

---

## 1. System Overview

The NSCA Demo System is a Python-based application that implements and exposes the 5-layer Neuro-Symbolic Cognitive Architecture through an interactive web interface. The system allows researchers and demo viewers to run the training pipeline, observe layer-by-layer information flow, configure modules, and compare NSCA against a baseline on sample efficiency benchmarks.

---

## 2. Tech Stack Justification

| Component | Choice | Justification |
|-----------|--------|---------------|
| **ML Framework** | PyTorch 2.0+ | Native support for dynamic computation graphs; ideal for the custom layer-wise architecture with adaptive physics priors and slot attention |
| **Physics Simulation** | MuJoCo (via `gymnasium`) | Industry-standard for robotic and physics-based RL tasks; required for Meta-World ablation benchmark |
| **Web Demo** | Streamlit | Fastest path from Python ML code to interactive UI; no frontend expertise needed; supports live parameter adjustment and real-time chart updates |
| **Language** | Python 3.9+ | Universal ML ecosystem compatibility; aligns with existing NSCA codebase |
| **LLM Integration** | OpenAI API (optional) | Layer 4 language grounding can leverage GPT-4o for concept verbalization without manual dictionaries |
| **Experiment Tracking** | Weights & Biases (optional) | Tracks ablation runs; useful for reproducing the reported +7.2% sample efficiency result |

---

## 3. Repository Structure

```
final-project-codebase-zhuxirui677-1/
├── src/
│   ├── cognitive_agent.py          # Unified entry point for all 5 layers
│   ├── priors/                     # Layer 0: Innate perceptual priors
│   │   ├── visual_prior.py
│   │   ├── audio_prior.py
│   │   └── temporal_prior.py
│   ├── encoders/                   # Layer 0: Multi-modal encoders
│   │   ├── vision_encoder.py
│   │   ├── audio_encoder.py
│   │   └── proprio_encoder.py
│   ├── world_model/                # Layer 0: World representation & dynamics
│   │   └── unified_world_model.py
│   ├── semantics/                  # Layer 1: Semantic property extraction
│   │   ├── property_layer.py       # Slot attention + DynamicPropertyBank
│   │   └── affordances.py
│   ├── reasoning/                  # Layer 2: Causal reasoning
│   │   ├── causal_layer.py
│   │   └── intuitive_physics.py    # AdaptivePhysicsPrior
│   ├── motivation/                 # Layer 3: Curiosity & continual learning
│   │   ├── drive_system.py
│   │   ├── intrinsic_reward.py     # Noisy-TV defense
│   │   └── ewc.py                  # Elastic Weight Consolidation
│   └── language/                   # Layer 4: Language grounding
│       └── llm_integration.py      # LearnedGrounding (no hard-coding)
├── diagnostics/
│   ├── noisy_tv_test.py
│   ├── forgetting_test.py
│   ├── balloon_test.py
│   └── slot_discovery_test.py
├── configs/
│   └── default.yaml                # Hyperparameters, toggleable modules
├── app.py                          # Streamlit demo entrypoint
├── train.py                        # Full training pipeline
├── requirements.txt
└── README.md
```

---

## 4. Data Model

There is no persistent database in this system. All state is held in-memory during a session or serialized to disk as checkpoints.

### 4.1 Core Data Structures

**PerceptualInput** — raw multi-modal input to Layer 0
```
vision:      Tensor [B, T, C, H, W]   # RGB video frames
audio:       Tensor [B, samples]        # Raw waveform
proprio:     Tensor [B, T, body_dim]    # Joint angles / forces
```

**WorldState** — output of Layer 0 (World Model)
```
latent:         Tensor [B, D]           # Fused multi-modal embedding
predicted_next: Tensor [B, D]           # Imagined next state
uncertainty:    float                   # Prediction confidence
```

**SemanticProperties** — output of Layer 1
```
hardness:    float
weight:      float
affordances: List[str]
category:    str
slots:       Tensor [B, num_slots, D]   # Dynamic property bank
```

**CausalGraph** — output of Layer 2
```
nodes:    List[str]                     # Object/event identifiers
edges:    List[(str, str, float)]       # Causal links with strengths
```

**DriveState** — output of Layer 3
```
curiosity_level:   float
competence_level:  float
intrinsic_reward:  float
ewc_penalty:       float
```

**LanguageGrounding** — output of Layer 4
```
concept_embeddings: Dict[str, Tensor]   # Word → latent mapping
verbalization:      str                 # Natural language description
```

### 4.2 Checkpoint Schema
Model weights are saved as `.pt` files per phase:
```
checkpoints/
├── phase1_world_model.pt
├── phase2_semantics.pt
├── phase3_reasoning.pt
└── phase4_language.pt
```

---

## 5. Architecture Diagram (C4 — Context Level)

```
┌─────────────────────────────────────────────────┐
│                  User / Researcher               │
└───────────────────────┬─────────────────────────┘
                        │  Browser
                        ▼
┌─────────────────────────────────────────────────┐
│              Streamlit Web Demo (app.py)         │
│  - Parameter sliders (priors on/off, EWC weight) │
│  - Layer-by-layer visualization                  │
│  - Sample efficiency benchmark chart             │
└───────────────────────┬─────────────────────────┘
                        │  Python calls
                        ▼
┌─────────────────────────────────────────────────┐
│           NSCA Cognitive Agent (src/)            │
│                                                  │
│  Layer 4: Language Integration                   │
│      ↑                                           │
│  Layer 3: Motivation / Drive System              │
│      ↑                                           │
│  Layer 2: Causal Reasoning                       │
│      ↑                                           │
│  Layer 1: Semantic Properties                    │
│      ↑                                           │
│  Layer 0: World Model + Priors                   │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│         MuJoCo Simulation Environment            │
│         (physics data for training/eval)         │
└─────────────────────────────────────────────────┘
```

---

## 6. Agentic Engineering Plan

All development will be conducted using **AI-first agentic engineering** — specifically Cursor with Claude as the primary coding assistant.

### 6.1 Approach

Rather than writing implementation code by hand, the development workflow is:

1. **Spec → Prompt:** Each GitHub Issue is translated into a detailed prompt describing the module's inputs, outputs, and constraints.
2. **Generate → Review:** Claude / Cursor Agent generates the module implementation.
3. **Test → Iterate:** Automated tests validate correctness; failing tests are fed back to the agent for correction.
4. **Integrate → PR:** Working modules are committed to a feature branch and submitted as a PR referencing the Issue.

### 6.2 Tools
| Tool | Role |
|------|------|
| **Cursor + Claude** | Primary code generation and refactoring |
| **Claude Code** | Long-context architecture reasoning and multi-file edits |
| **pytest** | Automated test generation and validation |
| **Streamlit** | Rapid UI prototyping with AI-assisted layout |

### 6.3 Development Phases

| Phase | Scope | Target PR | Target Date |
|-------|-------|-----------|-------------|
| **PR #1** | ARCHITECTURE.md, repo structure, module stubs, requirements.txt | PR #1 | Apr 13, 2026 |
| **PR #2** | Layer 0–1: World model + semantic property extraction | PR #2 | May 5, 2026 |
| **PR #3** | Layer 2–3: Causal reasoning + motivation/curiosity + EWC | PR #3 | May 19, 2026 |
| **PR #4** | Layer 4: Language grounding + Streamlit demo + diagnostics | PR #4 | Jun 1, 2026 |

### 6.4 Quality Gates
- Each PR must pass all existing pytest tests before review request
- Each PR must reference its corresponding GitHub Issue
- Client (Zhuxirui) reviews and approves each PR within 48 hours per contract terms

---

## 7. Key Design Decisions

1. **Streamlit over Gradio:** Streamlit provides richer layout control and native support for multi-panel dashboards, which is needed to visualize all 5 layers simultaneously.

2. **Modular toggle design:** Every major module (physics priors, EWC, LLM) is wrapped in a config flag so the client can perform ablation studies via the UI without touching source code.

3. **Scaled-down training for demo:** Full Meta-World training requires ~$200 GPU budget. The demo will use a pre-trained checkpoint for live inference, with a small-scale training run (N=20) available locally to reproduce the +7.2% benchmark.

4. **Existing codebase reuse:** A reference implementation of NSCA already exists. New development will adapt and clean this into the spec-compliant structure, using AI tooling to refactor and modularize.
