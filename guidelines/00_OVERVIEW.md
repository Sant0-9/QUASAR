# QUASAR Guideline: Overview

> Quick reference for project state. Read only this file to know where we are.

---

## Project Summary

**What**: QUASAR (QUantum Algorithm Search via Augmented Reasoning) is an autonomous agent that discovers novel quantum circuits using a fine-tuned LLM, barren plateau detection, iterative feedback, IBM quantum hardware validation, surrogate evaluation, and physics-augmented training from The Well dataset.

**Goal**: Discover circuits for physics problems that outperform human-designed ansatzes by 10%+ in energy error while using fewer gates. Use 15TB of classical physics simulations to make discovery 100x faster.

**Claim**: "I built QUASAR, a system that uses 15TB of classical physics simulations to accelerate quantum circuit discovery by 100x. The physics-informed surrogate predicts circuit quality in milliseconds, enabling exploration of 10,000+ candidates. The physics-augmented LLM proposes circuits that achieve 25% lower energy error than baseline."

---

## Architecture

```
THE WELL (15TB) ──► PHYSICS ENCODER ──┬──► SURROGATE EVALUATOR (10ms)
                                      │
                                      ├──► PHYSICS-AUGMENTED LLM TRAINING
                                      │
                                      └──► NEW HAMILTONIAN TARGETS (MHD)
                                              │
                                              ▼
                    ┌──────────────────────────────────────────┐
                    │              QUASAR AGENT                │
                    │                                          │
                    │  Physics Goal ──► LLM Proposer ──► Verifier
                    │                       │                  │
                    │                       │    ┌─────────────┘
                    │                       ▼    ▼
                    │                   SURROGATE FILTER (fast)
                    │                       │
                    │                       ▼ (top 10% only)
                    │                   VQE EXECUTOR (slow)
                    │                       │
                    │                       ▼
                    │                   MEMORY + LEARNING
                    └──────────────────────────────────────────┘
```

---

## Current State

### Completed (QUASAR Core)

| Component | Source File | Tests | Status |
|-----------|-------------|-------|--------|
| Hamiltonians | `src/quantum/hamiltonians.py` | 29 | DONE |
| Barren plateau detector | `src/quantum/barren_plateau.py` | 17 | DONE |
| Circuit verifier | `src/quantum/verifier.py` | 25 | DONE |
| Quantum executor | `src/quantum/executor.py` | 21 | DONE |
| Dataset utilities | `src/training/dataset.py` | 24 | DONE |
| LLM Proposer | `src/agent/proposer.py` | 25 | DONE |
| Memory system | `src/agent/memory.py` | 23 | DONE |
| Result analyzer | `src/agent/analyzer.py` | 21 | DONE |
| Discovery agent | `src/agent/discovery.py` | 18 | DONE |
| Baselines | `src/evaluation/baselines.py` | - | DONE |
| Metrics | `src/evaluation/metrics.py` | - | DONE |

### Pending (QUASAR Extension)

| Component | Target File | Guideline | Status |
|-----------|-------------|-----------|--------|
| The Well data loader | `src/quasar/well_loader.py` | `02_WELL.md` | PENDING |
| Physics encoder | `src/quasar/physics_encoder.py` | `02_WELL.md` | PENDING |
| Surrogate model | `src/quasar/surrogate.py` | `04_SURROGATE.md` | PENDING |
| Circuit encoder | `src/quasar/circuit_encoder.py` | `04_SURROGATE.md` | PENDING |
| Surrogate-VQE integration | `src/agent/discovery.py` | `04_SURROGATE.md` | PENDING |
| Physics-augmented data | `src/quasar/augmented_data.py` | `03_FINETUNING.md` | PENDING |
| PDE-to-Hamiltonian mapper | `src/quasar/hamiltonian_map.py` | `07_DISCOVERY.md` | PENDING |
| MHD discovery campaign | `experiments/mhd/` | `07_DISCOVERY.md` | PENDING |

---

## Phase Progress

| Phase | Steps | Progress | Guideline |
|-------|-------|----------|-----------|
| 1. Environment | IBM token, The Well deps | 1/2 | `01_ENVIRONMENT.md` |
| 2. The Well Integration | Data loader, physics encoder | 0/2 | `02_WELL.md` |
| 3. Fine-Tuning | Physics-augmented training | 0/3 | `03_FINETUNING.md` |
| 4. Surrogate | Model, integration | 0/3 | `04_SURROGATE.md` |
| 5. Evaluation | Baselines, metrics, stats | 0/4 | `05_EVALUATION.md` |
| 6. Hardware | IBM runner, error mitigation | 0/3 | `06_HARDWARE.md` |
| 7. Discovery | XY, Heisenberg, TFIM, MHD | 0/4 | `07_DISCOVERY.md` |
| 8. Paper | Results, draft | 0/2 | `08_PAPER.md` |
| **TOTAL** | | **1/23** | |

---

## Execution Order

**Week 1**
1. Phase 1: Environment (The Well dependencies)
2. Phase 2: The Well Integration

**Week 2**
3. Phase 4: Surrogate (can start parallel with Phase 2)
4. Phase 3: Fine-Tuning (physics-augmented)

**Week 3**
5. Phase 5: Evaluation
6. Phase 6: Hardware
7. Phase 7: Discovery (toy problems + MHD)

**Week 4**
8. Phase 8: Paper

---

## Rules (Non-Negotiable)

1. **No code in guidelines** - Code lives in source files, guidelines reference them
2. **100% complete or not done** - No MVPs, no skeletons
3. **All checks must pass** - Before moving to next step
4. **Sequential execution** - No skipping ahead (except marked parallel tasks)
5. **Test everything** - No untested code
6. **Document as you go** - Not at the end

---

## Directory Structure

```
quantum-mind/
├── src/
│   ├── quantum/          # Hamiltonians, BP detector, verifier, executor
│   ├── agent/            # Proposer, memory, analyzer, discovery
│   ├── training/         # Dataset, fine-tuning scripts
│   ├── evaluation/       # Baselines, metrics, comparison
│   └── quasar/           # Surrogate, Well loader, physics encoder
├── data/
│   ├── the_well/         # The Well datasets
│   ├── processed/        # Training data
│   └── surrogate/        # Surrogate training data
├── experiments/
│   ├── xy_chain/
│   ├── heisenberg/
│   ├── tfim/
│   └── mhd/              # NEW: MHD discovery
├── models/
│   └── checkpoints/
├── guidelines/           # This folder - no code here
└── paper/
```

---

## Key Metrics to Hit

| Metric | Target | Measured By |
|--------|--------|-------------|
| Surrogate speedup | 100x | Circuits explored per hour |
| Surrogate accuracy | R^2 > 0.7 | Predicted vs actual energy error |
| LLM improvement | 20% lower error | Physics-augmented vs base |
| Beat baselines | All 5 | Energy error comparison |
| Hardware validation | Confirms simulation | IBM Quantum results |
| New target fidelity | >80% | MHD circuit vs classical |

---

## How to Use These Guidelines

1. Check this file (`00_OVERVIEW.md`) to see current progress
2. Read ONLY the guideline doc for the current phase
3. Implement code in the referenced source files
4. Complete ALL verification checks before moving on
5. Update this overview when a phase is complete

---

*This file is the index. Individual phase docs contain specifications and verification checklists. Code lives in source files.*
