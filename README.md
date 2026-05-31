# QUASAR: QUantum Algorithm Search via Augmented Reasoning

> An autonomous agent that discovers novel quantum circuits by pairing a physics-augmented LLM proposer with a learned surrogate filter and VQE, using classical physics-simulation data (Polymathic AI's The Well) to guide the search.

---

## What QUASAR Does

QUASAR takes a physics problem and discovers quantum circuits that solve it:

1. **Input**: Physics goal (e.g., "Find ground state of 6-qubit MHD lattice")
2. **Propose**: Physics-augmented LLM generates candidate circuits informed by conservation laws and symmetries
3. **Filter**: A learned surrogate scores candidate circuits far more cheaply than VQE, so only the most promising are optimized
4. **Evaluate**: Top 10% undergo full VQE optimization
5. **Validate**: Best circuits run on IBM quantum hardware
6. **Output**: Novel circuits that outperform human-designed ansatzes

---

## Demo defaults vs full run

Out of the box, QUASAR runs in a **demo configuration** so it is runnable without a GPU, an IBM Quantum account, or large downloads:

- The discovery loop uses a **mock proposer** (`use_mock_proposer=True`), not the fine-tuned LLM.
- The surrogate is **untrained** — no model checkpoint is bundled.
- **The Well** dataset is **not bundled** (it is ~15TB; download it separately with `the-well-download`).

A full discovery run requires downloading The Well, training/loading a surrogate checkpoint, the fine-tuned physics-augmented LLM, and IBM Quantum credentials. The goals below describe that full configuration and have not yet been benchmarked.

---

## Architecture

```
THE WELL (sims) ──► PHYSICS ENCODER ──┬──► SURROGATE EVALUATOR (fast)
                                      │
                                      ├──► PHYSICS-AUGMENTED LLM TRAINING
                                      │
                                      └──► NEW HAMILTONIAN TARGETS (MHD)
                                              │
                                              ▼
                    ┌──────────────────────────────────────────┐
                    │            QUANTUMMIND AGENT             │
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

## Core Components

### Surrogate Evaluator

Predicts circuit quality without running VQE. Trained on physics dynamics from The Well, then fine-tuned on quantum circuit data.

| Metric | Design target |
|--------|---------------|
| Inference cost | far cheaper than a VQE run per circuit |
| Accuracy | R^2 > 0.7 once trained |
| Role | cheaply screen candidates before expensive VQE |

### Physics-Augmented LLM

Fine-tuned Qwen2.5-Coder-7B that understands physics, not just code patterns. Training data includes:

- **50%** QuantumLLMInstruct (code patterns)
- **30%** Physics-augmented examples from The Well
- **20%** Physics reasoning chains

### Barren Plateau Detector

Screens circuits before VQE to avoid untrainable parameter landscapes. Catches >90% of barren plateaus using gradient variance analysis.

### Hardware Validation

Discovered circuits run on IBM Quantum hardware (ibm_sherbrooke, 127 qubits) with ZNE error mitigation.

---

## The Well Integration

QUASAR uses The Well dataset (15TB of physics simulations from Polymathic AI) for three purposes:

| Purpose | How |
|---------|-----|
| Surrogate pretraining | Learn physics dynamics before quantum circuit data |
| LLM augmentation | Generate physics-to-circuit training examples |
| New targets | Map MHD/turbulence simulations to quantum Hamiltonians |

### Supported Datasets

| Dataset | Domain | Use |
|---------|--------|-----|
| shear_flow | Fluid dynamics | Initial proof of concept |
| MHD_64 | Magnetohydrodynamics | Novel discovery target |
| turbulence_gravity_cooling | Astrophysics | Many-body correlations |
| rayleigh_benard | Convection | Thermal physics |

---

## Discovery Targets

### Toy Problems (Validation)

| Hamiltonian | Qubits | Purpose |
|-------------|--------|---------|
| XY Chain | 4, 6, 8 | Validate against known solutions |
| Heisenberg | 4, 6 | Test physics reasoning |
| TFIM | 4, 6, 8 | Multiple field strengths |

### Novel Targets (Contribution)

| Hamiltonian | Source | Why Novel |
|-------------|--------|-----------|
| MHD_LATTICE | The Well MHD_64 | First quantum circuits for MHD |

MHD circuits are validated against The Well ground truth, targeting >80% fidelity.

---

## Directory Structure

```
quantum-mind/
├── src/
│   ├── quantum/           # Hamiltonians, BP detector, verifier, executor
│   ├── agent/             # Proposer, memory, analyzer, discovery loop
│   ├── training/          # Dataset utilities, fine-tuning
│   ├── evaluation/        # Baselines, metrics, statistical tests
│   └── quasar/            # Surrogate, Well loader, physics encoder
├── data/
│   ├── the_well/          # The Well datasets (HDF5)
│   ├── processed/         # Training data
│   └── surrogate/         # Surrogate training data
├── experiments/
│   ├── xy_chain/          # XY chain discovery campaigns
│   ├── heisenberg/        # Heisenberg campaigns
│   ├── tfim/              # TFIM campaigns
│   └── mhd/               # MHD discovery (novel)
├── models/
│   └── checkpoints/       # Fine-tuned model weights
├── guidelines/            # Build specifications (no code)
└── paper/                 # Research paper materials
```

---

## Evaluation

### Baselines

| Baseline | Description |
|----------|-------------|
| HEA (2-layer) | Hardware-efficient ansatz |
| HEA (3-layer) | Deeper HEA |
| EfficientSU2 | Qiskit optimized ansatz |
| Random | Control baseline |

### Metrics

| Metric | Description |
|--------|-------------|
| Energy error | Absolute difference from exact ground state |
| Relative error | Percentage error |
| Circuit depth | Total layers |
| Two-qubit gates | CX/CZ count (hardware cost) |
| Trainability | Gradient variance (BP indicator) |

### Statistical Testing

All comparisons use paired t-tests or Wilcoxon signed-rank with 10+ trials per circuit. Effect sizes reported as Cohen's d.

---

## Hardware Validation

Circuits run on IBM Quantum with:

| Parameter | Value |
|-----------|-------|
| Backend | ibm_sherbrooke (127 qubits) |
| Resilience level | 2 (ZNE) |
| Shots | 4000 per circuit |
| Runs | 5x for statistics |

Each hardware run records job ID, backend calibration date, transpiled depth, and raw/mitigated energies.

---

## Goals

Design goals, not yet benchmarked (the default config runs a mock proposer and an untrained surrogate — see [Demo defaults vs full run](#demo-defaults-vs-full-run)):

| Goal | Rationale |
|------|-----------|
| Faster exploration | A cheap surrogate filter lets the agent screen far more candidates than running VQE on each |
| Physics-augmented improvement | Target lower energy error than a code-only LLM |
| Novel MHD circuits | First quantum circuits targeting magnetohydrodynamics |
| Hardware validation | Circuits run on IBM Quantum hardware (ibm_sherbrooke) |

---

## Requirements

- Python 3.10+
- IBM Quantum account (free tier works)
- GPU for training (A100 recommended)

### Dependencies

```
qiskit
qiskit-ibm-runtime
qiskit-aer
torch
transformers
the-well
h5py
unsloth
peft
scipy
```

---

## References

### Papers

- [Agent-Q: Fine-Tuning LLMs for Quantum Circuits](https://arxiv.org/abs/2504.11109)
- [LLM-Discovered Ansatzes](https://arxiv.org/html/2505.06347)
- [Barren Plateaus Two-Step Solution](https://arxiv.org/abs/2601.18060)
- [QuantumLLMInstruct Dataset](https://arxiv.org/abs/2412.20956)
- [The Well: 15TB Physics Simulations](https://polymathic-ai.org/the_well)

### Datasets

- [QuantumLLMInstruct (500k pairs)](https://huggingface.co/datasets/BoltzmannEntropy/QuantumLLMInstruct)
- [The Well (15TB physics)](https://polymathic-ai.org/the_well)

### Tools

- [IBM Quantum Platform](https://quantum.cloud.ibm.com)
- [Qiskit Documentation](https://quantum.cloud.ibm.com/docs)

---

## Build Guidelines

Detailed specifications in `guidelines/`:

| Phase | Guideline | Content |
|-------|-----------|---------|
| 1 | 01_ENVIRONMENT.md | IBM token, The Well setup |
| 2 | 02_WELL.md | Data loader, physics encoder |
| 3 | 03_FINETUNING.md | Physics-augmented LLM training |
| 4 | 04_SURROGATE.md | Surrogate model, discovery integration |
| 5 | 05_EVALUATION.md | Baselines, metrics, statistics |
| 6 | 06_HARDWARE.md | IBM runner, error mitigation |
| 7 | 07_DISCOVERY.md | Discovery campaigns (XY, Heisenberg, TFIM, MHD) |
| 8 | 08_PAPER.md | Results documentation, paper draft |

---

## The Contribution

QUASAR fills a gap: no one has used large-scale classical physics simulation data to improve quantum algorithm discovery.

| Current State | QUASAR Approach |
|---------------|-----------------|
| LLMs trained on code only | LLMs trained on code + physics dynamics |
| VQE evaluated per circuit (slow) | Surrogate filters many candidates before VQE |
| Toy problems only | Real physics targets (MHD) |
| No physics grounding | Physics-informed from classical simulation data (The Well) |
