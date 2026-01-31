# Phase 3: Physics-Augmented Fine-Tuning

> Train LLM on physics + code, not just code.

---

## Step 3.1: Physics-Augmented Data Generation

**Status**: PENDING

**Prerequisites**: Phase 2 complete (physics encoder working)

### Purpose

Generate training examples that teach the LLM physics reasoning, not just code patterns.

### Training Example Types

**Type 1: Physics Description to Circuit**
- Input: Description of system symmetries and conservation laws
- Output: Circuit design that respects those properties

**Type 2: Simulation Data to Physics Insight**
- Input: Description of simulation behavior (from The Well)
- Output: Physics interpretation and circuit implications

**Type 3: Conservation Law to Circuit Constraint**
- Input: Conservation law specification
- Output: Circuit constraints that preserve the symmetry

### Data Generation Pipeline

1. Load trajectory from The Well
2. Run physics encoder to extract features
3. Generate natural language description of physics
4. Generate corresponding circuit design rationale
5. Generate Qiskit code template
6. Quality filter (code must be valid)

### Target: 5000+ physics-augmented examples

### Verification Checklist

- [ ] `src/quasar/augmented_data.py` created
- [ ] Can generate Type 1 examples
- [ ] Can generate Type 2 examples
- [ ] Can generate Type 3 examples
- [ ] Generated code is syntactically valid
- [ ] Generated reasoning is coherent
- [ ] 5000+ examples generated
- [ ] Examples saved to `data/physics_augmented/`

### Source Files

- `src/quasar/augmented_data.py` - Data generation logic
- `data/physics_augmented/train.jsonl` - Generated training data
- `tests/test_augmented_data.py` - Unit tests

---

## Step 3.2: Dataset Mixing

**Status**: PENDING

**Prerequisites**: Step 3.1 complete

### Purpose

Combine existing QuantumLLMInstruct with physics-augmented data.

### Dataset Weights

| Dataset | Weight | Purpose |
|---------|--------|---------|
| QuantumLLMInstruct | 50% | Core quantum code patterns |
| Physics-Augmented | 30% | Physics reasoning |
| Physics-Reasoning | 20% | Chain-of-thought examples |

### Requirements

1. Update `src/training/dataset.py` to load multiple sources
2. Implement weighted sampling
3. Shuffle across sources
4. Maintain train/val split

### Verification Checklist

- [ ] `src/training/dataset.py` updated
- [ ] Can load all three data sources
- [ ] Weighted sampling works correctly
- [ ] Combined dataset size matches expectations
- [ ] Val split maintained

### Source Files

- `src/training/dataset.py` - Update for mixed loading
- `configs/training.yaml` - Dataset weights configuration

---

## Step 3.3: Fine-Tune with Physics Data

**Status**: PENDING

**Prerequisites**: Step 3.2 complete, GPU access

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen2.5-Coder-7B-Instruct |
| Method | QLoRA (4-bit) |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Learning rate | 2e-4 |
| Batch size | 16 (effective) |
| Epochs | 3 |
| Max sequence | 2048 |

### Training Steps

1. Download base model
2. Add LoRA adapters
3. Load mixed dataset
4. Configure training arguments
5. Run training
6. Save model and logs

### Expected Compute

- Time: ~4-6 hours on A100
- Cost: ~$18-25 on Vast.ai

### Verification Checklist

- [ ] Model downloads successfully
- [ ] LoRA adapters added
- [ ] Training completes without OOM
- [ ] Final loss < 0.5
- [ ] Model saved to `models/checkpoints/quantum-mind-v2/`
- [ ] Training log saved with all metrics

### Source Files

- `src/training/finetune.py` - Training script
- `models/checkpoints/quantum-mind-v2/` - Output model
- `models/checkpoints/quantum-mind-v2/training_log.yaml` - Metrics

---

## Step 3.4: Validate Physics-Augmented Model

**Status**: PENDING

**Prerequisites**: Step 3.3 complete

### Validation Suite

Run 100 test prompts comparing:
1. Base model (Qwen2.5-Coder-7B)
2. Code-only fine-tuned (quantum-mind-v1)
3. Physics-augmented (quantum-mind-v2)

### Metrics to Measure

| Metric | Target | Description |
|--------|--------|-------------|
| Syntax validity | >95% | Code compiles |
| Verification pass | >80% | Circuit passes all checks |
| BP-free rate | >60% | No barren plateau detected |
| Physics reasoning | Manual | Quality of explanations |

### Key Comparison

**Physics-augmented should beat code-only on:**
- Circuits for novel physics problems
- Explanations of circuit design choices
- Respecting symmetries when specified

### Verification Checklist

- [ ] Validation runs on 100 prompts
- [ ] Syntax validity > 95%
- [ ] Verification pass > 80%
- [ ] BP-free rate > 60%
- [ ] Physics-augmented beats code-only on physics prompts
- [ ] Results documented in `models/checkpoints/quantum-mind-v2/validation_log.yaml`

### Source Files

- `src/training/validate_model.py` - Validation script
- `src/training/compare_models.py` - Comparison script

---

## Prompt Templates

### Type 1: Physics to Circuit

```
System: You are a quantum computing expert that designs circuits respecting physics principles.

User: Design an ansatz for a system with the following properties:
- Symmetry: U(1) (conserved total magnetization)
- Interactions: Nearest-neighbor on 1D chain
- Constraint: Hardware-efficient with linear connectivity

Explain your reasoning, then provide the Qiskit code.