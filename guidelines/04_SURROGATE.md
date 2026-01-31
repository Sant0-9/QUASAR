# Phase 4: Surrogate Evaluator

> Build neural network to predict circuit quality without VQE.

---

## Why Surrogate?

| Without Surrogate | With Surrogate |
|-------------------|----------------|
| ~100 circuits/day | ~10,000 circuits/day |
| 5 min per circuit (VQE) | 10ms per circuit (surrogate) |
| Limited exploration | Massive exploration |

The surrogate predicts energy error in milliseconds. Only top 10% go to VQE.

---

## Step 4.1: Circuit Encoder

**Status**: PENDING

**Prerequisites**: None (can start immediately)

### Purpose

Convert quantum circuits into fixed-size vectors for the surrogate model.

### Encoding Options

**Option A: Sequence Encoding**
- Treat circuit as sequence of gates
- Each gate = token (type, qubits, parameters)
- Use transformer to encode

**Option B: Graph Encoding (Recommended)**
- Nodes = qubits
- Edges = gates connecting qubits
- Use GNN to encode

### Specification

**Input**: QuantumCircuit object

**Output**: Dense vector, shape `(embedding_dim,)` where `embedding_dim=256`

### Encoded Information

- Gate types and counts
- Circuit depth
- Parameter count
- Entanglement pattern
- Qubit connectivity

### Verification Checklist

- [ ] `src/quasar/circuit_encoder.py` created
- [ ] Can encode any valid QuantumCircuit
- [ ] Output shape is consistent `(256,)`
- [ ] Similar circuits produce similar embeddings
- [ ] Different circuits produce different embeddings
- [ ] All tests pass: `pytest tests/test_circuit_encoder.py -v`

### Source Files

- `src/quasar/circuit_encoder.py` - Encoder implementation
- `tests/test_circuit_encoder.py` - Unit tests

---

## Step 4.2: Surrogate Model

**Status**: PENDING

**Prerequisites**: Step 4.1 complete

### Purpose

Predict circuit quality metrics from circuit + Hamiltonian encoding.

### Architecture

```
Circuit ──► Circuit Encoder ──► [256]
                                  │
                                  ├──► Concat ──► MLP ──► Predictions
                                  │
Hamiltonian ──► Ham Encoder ──► [256]

Predictions:
- energy_error (regression)
- trainability (classification: high/medium/low)
- confidence (0-1)
```

### Training Data Sources

1. **Bootstrap**: Run VQE on baseline circuits, collect (circuit, energy_error) pairs
2. **Physics pretrain**: Use The Well to learn physics dynamics
3. **Active learning**: Update as discovery runs

### Specification

**Input**:
- Circuit: QuantumCircuit
- Hamiltonian: SparsePauliOp

**Output**:
- `predicted_error`: float
- `trainability`: str
- `confidence`: float (0-1)

### Training Strategy

1. Generate 1000+ bootstrap examples from baselines
2. Pretrain on physics dynamics (optional, from The Well)
3. Fine-tune on quantum circuit data
4. Continue updating during discovery (active learning)

### Verification Checklist

- [ ] `src/quasar/surrogate.py` created
- [ ] Model architecture defined
- [ ] Can train on bootstrap data
- [ ] R^2 > 0.7 on held-out test set
- [ ] Inference time < 50ms per circuit
- [ ] All tests pass: `pytest tests/test_surrogate.py -v`

### Source Files

- `src/quasar/surrogate.py` - Model implementation
- `data/surrogate/bootstrap.jsonl` - Bootstrap training data
- `models/surrogate/` - Saved model checkpoints
- `tests/test_surrogate.py` - Unit tests

---

## Step 4.3: Discovery Integration

**Status**: PENDING

**Prerequisites**: Step 4.2 complete

### Purpose

Modify discovery loop to use surrogate as fast filter.

### Modified Discovery Flow

```
1. LLM proposes N circuits (N=100)
2. Verifier checks syntax/constraints
3. BP detector screens for trainability
4. SURROGATE scores all valid circuits (NEW)
5. Select top K (K=10) by surrogate score
6. Run VQE only on top K
7. Update surrogate with actual results
8. Update memory
9. Loop
```

### Key Changes to discovery.py

1. Add surrogate model as class attribute
2. Add `score_circuits()` method using surrogate
3. Add `select_top_k()` method
4. Add `update_surrogate()` method for active learning
5. Modify main loop to use surrogate filter

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| N (proposed) | 100 | Explore many options |
| K (VQE'd) | 10 | 10% selection rate |
| Update frequency | Every 10 VQE runs | Keep surrogate fresh |
| Confidence threshold | 0.5 | Skip low-confidence predictions |

### Verification Checklist

- [ ] `src/agent/discovery.py` updated with surrogate
- [ ] Surrogate loads on agent initialization
- [ ] `score_circuits()` returns ranked list
- [ ] Top-K selection works correctly
- [ ] Active learning updates surrogate
- [ ] End-to-end discovery runs with surrogate
- [ ] 10x+ speedup measured (circuits explored per hour)

### Source Files

- `src/agent/discovery.py` - Update with surrogate integration
- `tests/test_discovery_surrogate.py` - Integration tests

---

## Bootstrap Data Generation

### Method

1. Create all baseline circuits for each Hamiltonian
2. Run VQE with 5 random seeds each
3. Record (circuit, hamiltonian, energy_error, converged)
4. Repeat for 4, 6, 8 qubit systems
5. Target: 1000+ examples

### Baseline Circuits

| Baseline | Depths | Total Circuits |
|----------|--------|----------------|
| HEA | 1, 2, 3 | 3 per qubit count |
| EfficientSU2 | 1, 2, 3 | 3 per qubit count |
| Custom HEA | 1, 2 | 2 per qubit count |
| Random | - | 5 per qubit count |

### Hamiltonians

| Hamiltonian | Qubit Counts | Total |
|-------------|--------------|-------|
| XY Chain | 4, 6, 8 | 3 |
| Heisenberg | 4, 6 | 2 |
| TFIM | 4, 6, 8 | 3 |

### Total Bootstrap Examples

~13 circuits x 8 hamiltonians x 5 seeds = 520 minimum
Target: 1000+ with variations

---

## Surrogate Accuracy Requirements

| Metric | Minimum | Target |
|--------|---------|--------|
| R^2 (energy error) | 0.7 | 0.85 |
| Trainability accuracy | 80% | 90% |
| Ranking correlation | 0.7 | 0.9 |

### What Ranking Correlation Means

If surrogate ranks circuits A > B > C by predicted quality,
actual VQE results should mostly agree with that ranking.

Spearman correlation > 0.7 means surrogate is useful for filtering.

---

## After Completion

Update `00_OVERVIEW.md`:
- Phase 4 progress: 3/3
- Mark all steps as DONE

Next: Proceed to `05_EVALUATION.md`
