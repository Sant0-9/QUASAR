# Phase 7: Discovery Campaigns

> Run discovery on toy problems and new physics targets.

---

## Step 7.1: XY Chain Discovery

**Status**: PENDING

**Prerequisites**: All previous phases complete

### Purpose

Discover circuits for XY spin chain ground states.

### Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Hamiltonian | XY_CHAIN |
| Qubit counts | 4, 6, 8 |
| Max iterations | 100, 150, 200 |
| Target error | 0.01, 0.01, 0.02 |
| VQE max iterations | 200, 250, 300 |
| Hardware validation | Yes |

### Directory Structure

```
experiments/xy_chain/
├── config.yaml
├── run_discovery.py
├── results/
│   ├── 4q/
│   ├── 6q/
│   └── 8q/
└── logs/
```

### Output Files Per Target

| File | Content |
|------|---------|
| results.json | All metrics and data |
| best_circuit.py | Discovered circuit code |
| comparison.json | Baseline comparison data |
| comparison_report.md | Formatted report |
| significance_report.md | Statistical tests |
| hardware_results.md | IBM validation |

### Verification Checklist

- [ ] `experiments/xy_chain/` directory created
- [ ] `config.yaml` with all parameters
- [ ] Discovery runs for 4, 6, 8 qubits
- [ ] At least one circuit beats all baselines
- [ ] Results saved to `results/`
- [ ] Circuit code saved
- [ ] Comparison reports generated
- [ ] Statistical tests completed
- [ ] Hardware validation completed
- [ ] `campaign_summary.json` generated

### Source Files

- `experiments/xy_chain/config.yaml` - Configuration
- `experiments/xy_chain/run_discovery.py` - Runner script

---

## Step 7.2: Heisenberg Discovery

**Status**: PENDING

**Prerequisites**: Step 7.1 complete

### Purpose

Discover circuits for Heisenberg model ground states.

### Key Differences from XY

- More complex Hamiltonian (XXX interactions)
- Higher expected errors
- Only 4 and 6 qubits (8 too expensive)
- More VQE iterations needed

### Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Hamiltonian | HEISENBERG |
| Qubit counts | 4, 6 |
| Max iterations | 150, 200 |
| Target error | 0.015, 0.02 |

### Verification Checklist

- [ ] `experiments/heisenberg/` directory created
- [ ] Discovery runs for 4 and 6 qubits
- [ ] Results saved
- [ ] Comparison complete
- [ ] Statistical tests complete
- [ ] Hardware validation complete
- [ ] `campaign_summary.json` generated

---

## Step 7.3: TFIM Discovery

**Status**: PENDING

**Prerequisites**: Step 7.2 complete

### Purpose

Discover circuits for transverse-field Ising model.

### Key Features

- Includes transverse field parameter h
- Test multiple field strengths
- Critical point (h ~ J) is hardest

### Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Hamiltonian | TFIM |
| Qubit counts | 4, 6, 8 |
| Field strengths | h = 1.0, 0.5 |
| Max iterations | 100-200 |

### Verification Checklist

- [ ] `experiments/tfim/` directory created
- [ ] Discovery runs for all qubit counts
- [ ] Multiple field strengths tested
- [ ] Results saved
- [ ] Comparison complete
- [ ] Hardware validation complete
- [ ] `campaign_summary.json` generated

---

## Step 7.4: MHD Discovery (QUASAR Novel Target)

**Status**: PENDING

**Prerequisites**: Steps 7.1-7.3 complete, Phase 2 complete (The Well)

### Purpose

Discover circuits for magnetohydrodynamics problems derived from The Well.

### This Is The Novel Contribution

No one has done quantum circuit discovery for MHD. This is where QUASAR differentiates itself.

### PDE-to-Hamiltonian Mapping

1. Extract MHD dynamics from `MHD_64` dataset
2. Discretize on lattice (4-8 qubits for tractability)
3. Map field variables to qubit states
4. Express interactions as Hamiltonian terms

### New Hamiltonian: MHD_LATTICE

| Term | Physical Origin |
|------|-----------------|
| ZZ interactions | Magnetic pressure |
| XX + YY | Field line tension |
| Transverse field | External driving |

### Validation Strategy

Since no exact solution exists:
1. Run discovered circuit
2. Map quantum state back to classical field
3. Compare to The Well ground truth
4. Compute fidelity metric

### Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Hamiltonian | MHD_LATTICE (new) |
| Qubit counts | 4, 6 |
| Max iterations | 200 |
| Validation | Against The Well |

### Verification Checklist

- [ ] `src/quasar/hamiltonian_map.py` created
- [ ] MHD_LATTICE Hamiltonian implemented
- [ ] Added to Hamiltonian registry
- [ ] `experiments/mhd/` directory created
- [ ] Discovery runs for 4 and 6 qubits
- [ ] Validation against The Well completed
- [ ] Fidelity > 80% achieved (target)
- [ ] Results documented

### Source Files

- `src/quasar/hamiltonian_map.py` - PDE to Hamiltonian mapping
- `src/quantum/hamiltonians.py` - Add MHD_LATTICE
- `experiments/mhd/` - MHD discovery campaign

---

## Campaign Summary Script

After all campaigns, aggregate results:

1. Load all `campaign_summary.json` files
2. Compute totals:
   - Total circuits discovered
   - Circuits beating all baselines
   - Hardware-validated circuits
3. Generate `experiments/overall_summary.json`

---

## Success Criteria

### Minimum Success

- [ ] At least 5 circuits discovered
- [ ] At least 3 beat all baselines
- [ ] At least 2 hardware-validated
- [ ] XY, Heisenberg, TFIM all have discoveries

### Target Success

- [ ] 10+ circuits discovered
- [ ] 7+ beat all baselines
- [ ] 5+ hardware-validated
- [ ] MHD discovery with >80% fidelity
- [ ] Best improvement >20%

### Stretch Success

- [ ] Novel circuit architecture identified
- [ ] MHD results publishable standalone
- [ ] Framework generalizes to other physics

---

## After Completion

Update `00_OVERVIEW.md`:
- Phase 7 progress: 4/4
- Mark all steps as DONE

Next: Proceed to `08_PAPER.md`
