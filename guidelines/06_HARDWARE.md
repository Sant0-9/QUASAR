# Phase 6: Hardware Validation

> Validate discovered circuits on IBM quantum hardware.

---

## Step 6.1: IBM Hardware Runner

**Status**: PENDING

**Prerequisites**: Phase 1 complete (IBM token), Phase 5 complete

### Purpose

Execute circuits on real IBM quantum processors.

### HardwareResult Dataclass

| Field | Type | Description |
|-------|------|-------------|
| backend_name | str | IBM backend used |
| energy | float | Measured energy |
| std_error | float | Statistical error |
| shots | int | Measurement shots |
| job_id | str | IBM job identifier |
| execution_time_seconds | float | Wall clock time |
| transpiled_depth | int | Post-transpilation depth |
| transpiled_gates | int | Post-transpilation gates |

### IBMHardwareRunner Class

Methods:
- `__init__(backend_name, resilience_level)` - Initialize connection
- `get_backend_info()` - Return backend details
- `transpile(circuit)` - Transpile for hardware
- `estimate_energy(circuit, hamiltonian, params, shots)` - Single run
- `run_multiple(circuit, hamiltonian, params, num_runs, shots)` - Statistics

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| backend_name | ibm_sherbrooke | 127-qubit Eagle |
| resilience_level | 2 | ZNE error mitigation |
| shots | 4000 | Measurements per circuit |
| optimization_level | 3 | Transpilation optimization |

### Verification Checklist

- [ ] `src/quantum/hardware.py` updated
- [ ] IBMHardwareRunner connects to backend
- [ ] `get_backend_info()` returns valid data
- [ ] `transpile()` produces hardware-compatible circuit
- [ ] `estimate_energy()` returns HardwareResult
- [ ] `run_multiple()` executes without error
- [ ] `format_hardware_results()` produces markdown

### Source Files

- `src/quantum/hardware.py` - Hardware runner implementation
- `tests/test_hardware.py` - Unit tests (skip if no token)

---

## Step 6.2: Error Mitigation

**Status**: PENDING

**Prerequisites**: Step 6.1 complete

### Purpose

Configure and compare error mitigation strategies.

### Resilience Levels

| Level | Name | Description | Overhead |
|-------|------|-------------|----------|
| 0 | None | Raw results | 1x |
| 1 | Twirling | Pauli twirling | ~1.5x |
| 2 | ZNE | Zero-noise extrapolation | ~3x |
| 3 | PEC | Probabilistic error cancellation | ~10x |

### ErrorMitigationConfig Dataclass

| Field | Type | Description |
|-------|------|-------------|
| resilience_level | ResilienceLevel | Enum 0-3 |
| zne_noise_factors | List[float] | [1, 3, 5] for ZNE |
| twirling_num_randomizations | int | 32 default |

### Comparison Function

`compare_error_mitigation_levels()`:
- Runs same circuit at levels 0, 1, 2
- Returns dict with energy and error for each
- Identifies best level for this circuit

### Verification Checklist

- [ ] ResilienceLevel enum defined
- [ ] ErrorMitigationConfig dataclass complete
- [ ] `to_estimator_options()` returns valid dict
- [ ] `compare_error_mitigation_levels()` tests all levels
- [ ] `format_mitigation_comparison()` produces markdown
- [ ] Best level identified for test circuit

### Source Files

- `src/quantum/hardware.py` - Add error mitigation config
- `tests/test_error_mitigation.py` - Unit tests

---

## Step 6.3: Noise Analysis

**Status**: PENDING

**Prerequisites**: Steps 6.1 and 6.2 complete

### Purpose

Compare ideal simulation, noisy simulation, and hardware results.

### NoiseAnalysisResult Dataclass

| Field | Type | Description |
|-------|------|-------------|
| circuit_name | str | Identifier |
| num_qubits | int | System size |
| exact_energy | float | Known exact |
| ideal_sim_energy | float | Noiseless simulation |
| noisy_sim_energy | float | Noise model simulation |
| hardware_energy | float | Real hardware |
| ideal_sim_error | float | Ideal vs exact |
| noisy_sim_error | float | Noisy vs exact |
| hardware_error | float | Hardware vs exact |
| noisy_to_ideal_gap | float | Noise model accuracy |
| hw_to_ideal_gap | float | Total degradation |
| hw_to_noisy_gap | float | Simulation fidelity |

### Analysis Functions

- `get_hardware_noise_model(backend_name)` - Extract noise model
- `run_ideal_simulation(circuit, hamiltonian, params)` - Statevector
- `run_noisy_simulation(circuit, hamiltonian, params, noise_model)` - Aer with noise
- `analyze_noise(circuit, hamiltonian, params, exact_energy)` - Full comparison

### Key Insights

1. **Noisy sim close to hardware?** - Noise model is accurate
2. **Hardware worse than noisy sim?** - Additional real-world effects
3. **Error mitigation effective?** - Compare raw vs mitigated

### Verification Checklist

- [ ] `src/evaluation/noise.py` created
- [ ] NoiseAnalysisResult dataclass complete
- [ ] `get_hardware_noise_model()` returns valid NoiseModel
- [ ] `run_ideal_simulation()` produces energy
- [ ] `run_noisy_simulation()` produces energy
- [ ] `analyze_noise()` runs full pipeline
- [ ] `format_noise_report()` produces markdown

### Source Files

- `src/evaluation/noise.py` - Noise analysis implementation
- `tests/test_noise.py` - Unit tests

---

## Hardware Validation Protocol

### For Each Discovered Circuit

1. Run 5x on hardware with resilience level 2
2. Compute mean and std of energy
3. Compare to simulation results
4. Document in results file

### Required Documentation

For each hardware run, record:
- Job ID (for reproducibility)
- Backend name and calibration date
- Shots per circuit
- Resilience level used
- Transpiled circuit depth
- Raw and mitigated energies

### Cost Estimation

| Task | Shots | Circuits | Total Shots |
|------|-------|----------|-------------|
| Single validation | 4000 | 1 | 4000 |
| 5-run statistics | 4000 | 5 | 20000 |
| Mitigation comparison | 4000 | 3 | 12000 |

IBM Quantum free tier: 10 minutes/month
IBM Quantum open plan: More generous limits

---

## After Completion

Update `00_OVERVIEW.md`:
- Phase 6 progress: 3/3
- Mark all steps as DONE

Next: Proceed to `07_DISCOVERY.md`
