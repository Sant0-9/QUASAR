# Phase 5: Evaluation

> Baselines, metrics, comparison, and statistical testing.

---

## Step 5.1: Baseline Ansatzes

**Status**: PENDING

**Prerequisites**: None

### Purpose

Implement standard ansatzes for fair comparison.

### Required Baselines

| Baseline | Description | Source |
|----------|-------------|--------|
| HEA (2-layer) | Hardware-efficient ansatz | Qiskit TwoLocal |
| HEA (3-layer) | Deeper HEA | Qiskit TwoLocal |
| EfficientSU2 | Qiskit's optimized ansatz | Qiskit library |
| Custom HEA | Our implementation | Manual |
| Random | Control baseline | Random gates |

### Specification

Each baseline function takes `num_qubits` and returns `QuantumCircuit`.

All baselines must:
- Have trainable parameters
- Include entanglement (except trivial cases)
- Be deterministic (same input = same output)

### Verification Checklist

- [ ] `src/evaluation/baselines.py` implemented
- [ ] All 5 baselines defined
- [ ] Registry with descriptions
- [ ] `get_baseline(name, num_qubits)` works
- [ ] `get_all_baselines(num_qubits)` works
- [ ] All tests pass (minimum 15 tests)

### Source Files

- `src/evaluation/baselines.py` - Baseline implementations
- `tests/test_baselines.py` - Unit tests

---

## Step 5.2: Metrics Module

**Status**: PENDING

**Prerequisites**: Step 5.1 complete

### Purpose

Compute comprehensive metrics for circuit evaluation.

### CircuitMetrics Dataclass

| Field | Type | Description |
|-------|------|-------------|
| name | str | Circuit identifier |
| num_qubits | int | System size |
| depth | int | Circuit depth |
| gate_count | int | Total gates |
| param_count | int | Trainable parameters |
| two_qubit_gates | int | CX/CZ/SWAP count |
| energy | float | Optimized energy |
| energy_error | float | Absolute error vs exact |
| relative_error_percent | float | Percentage error |
| gradient_variance | float | BP indicator |
| trainability | str | high/medium/low |
| vqe_iterations | int | Steps to converge |
| converged | bool | VQE converged? |
| fidelity | float (optional) | State overlap |
| hardware_energy | float (optional) | IBM result |

### StatisticalMetrics Dataclass

Aggregates multiple trials:
- mean_energy, std_energy
- mean_error, std_error
- min_error, max_error
- convergence_rate
- all_results (list)

### Verification Checklist

- [ ] `src/evaluation/metrics.py` implemented
- [ ] CircuitMetrics with 15+ fields
- [ ] StatisticalMetrics with aggregation
- [ ] `compute_circuit_metrics()` works
- [ ] `compute_statistical_metrics()` runs multiple trials
- [ ] `format_metrics_table()` produces markdown
- [ ] All tests pass (minimum 10 tests)

### Source Files

- `src/evaluation/metrics.py` - Metrics implementation
- `tests/test_metrics.py` - Unit tests

---

## Step 5.3: Comparative Evaluation

**Status**: PENDING

**Prerequisites**: Steps 5.1 and 5.2 complete

### Purpose

Compare discovered circuits against all baselines.

### ComparisonResult Dataclass

| Field | Type | Description |
|-------|------|-------------|
| discovered_name | str | Discovered circuit name |
| hamiltonian_name | str | Target Hamiltonian |
| num_qubits | int | System size |
| num_trials | int | Statistical trials |
| discovered | StatisticalMetrics | Discovered results |
| baselines | Dict[str, StatisticalMetrics] | Baseline results |
| improvements | Dict[str, Dict] | Improvement percentages |
| best_baseline_name | str | Strongest competitor |
| beats_all_baselines | bool | Success flag |
| best_improvement_percent | float | Maximum improvement |

### Improvements Dict Structure

For each baseline:
- error_reduction_percent
- depth_reduction_percent
- gate_reduction_percent

### Output Formats

1. Markdown comparison report
2. JSON data file for further analysis
3. LaTeX table for paper

### Verification Checklist

- [ ] `src/evaluation/compare.py` implemented
- [ ] ComparisonResult dataclass complete
- [ ] `compare_to_baselines()` runs end-to-end
- [ ] `format_comparison_report()` produces markdown
- [ ] `save_comparison()` creates valid JSON
- [ ] All tests pass (minimum 12 tests)

### Source Files

- `src/evaluation/compare.py` - Comparison implementation
- `tests/test_compare.py` - Unit tests

---

## Step 5.4: Statistical Tests

**Status**: PENDING

**Prerequisites**: Step 5.3 complete

### Purpose

Determine if improvements are statistically significant.

### Tests to Implement

| Test | Use Case | Output |
|------|----------|--------|
| Paired t-test | Normal data, same seeds | p-value, t-stat |
| Wilcoxon | Non-normal data | p-value, W-stat |
| Bootstrap CI | Any distribution | Confidence interval |

### SignificanceResult Dataclass

| Field | Type | Description |
|-------|------|-------------|
| test_name | str | Test identifier |
| statistic | float | Test statistic |
| p_value | float | Significance |
| significant | bool | p < 0.05 |
| effect_size | float | Cohen's d or r |
| effect_interpretation | str | small/medium/large |
| confidence_interval | Tuple | (lower, upper) |

### Effect Size Interpretation

| Cohen's d | Interpretation |
|-----------|----------------|
| < 0.5 | Small |
| 0.5 - 0.8 | Medium |
| >= 0.8 | Large |

### Verification Checklist

- [ ] `src/evaluation/statistics.py` implemented
- [ ] `test_improvement_significance()` works
- [ ] `wilcoxon_test()` works
- [ ] `bootstrap_confidence_interval()` works
- [ ] `format_significance_report()` produces markdown
- [ ] All tests pass (minimum 10 tests)

### Source Files

- `src/evaluation/statistics.py` - Statistics implementation
- `tests/test_statistics.py` - Unit tests

---

## Evaluation Protocol

### For Each Discovery Campaign

1. Run discovery with 10+ random seeds
2. Compute metrics for best discovered circuit
3. Run same 10 seeds for each baseline
4. Compare using statistical tests
5. Report:
   - Mean +/- std for all methods
   - Improvement percentages
   - Statistical significance
   - Effect sizes

### Minimum Requirements

| Requirement | Value |
|-------------|-------|
| Trials per circuit | 10 |
| Random seeds | Shared across methods |
| Significance level | 0.05 |
| Report format | Markdown + JSON |

---

## After Completion

Update `00_OVERVIEW.md`:
- Phase 5 progress: 4/4
- Mark all steps as DONE

Next: Proceed to `06_HARDWARE.md`
