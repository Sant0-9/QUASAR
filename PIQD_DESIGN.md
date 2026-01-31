# QUASAR: QUantum Algorithm Search via Augmented Reasoning

## The Invention

**One sentence**: Use 15TB of classical physics simulations (The Well) to make quantum circuit discovery 100x faster and scientifically meaningful.

---

## Why This Is Novel

| Current State | QUASAR Innovation |
|---------------|-----------------|
| LLMs trained on code only | LLMs trained on code + physics dynamics |
| VQE bottleneck (~100 circuits) | Surrogate enables ~10,000 circuits |
| Toy problems (analytically solvable) | Real physics (computationally expensive) |
| No one has done this | First to combine classical sim data with quantum circuit discovery |

**The gap we fill**: No one has used large-scale classical physics simulation data to improve quantum algorithm discovery.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                             QUASAR SYSTEM                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                         THE WELL (15TB)                              │    │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │    │
│   │  │Turbulence│  │   MHD    │  │ Acoustic │  │Biological│            │    │
│   │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │    │
│   └───────┼─────────────┼─────────────┼─────────────┼────────────────────┘    │
│           │             │             │             │                         │
│           └─────────────┴──────┬──────┴─────────────┘                         │
│                                │                                              │
│                    ┌───────────┴───────────┐                                  │
│                    │   PHYSICS ENCODER     │                                  │
│                    │  (learns dynamics)    │                                  │
│                    └───────────┬───────────┘                                  │
│                                │                                              │
│           ┌────────────────────┼────────────────────┐                         │
│           │                    │                    │                         │
│           ▼                    ▼                    ▼                         │
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                  │
│   │   SURROGATE   │   │   AUGMENTED   │   │     NEW       │                  │
│   │   EVALUATOR   │   │   LLM TRAIN   │   │   TARGETS     │                  │
│   │               │   │               │   │               │                  │
│   │ Predicts E_err│   │ Physics +     │   │ MHD problems  │                  │
│   │ in 10ms       │   │ Code training │   │ as Hamiltonians│                  │
│   └───────┬───────┘   └───────┬───────┘   └───────┬───────┘                  │
│           │                   │                   │                          │
│           └───────────────────┼───────────────────┘                          │
│                               │                                              │
│                               ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      QUANTUMMIND AGENT                               │   │
│   │                                                                      │   │
│   │   Physics Goal ──► LLM Proposer ──► Verifier ──► BP Detector        │   │
│   │                         │                              │             │   │
│   │                         │         ┌────────────────────┘             │   │
│   │                         │         │                                  │   │
│   │                         │         ▼                                  │   │
│   │                         │   ┌───────────┐                            │   │
│   │                         │   │ SURROGATE │◄── Fast path (10ms)       │   │
│   │                         │   │  FILTER   │                            │   │
│   │                         │   └─────┬─────┘                            │   │
│   │                         │         │                                  │   │
│   │                         │         ▼ (top 10% only)                   │   │
│   │                         │   ┌───────────┐                            │   │
│   │                         │   │    VQE    │◄── Slow path (minutes)    │   │
│   │                         │   │ EXECUTOR  │                            │   │
│   │                         │   └─────┬─────┘                            │   │
│   │                         │         │                                  │   │
│   │                         │         ▼                                  │   │
│   │                         │   ┌───────────┐                            │   │
│   │                         └──►│  MEMORY   │                            │   │
│   │                             │  SYSTEM   │                            │   │
│   │                             └───────────┘                            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Surrogate Evaluator

### Purpose

Predict circuit quality WITHOUT running VQE. This is the key speedup.

### How It Works

**Input**:
- Circuit structure (encoded as graph or sequence)
- Hamiltonian (encoded)
- Optional: initial parameter values

**Output**:
- Predicted energy error
- Confidence score
- Trainability estimate

### Architecture Options

**Option A: Graph Neural Network (Recommended)**
```python
class CircuitSurrogate(nn.Module):
    """
    GNN that predicts circuit quality.

    Circuit → Graph (qubits=nodes, gates=edges)
    Hamiltonian → Encoded vector
    Output → Predicted energy error
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.circuit_encoder = GATConv(...)  # Graph attention
        self.hamiltonian_encoder = nn.Linear(...)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, circuit_graph, hamiltonian):
        circuit_emb = self.circuit_encoder(circuit_graph)
        ham_emb = self.hamiltonian_encoder(hamiltonian)
        combined = torch.cat([circuit_emb, ham_emb], dim=-1)
        return self.predictor(combined)
```

**Option B: Transformer (simpler)**
```python
class CircuitSurrogateTransformer(nn.Module):
    """
    Treat circuit as sequence of gates.
    """
    def __init__(self):
        self.gate_embedding = nn.Embedding(num_gates, dim)
        self.transformer = nn.TransformerEncoder(...)
        self.predictor = nn.Linear(dim, 1)
```

### Training Data

**Source 1: Your VQE runs**
- Every time QuantumMind runs VQE, log (circuit, hamiltonian, energy_error)
- Bootstrap with baseline circuits (HEA, EfficientSU2)

**Source 2: The Well (key innovation)**
- The Well contains physics dynamics: state(t) → state(t+dt)
- Train surrogate to understand "what makes good physics simulation"
- Transfer this understanding to quantum circuits

### Training Strategy

```python
# Phase 1: Pretrain on The Well physics
surrogate.pretrain(the_well_data)  # Learn physics dynamics

# Phase 2: Fine-tune on quantum circuit data
surrogate.finetune(circuit_vqe_results)  # Learn circuit→energy mapping

# Phase 3: Active learning during discovery
while discovering:
    circuits = llm.propose(N=100)
    predictions = surrogate.predict(circuits)
    top_circuits = select_top(circuits, predictions, k=10)
    actual_results = vqe.run(top_circuits)
    surrogate.update(top_circuits, actual_results)  # Improve surrogate
```

### Expected Speedup

| Without Surrogate | With Surrogate |
|-------------------|----------------|
| 100 VQE runs/day | 10 VQE runs/day |
| 100 circuits explored | 10,000 circuits explored |
| ~5 min per circuit | ~10ms + 5min for top 1% |

**Net effect: 100x more exploration**

---

## Component 2: Physics-Augmented Training

### Purpose

Make the LLM understand physics, not just code syntax.

### Current Training Data (QuantumLLMInstruct)

```json
{
  "instruction": "Generate Bell state circuit",
  "output": "qc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0,1)"
}
```

**Problem**: LLM learns patterns, not physics.

### New Training Data (The Well Augmented)

**Type 1: Physics Description → Circuit**
```json
{
  "instruction": "Design an ansatz for a system with U(1) symmetry (conserved total magnetization) and nearest-neighbor interactions on a 1D chain.",
  "reasoning": "U(1) symmetry means total Sz is conserved. The ansatz should:\n1. Preserve particle number\n2. Use gates that respect the symmetry (XX+YY interactions)\n3. Match the 1D geometry with linear entanglement",
  "output": "def create_ansatz(n):\n    qc = QuantumCircuit(n)\n    # Symmetry-preserving rotations\n    for i in range(n):\n        qc.ry(Parameter(f'θ_{i}'), i)\n    # Linear entanglement (matches 1D chain)\n    for i in range(n-1):\n        qc.cx(i, i+1)\n    return qc"
}
```

**Type 2: Simulation Data → Physics Insight**
```json
{
  "instruction": "This MHD simulation shows magnetic reconnection. The magnetic field lines break and reconnect at t=0.5, releasing energy. What circuit properties would capture this topology change?",
  "reasoning": "Magnetic reconnection involves:\n1. Initial ordered state (aligned field lines)\n2. Sudden topology change (reconnection event)\n3. Energy release\n\nA circuit capturing this needs:\n1. Initial state preparation (product state)\n2. Entanglement generation (topology change)\n3. Ability to represent the energy landscape",
  "output": "Key circuit properties:\n- Layered structure with increasing entanglement depth\n- Local rotations before entanglement (initial state)\n- Non-local gates at critical layer (reconnection)\n- Sufficient parameters to capture energy release"
}
```

**Type 3: Conservation Law → Circuit Constraint**
```json
{
  "instruction": "The system has energy conservation (Hamiltonian is time-independent). How should this constrain the ansatz?",
  "reasoning": "Energy conservation means the Hamiltonian commutes with the time evolution operator. The ansatz should:\n1. Be expressible as exp(-iHt) for some H\n2. Preserve energy expectation values during optimization\n3. Not introduce artificial energy sinks",
  "output": "Constraints:\n1. Use Hamiltonian variational ansatz (HVA) structure\n2. Gates should be generated by Hamiltonian terms\n3. Avoid measurement-based feedback during circuit"
}
```

### Data Generation Pipeline

```python
def generate_physics_augmented_data(well_dataset):
    """
    Generate training examples from The Well.
    """
    examples = []

    for simulation in well_dataset:
        # Extract physics properties
        symmetries = analyze_symmetries(simulation)
        conservation_laws = find_conservation_laws(simulation)
        dynamics = characterize_dynamics(simulation)

        # Generate instruction-response pairs
        example = {
            "instruction": f"Design an ansatz for a system with {symmetries} "
                          f"and {conservation_laws}. The dynamics show {dynamics}.",
            "reasoning": generate_reasoning(symmetries, conservation_laws, dynamics),
            "output": generate_circuit_template(symmetries, conservation_laws)
        }
        examples.append(example)

    return examples
```

### Training Recipe

```yaml
# config/physics_augmented_training.yaml
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
method: QLoRA

datasets:
  - name: QuantumLLMInstruct
    weight: 0.5  # Existing quantum code data
  - name: PhysicsAugmented
    weight: 0.3  # The Well derived data
  - name: PhysicsReasoning
    weight: 0.2  # Custom physics reasoning examples

lora:
  r: 32
  alpha: 64
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  epochs: 3
  batch_size: 16
  learning_rate: 2e-4
  warmup_ratio: 0.1
```

---

## Component 3: New Discovery Targets

### Current Targets (Toy Problems)

| Problem | Qubits | Status |
|---------|--------|--------|
| XY Chain | 4-12 | Analytically solvable |
| Heisenberg | 4-8 | Analytically solvable |
| TFIM | 4-10 | Analytically solvable |

**Problem**: We already know the answers. Discovered circuits are cool but not useful.

### New Targets (From The Well)

| Problem | Source Dataset | Why Interesting |
|---------|----------------|-----------------|
| 2D Turbulence | turbulence_* | Classically expensive, has structure |
| MHD Dynamics | mhd_* | Astrophysics relevance, conservation laws |
| Acoustic Scattering | acoustic_* | Wave physics, interference patterns |
| Reaction-Diffusion | biological_* | Pattern formation, emergent behavior |

### Encoding Physics as Hamiltonians

**Challenge**: The Well has PDE simulations, not quantum Hamiltonians.

**Solution**: Map classical physics to quantum simulation targets.

```python
def pde_to_hamiltonian(pde_simulation):
    """
    Map a classical PDE simulation to a quantum Hamiltonian.

    Strategy:
    1. Discretize the PDE on a lattice
    2. Map field values to qubit states
    3. Express evolution as Hamiltonian
    """
    # Example: 1D wave equation → spin chain
    # ∂²u/∂t² = c² ∂²u/∂x²
    #
    # Discretize: u_i → qubit state
    # Coupling: ∂²/∂x² → nearest-neighbor interaction
    #
    # H = -J Σ (X_i X_{i+1} + Y_i Y_{i+1}) + h Σ Z_i

    lattice_size = pde_simulation.grid_size
    coupling = extract_coupling(pde_simulation)

    return construct_hamiltonian(lattice_size, coupling)
```

### Validation Strategy

For new targets, we validate against The Well's classical simulation:

```python
def validate_quantum_simulation(circuit, pde_ground_truth):
    """
    Compare quantum circuit output to classical simulation.
    """
    # Run quantum circuit
    quantum_state = run_circuit(circuit)

    # Map back to classical field
    quantum_field = state_to_field(quantum_state)

    # Compare to The Well ground truth
    fidelity = compute_fidelity(quantum_field, pde_ground_truth)

    return fidelity
```

---

## The Well Dataset Selection

### Complete Dataset Inventory (16 Datasets, 15TB Total)

| Dataset | Domain | Resolution | Trajectories | Time Steps | Priority |
|---------|--------|-----------|--------------|-----------|----------|
| `shear_flow` | Fluid dynamics | 256x512 | 1,120 | 200 | **START HERE** |
| `rayleigh_benard` | Convection | 512x128 | 1,750 | 200 | **START HERE** |
| `euler_multi_quadrants` | Fluid dynamics | 512x512 | 10,000 | 100 | HIGH |
| `MHD_64` | Magnetohydrodynamics | 64^3 | 100 | 100 | HIGH |
| `MHD_256` | Magnetohydrodynamics | 256^3 | 100 | 100 | HIGH (larger) |
| `gray_scott_reaction_diffusion` | Pattern formation | 128x128 | 1,200 | 1,001 | MEDIUM |
| `acoustic_scattering` | Wave propagation | 256x256 | 8,000 | 100 | MEDIUM |
| `turbulence_gravity_cooling` | Astrophysics/ISM | 64^3 | 2,700 | 50 | HIGH |
| `rayleigh_taylor_instability` | Fluid instability | 128^3 | 45 | 120 | MEDIUM |
| `supernova_explosion_64` | Astrophysics | 64^3 | 1,000 | 59 | STRETCH |
| `supernova_explosion_128` | Astrophysics | 128^3 | 1,000 | 59 | STRETCH |
| `convective_envelope_rsg` | Astrophysics | 256x128x256 | 29 | 100 | STRETCH |
| `post_neutron_star_merger` | Astrophysics | 192x128x66 | 8 | 181 | STRETCH |
| `active_matter` | Biological | 256x256 | 360 | 81 | OPTIONAL |
| `planetswe` | Atmospheric | 256x512 | 120 | 1,008 | OPTIONAL |
| `viscoelastic_instability` | Non-Newtonian | 512x512 | 260 | Variable | OPTIONAL |

### Priority Order

**Tier 1: Start Here (Proof of Concept)**
| Dataset | Why | Physics Connection to Quantum |
|---------|-----|------------------------------|
| `shear_flow` | Simple 2D, many trajectories | Velocity gradients ~ spin interactions |
| `rayleigh_benard` | Clear convection patterns | Thermal states ~ qubit states |

**Tier 2: Main Targets (Novel Contribution)**
| Dataset | Why | Physics Connection to Quantum |
|---------|-----|------------------------------|
| `MHD_64` | Magnetic fields, astrophysics | B-field topology ~ entanglement structure |
| `turbulence_gravity_cooling` | ISM physics, 2,700 trajectories | Many-body correlations ~ quantum correlations |
| `euler_multi_quadrants` | Shocks, 10,000 trajectories | Discontinuities ~ measurement collapse |

**Tier 3: Stretch Goals (High Impact if Achieved)**
| Dataset | Why | Physics Connection to Quantum |
|---------|-----|------------------------------|
| `supernova_explosion_*` | Extreme physics, astrophysics relevance | Explosive dynamics ~ state preparation |
| `post_neutron_star_merger` | Cutting-edge astrophysics | Dense matter ~ strongly correlated systems |

### Data Format

- **Storage**: Self-documenting HDF5 files
- **Shape**: `(n_traj, n_steps, coord1, coord2, [coord3])` in float32
- **Splits**: 80% train / 10% val / 10% test
- **Access**: Unified PyTorch interface via `the_well.data.WellDataset`

### Download Strategy

```bash
# Install
pip install the-well

# Download specific dataset
the-well-download --base-path ./data/the_well --dataset shear_flow --split train
```

```python
# Load in PyTorch
from the_well.data import WellDataset
from torch.utils.data import DataLoader

# Start with shear_flow (good balance of size and complexity)
dataset = WellDataset(
    well_base_path="./data/the_well",
    well_dataset_name="shear_flow",
    well_split_name="train"
)

# Inspect structure
print(f"Dataset size: {len(dataset)}")
print(f"Sample shape: {dataset[0].shape}")

# Create loader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Recommended Starting Point

1. **Day 1**: Download `shear_flow` (~50GB estimate)
2. **Day 2-3**: Explore data structure, visualize trajectories
3. **Day 4-5**: Build physics feature extractor
4. **Week 2**: Move to `MHD_64` for astrophysics angle

---

## Implementation Phases

### Phase 1: Surrogate MVP (Week 1-2)

**Goal**: Build surrogate that predicts energy error faster than VQE.

**Tasks**:
1. Generate training data from existing VQE runs
2. Implement simple transformer surrogate
3. Train and validate
4. Integrate into QuantumMind loop

**Success metric**: Surrogate predicts energy error with R² > 0.7

### Phase 2: The Well Integration (Week 2-3)

**Goal**: Use The Well data to improve surrogate.

**Tasks**:
1. Download shear_flow dataset
2. Extract physics features (symmetries, conservation laws)
3. Pretrain surrogate on physics dynamics
4. Fine-tune on quantum circuit data

**Success metric**: Surrogate R² improves to > 0.85

### Phase 3: Physics-Augmented LLM (Week 3-4)

**Goal**: Make LLM propose better circuits using physics knowledge.

**Tasks**:
1. Generate physics-augmented training examples
2. Fine-tune Qwen with augmented data
3. Compare circuit quality: base vs augmented

**Success metric**: Augmented LLM proposes circuits with 20% lower energy error

### Phase 4: New Targets (Week 4-5)

**Goal**: Discover circuits for real physics problems.

**Tasks**:
1. Encode MHD problem as Hamiltonian
2. Run QUASAR discovery
3. Validate against The Well ground truth

**Success metric**: Discover circuit that simulates MHD with >80% fidelity

### Phase 5: Paper Writing (Week 5-6)

**Goal**: Document and publish.

**Tasks**:
1. Ablation studies (with/without each component)
2. Comparison to baselines
3. Write paper
4. Submit to arXiv

---

## Compute Budget

| Task | GPU Hours | Cost (Vast.ai) |
|------|-----------|----------------|
| Surrogate training | 4h A100 | $12 |
| Surrogate pretraining (The Well) | 8h A100 | $24 |
| Physics-augmented LLM fine-tuning | 6h A100 | $18 |
| Discovery runs (many circuits) | 10h A100 | $30 |
| Experiments and iteration | 8h A100 | $24 |
| **Total** | **36h** | **~$108** |

Fits in your Brev/Vast.ai budget.

---

## Risk Analysis

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Surrogate doesn't generalize | Medium | Active learning, frequent retraining |
| Physics→Hamiltonian mapping fails | Medium | Start with simpler mappings (wave eq) |
| The Well data too large | Low | Use subset, streaming |
| LLM doesn't learn physics | Medium | More reasoning examples, longer training |
| No improvement over baseline | Low | Even documenting failure is publishable |

---

## Success Criteria

### Minimum Viable Success
- Surrogate provides 10x speedup
- Explore 1000+ circuits (vs current ~100)
- Energy error matches baseline

### Target Success
- Surrogate provides 50x speedup
- Physics-augmented LLM beats base by 15%
- Discover circuit for one new physics problem
- Paper submitted to arXiv

### Stretch Success
- 100x speedup enables novel discoveries
- Circuits for MHD that beat classical baselines
- Paper accepted to quantum venue
- Framework adopted by others

---

## The Claim You Make

> "I built QUASAR, a system that uses 15TB of classical physics simulations to accelerate quantum circuit discovery by 100x. The physics-informed surrogate model predicts circuit quality in milliseconds, enabling exploration of 10,000+ candidates. The physics-augmented LLM proposes circuits that achieve 25% lower energy error than baseline. We demonstrate the first quantum circuits for simulating magnetohydrodynamic phenomena, validated against classical simulations from The Well dataset."

That's not just "I used an LLM" - that's **inventing a new paradigm**.

---

## File Structure

```
quantum-mind/
├── src/
│   ├── quasar/                  # QUASAR components
│   │   ├── __init__.py
│   │   ├── surrogate.py         # Surrogate evaluator
│   │   ├── physics_encoder.py   # The Well physics encoding
│   │   ├── hamiltonian_map.py   # PDE → Hamiltonian mapping
│   │   └── augmented_data.py    # Physics-augmented data generation
│   ├── agent/                   # Existing (modified)
│   ├── quantum/                 # Existing
│   └── training/                # Existing (modified)
├── data/
│   ├── the_well/                # The Well datasets
│   ├── surrogate_training/      # Surrogate training data
│   └── physics_augmented/       # Augmented LLM training data
└── experiments/
    ├── surrogate/               # Surrogate experiments
    ├── augmented_llm/           # LLM experiments
    └── new_targets/             # MHD/turbulence experiments
```

---

## Next Steps

1. **Today**: Read The Well documentation, understand data format
2. **This week**: Download shear_flow, build surrogate MVP
3. **Next week**: Integrate with QuantumMind, measure speedup

Let's invent something.
