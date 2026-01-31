"""
Physics-augmented training data generator.

Generates training examples that teach LLMs physics reasoning,
not just code patterns. Uses The Well simulation data and the
physics encoder to create examples that connect physical properties
to quantum circuit design decisions.

Example types:
- Type 1: Physics Description to Circuit
- Type 2: Simulation Data to Physics Insight
- Type 3: Conservation Law to Circuit Constraint
"""

import json
import os
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator

import numpy as np

from src.quasar.physics_encoder import (
    DynamicsType,
    PhysicsEncoder,
    PhysicsEncoderConfig,
    PhysicsFeatures,
    SymmetryType,
)


class ExampleType(str, Enum):
    """Type of physics-augmented training example."""

    PHYSICS_TO_CIRCUIT = "physics_to_circuit"
    SIMULATION_TO_INSIGHT = "simulation_to_insight"
    CONSERVATION_TO_CONSTRAINT = "conservation_to_constraint"


@dataclass
class TrainingExample:
    """
    A single physics-augmented training example.

    Attributes:
        example_type: Type of training example
        instruction: User instruction/prompt
        input_text: Optional additional input context
        output: Expected model output (reasoning + code)
        metadata: Additional metadata (symmetries, conservation, etc.)
    """

    example_type: ExampleType
    instruction: str
    input_text: str
    output: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.example_type.value,
            "instruction": self.instruction,
            "input": self.input_text,
            "output": self.output,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingExample":
        """Create from dictionary."""
        return cls(
            example_type=ExampleType(data["type"]),
            instruction=data["instruction"],
            input_text=data.get("input", ""),
            output=data["output"],
            metadata=data.get("metadata", {}),
        )

    def is_valid(self) -> bool:
        """Check if the example is valid (output is syntactically correct)."""
        return validate_code_syntax(self.output)


@dataclass
class AugmentedDataConfig:
    """Configuration for augmented data generation."""

    # Output paths
    output_dir: str = "data/physics_augmented"
    train_file: str = "train.jsonl"
    val_file: str = "val.jsonl"

    # Generation settings
    target_examples: int = 5000
    val_split: float = 0.1
    seed: int = 42

    # Type distribution (should sum to 1.0)
    type1_weight: float = 0.4  # Physics to Circuit
    type2_weight: float = 0.35  # Simulation to Insight
    type3_weight: float = 0.25  # Conservation to Constraint

    # Quality settings
    require_valid_syntax: bool = True
    min_output_length: int = 100
    max_output_length: int = 4000


# ============================================================================
# Symmetry and Conservation Templates
# ============================================================================

SYMMETRY_DESCRIPTIONS = {
    SymmetryType.TRANSLATION_X: {
        "name": "x-translation symmetry",
        "physics": "uniform in the x-direction",
        "implication": "momentum is conserved along x",
        "circuit_hint": "use identical operations across qubits in x",
    },
    SymmetryType.TRANSLATION_Y: {
        "name": "y-translation symmetry",
        "physics": "uniform in the y-direction",
        "implication": "momentum is conserved along y",
        "circuit_hint": "use identical operations across qubits in y",
    },
    SymmetryType.ROTATION: {
        "name": "rotational symmetry",
        "physics": "invariant under rotations",
        "implication": "angular momentum is conserved",
        "circuit_hint": "use symmetric entanglement patterns",
    },
    SymmetryType.REFLECTION_X: {
        "name": "x-reflection symmetry",
        "physics": "invariant under x-mirror",
        "implication": "parity is conserved",
        "circuit_hint": "use symmetric qubit arrangements",
    },
    SymmetryType.REFLECTION_Y: {
        "name": "y-reflection symmetry",
        "physics": "invariant under y-mirror",
        "implication": "parity is conserved",
        "circuit_hint": "use symmetric qubit arrangements",
    },
    SymmetryType.TIME_TRANSLATION: {
        "name": "time-translation symmetry",
        "physics": "stationary or periodic in time",
        "implication": "energy is conserved",
        "circuit_hint": "focus on energy-preserving operations",
    },
}

DYNAMICS_DESCRIPTIONS = {
    DynamicsType.STEADY_STATE: {
        "name": "steady state",
        "behavior": "constant in time after initial transient",
        "circuit_strategy": "shallow circuits with fixed point convergence",
    },
    DynamicsType.PERIODIC: {
        "name": "periodic oscillation",
        "behavior": "repeating pattern with fixed period",
        "circuit_strategy": "layered circuits matching periodicity",
    },
    DynamicsType.QUASI_PERIODIC: {
        "name": "quasi-periodic oscillation",
        "behavior": "multiple frequencies without exact repetition",
        "circuit_strategy": "multi-scale entanglement patterns",
    },
    DynamicsType.CHAOTIC: {
        "name": "chaotic dynamics",
        "behavior": "sensitive to initial conditions, mixing",
        "circuit_strategy": "deep circuits with strong entanglement",
    },
    DynamicsType.TRANSIENT: {
        "name": "transient dynamics",
        "behavior": "evolving toward a different state",
        "circuit_strategy": "time-dependent parameter scheduling",
    },
    DynamicsType.UNKNOWN: {
        "name": "unknown dynamics",
        "behavior": "complex or unclassified",
        "circuit_strategy": "general variational ansatz",
    },
}

CONSERVATION_LAWS = [
    {
        "name": "energy",
        "symbol": "E",
        "operator": "Hamiltonian",
        "constraint": "commutes with H",
    },
    {
        "name": "total magnetization",
        "symbol": "M_z",
        "operator": "sum of Z operators",
        "constraint": "preserves total spin",
    },
    {
        "name": "particle number",
        "symbol": "N",
        "operator": "number operator",
        "constraint": "preserves occupation",
    },
    {
        "name": "parity",
        "symbol": "P",
        "operator": "parity operator",
        "constraint": "preserves even/odd sectors",
    },
    {
        "name": "momentum",
        "symbol": "p",
        "operator": "translation generator",
        "constraint": "translation invariant",
    },
]

PROBLEM_TYPES = [
    {
        "name": "XY chain",
        "description": "nearest-neighbor XX+YY interactions on 1D chain",
        "symmetries": ["U(1)", "translation"],
        "typical_depth": "2-4 layers",
    },
    {
        "name": "Heisenberg model",
        "description": "isotropic spin-spin interactions",
        "symmetries": ["SU(2)", "translation"],
        "typical_depth": "3-6 layers",
    },
    {
        "name": "TFIM",
        "description": "transverse field Ising model with ZZ and X terms",
        "symmetries": ["Z2 parity"],
        "typical_depth": "2-4 layers",
    },
    {
        "name": "Hubbard model",
        "description": "fermions on lattice with on-site interaction",
        "symmetries": ["U(1) charge", "SU(2) spin"],
        "typical_depth": "4-8 layers",
    },
    {
        "name": "Molecular Hamiltonian",
        "description": "electronic structure of molecules",
        "symmetries": ["particle number", "spin"],
        "typical_depth": "varies with active space",
    },
]


# ============================================================================
# Code Templates
# ============================================================================

def get_hardware_efficient_template(n_qubits: int, n_layers: int) -> str:
    """Generate hardware-efficient ansatz code template."""
    return f'''from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_hardware_efficient_ansatz(n_qubits: int = {n_qubits}, n_layers: int = {n_layers}) -> QuantumCircuit:
    """
    Hardware-efficient ansatz with RY-RZ rotations and linear CNOT connectivity.

    This ansatz is suitable for NISQ devices with limited connectivity.
    It uses alternating layers of single-qubit rotations and entangling gates.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers

    Returns:
        Parameterized QuantumCircuit
    """
    qc = QuantumCircuit(n_qubits)

    param_idx = 0
    for layer in range(n_layers):
        # Single-qubit rotations
        for i in range(n_qubits):
            theta = Parameter(f"theta_{{layer}}_{{i}}_ry")
            phi = Parameter(f"phi_{{layer}}_{{i}}_rz")
            qc.ry(theta, i)
            qc.rz(phi, i)
            param_idx += 2

        # Entangling layer (linear connectivity)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

    return qc
'''


def get_symmetry_preserving_template(n_qubits: int, symmetry: str) -> str:
    """Generate symmetry-preserving ansatz code template."""
    return f'''from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np

def create_{symmetry.lower()}_preserving_ansatz(n_qubits: int = {n_qubits}) -> QuantumCircuit:
    """
    Ansatz that preserves {symmetry} symmetry.

    This circuit is designed to respect the {symmetry} conservation law,
    ensuring that the variational optimization stays within the correct
    symmetry sector.

    Args:
        n_qubits: Number of qubits

    Returns:
        Parameterized QuantumCircuit respecting {symmetry} symmetry
    """
    qc = QuantumCircuit(n_qubits)

    # Initialize in symmetric subspace
    for i in range(n_qubits // 2):
        qc.x(i)

    param_idx = 0
    n_layers = 2

    for layer in range(n_layers):
        # Symmetry-preserving two-qubit gates
        for i in range(n_qubits - 1):
            theta = Parameter(f"theta_{{layer}}_{{i}}")
            # XY-type interaction preserves total magnetization
            qc.rxx(theta, i, i + 1)
            qc.ryy(theta, i, i + 1)
            param_idx += 1

    return qc
'''


def get_problem_inspired_template(problem_type: str, n_qubits: int) -> str:
    """Generate problem-inspired ansatz code template."""
    return f'''from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_{problem_type.lower().replace(" ", "_")}_ansatz(n_qubits: int = {n_qubits}) -> QuantumCircuit:
    """
    Problem-inspired ansatz for {problem_type}.

    This ansatz is designed based on the structure of the {problem_type},
    using gate patterns that match the problem's interaction structure.

    Args:
        n_qubits: Number of qubits

    Returns:
        Parameterized QuantumCircuit for {problem_type}
    """
    qc = QuantumCircuit(n_qubits)

    n_layers = 3
    param_idx = 0

    for layer in range(n_layers):
        # Local rotations
        for i in range(n_qubits):
            theta = Parameter(f"rx_{{layer}}_{{i}}")
            phi = Parameter(f"rz_{{layer}}_{{i}}")
            qc.rx(theta, i)
            qc.rz(phi, i)
            param_idx += 2

        # Problem-specific entanglement pattern
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            qc.cx(i, i + 1)

    return qc
'''


# ============================================================================
# Code Validation
# ============================================================================

def validate_code_syntax(code: str) -> bool:
    """
    Check if Python code is syntactically valid.

    Args:
        code: Python code string

    Returns:
        True if code compiles successfully
    """
    # Extract code from markdown if present
    if "```python" in code:
        start = code.find("```python") + 9
        end = code.find("```", start)
        if end > start:
            code = code[start:end].strip()
    elif "```" in code:
        code = code.replace("```", "").strip()

    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_code_from_output(output: str) -> str:
    """
    Extract Python code from model output.

    Args:
        output: Full model output with reasoning and code

    Returns:
        Extracted code portion
    """
    if "```python" in output:
        start = output.find("```python") + 9
        end = output.find("```", start)
        if end > start:
            return output[start:end].strip()

    # Look for function definition
    if "def " in output:
        lines = output.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith("def ") or line.strip().startswith("from ") or line.strip().startswith("import "):
                in_code = True
            if in_code:
                code_lines.append(line)
        return "\n".join(code_lines)

    return output


# ============================================================================
# Example Generators
# ============================================================================

class Type1Generator:
    """
    Generate Type 1 examples: Physics Description to Circuit.

    These examples take a description of system symmetries and conservation
    laws and generate circuit designs that respect those properties.
    """

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def generate(self, physics_features: PhysicsFeatures | None = None) -> TrainingExample:
        """
        Generate a Type 1 training example.

        Args:
            physics_features: Optional physics features from encoder

        Returns:
            TrainingExample with physics-to-circuit mapping
        """
        # Select problem type and parameters
        problem = self.rng.choice(PROBLEM_TYPES)
        n_qubits = self.rng.choice([4, 6, 8])

        # Build symmetries list
        symmetries = problem["symmetries"].copy()
        if physics_features:
            for sym in physics_features.detected_symmetries:
                if sym in SYMMETRY_DESCRIPTIONS:
                    sym_info = SYMMETRY_DESCRIPTIONS[sym]
                    symmetries.append(sym_info["name"])

        # Select conservation law
        conservation = self.rng.choice(CONSERVATION_LAWS)

        # Build instruction
        instruction = self._build_instruction(problem, symmetries, conservation, n_qubits)

        # Build output with reasoning and code
        output = self._build_output(problem, symmetries, conservation, n_qubits)

        return TrainingExample(
            example_type=ExampleType.PHYSICS_TO_CIRCUIT,
            instruction=instruction,
            input_text="",
            output=output,
            metadata={
                "problem_type": problem["name"],
                "n_qubits": n_qubits,
                "symmetries": symmetries,
                "conservation": conservation["name"],
            },
        )

    def _build_instruction(
        self,
        problem: dict,
        symmetries: list[str],
        conservation: dict,
        n_qubits: int,
    ) -> str:
        """Build the instruction prompt."""
        sym_str = ", ".join(symmetries[:3]) if symmetries else "none specified"
        return f"""Design a variational quantum circuit (ansatz) for the following physics problem:

Problem: {problem["name"]}
Description: {problem["description"]}
Number of qubits: {n_qubits}
Symmetries to preserve: {sym_str}
Conservation law: {conservation["name"]} ({conservation["symbol"]})

Requirements:
1. The circuit should respect the specified symmetries
2. The circuit should preserve {conservation["name"]} if applicable
3. Use a reasonable circuit depth ({problem["typical_depth"]})
4. Make the circuit hardware-efficient with local connectivity

Explain your design choices, then provide the Qiskit implementation."""

    def _build_output(
        self,
        problem: dict,
        symmetries: list[str],
        conservation: dict,
        n_qubits: int,
    ) -> str:
        """Build the expected output with reasoning and code."""
        n_layers = self.rng.choice([2, 3, 4])

        reasoning = f"""## Design Analysis

The {problem["name"]} has the following key properties:
- {problem["description"]}
- Symmetries: {", ".join(symmetries[:3]) if symmetries else "general variational"}
- Conservation: {conservation["name"]} must be preserved

### Circuit Design Strategy

1. **Initialization**: Start in an appropriate symmetry sector
2. **Variational layers**: Use {n_layers} layers of parameterized gates
3. **Entanglement**: Match the problem's interaction structure
4. **Gate choice**: Use gates that respect {conservation["name"]} conservation

The circuit uses:
- Local RY/RZ rotations for single-qubit flexibility
- Entangling gates that preserve the conservation law
- Linear connectivity for hardware compatibility

### Implementation

```python
{get_problem_inspired_template(problem["name"], n_qubits)}
```
"""
        return reasoning


class Type2Generator:
    """
    Generate Type 2 examples: Simulation Data to Physics Insight.

    These examples take descriptions of simulation behavior (from The Well)
    and generate physics interpretations and circuit implications.
    """

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def generate(self, physics_features: PhysicsFeatures | None = None) -> TrainingExample:
        """
        Generate a Type 2 training example.

        Args:
            physics_features: Optional physics features from encoder

        Returns:
            TrainingExample with simulation-to-insight mapping
        """
        # Use provided features or generate synthetic ones
        if physics_features:
            dynamics = physics_features.dynamics_type
            symmetries = physics_features.detected_symmetries
            conserved = physics_features.conserved_quantities
            scales = physics_features.characteristic_scales
        else:
            dynamics = self.rng.choice(list(DynamicsType))
            symmetries = self.rng.sample(list(SymmetryType), k=self.rng.randint(0, 3))
            conserved = self.rng.sample(
                ["total_mass", "energy_proxy", "momentum_x", "momentum_y"],
                k=self.rng.randint(1, 3),
            )
            scales = None

        instruction = self._build_instruction(dynamics, symmetries, conserved, scales)
        output = self._build_output(dynamics, symmetries, conserved, scales)

        return TrainingExample(
            example_type=ExampleType.SIMULATION_TO_INSIGHT,
            instruction=instruction,
            input_text="",
            output=output,
            metadata={
                "dynamics_type": dynamics.value if isinstance(dynamics, DynamicsType) else str(dynamics),
                "symmetries": [s.value if isinstance(s, SymmetryType) else str(s) for s in symmetries],
                "conserved_quantities": conserved,
            },
        )

    def _build_instruction(
        self,
        dynamics: DynamicsType,
        symmetries: list[SymmetryType],
        conserved: list[str],
        scales: object | None,
    ) -> str:
        """Build the instruction prompt."""
        dyn_desc = DYNAMICS_DESCRIPTIONS.get(dynamics, DYNAMICS_DESCRIPTIONS[DynamicsType.UNKNOWN])

        sym_list = []
        for s in symmetries:
            if s in SYMMETRY_DESCRIPTIONS:
                sym_list.append(SYMMETRY_DESCRIPTIONS[s]["name"])

        return f"""A physics simulation shows the following behavior:

Dynamics: {dyn_desc["name"]}
- Observed behavior: {dyn_desc["behavior"]}

Detected symmetries: {", ".join(sym_list) if sym_list else "none detected"}

Conserved quantities: {", ".join(conserved) if conserved else "none verified"}

Based on this simulation data, analyze the physics and recommend a quantum circuit design approach. Consider:
1. What physical laws govern this system?
2. What symmetries should the quantum ansatz respect?
3. How should the circuit depth relate to the dynamics complexity?
4. What gate patterns would be most appropriate?"""

    def _build_output(
        self,
        dynamics: DynamicsType,
        symmetries: list[SymmetryType],
        conserved: list[str],
        scales: object | None,
    ) -> str:
        """Build the expected output."""
        dyn_desc = DYNAMICS_DESCRIPTIONS.get(dynamics, DYNAMICS_DESCRIPTIONS[DynamicsType.UNKNOWN])

        sym_insights = []
        for s in symmetries:
            if s in SYMMETRY_DESCRIPTIONS:
                info = SYMMETRY_DESCRIPTIONS[s]
                sym_insights.append(f"- {info['name']}: {info['physics']} => {info['implication']}")

        n_qubits = self.rng.choice([4, 6, 8])
        n_layers = {"steady_state": 2, "periodic": 3, "chaotic": 5}.get(dynamics.value, 3)

        return f"""## Physics Analysis

### Dynamics Interpretation

The simulation exhibits **{dyn_desc["name"]}**:
- {dyn_desc["behavior"]}
- This suggests the underlying Hamiltonian has specific spectral properties

### Symmetry Analysis

{"Detected symmetries and their implications:" if sym_insights else "No strong symmetries detected, suggesting a general ansatz."}
{chr(10).join(sym_insights) if sym_insights else ""}

### Conservation Laws

{"The following quantities appear conserved:" if conserved else "No strictly conserved quantities detected."}
{chr(10).join(f"- {c}" for c in conserved) if conserved else ""}

### Circuit Design Recommendations

Based on this analysis:

1. **Circuit depth**: {dyn_desc["circuit_strategy"]}
   - Recommended: {n_layers} layers for {dynamics.value} dynamics

2. **Gate selection**:
   - Use parameterized rotations (RY, RZ) for variational flexibility
   - Include entangling gates that match the interaction structure

3. **Symmetry preservation**:
   {f"- Design gates to preserve: {', '.join(conserved)}" if conserved else "- No specific symmetry constraints"}

### Recommended Ansatz

```python
{get_hardware_efficient_template(n_qubits, n_layers)}
```
"""


class Type3Generator:
    """
    Generate Type 3 examples: Conservation Law to Circuit Constraint.

    These examples take a conservation law specification and generate
    circuit constraints that preserve the symmetry.
    """

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def generate(self, physics_features: PhysicsFeatures | None = None) -> TrainingExample:
        """
        Generate a Type 3 training example.

        Args:
            physics_features: Optional physics features from encoder

        Returns:
            TrainingExample with conservation-to-constraint mapping
        """
        conservation = self.rng.choice(CONSERVATION_LAWS)
        n_qubits = self.rng.choice([4, 6, 8])

        instruction = self._build_instruction(conservation, n_qubits)
        output = self._build_output(conservation, n_qubits)

        return TrainingExample(
            example_type=ExampleType.CONSERVATION_TO_CONSTRAINT,
            instruction=instruction,
            input_text="",
            output=output,
            metadata={
                "conservation_law": conservation["name"],
                "n_qubits": n_qubits,
            },
        )

    def _build_instruction(self, conservation: dict, n_qubits: int) -> str:
        """Build the instruction prompt."""
        return f"""Design a quantum ansatz that strictly preserves {conservation["name"]} ({conservation["symbol"]}).

System specifications:
- Number of qubits: {n_qubits}
- Conservation law: {conservation["name"]}
- Associated operator: {conservation["operator"]}
- Constraint: Circuit must {conservation["constraint"]}

Your task:
1. Explain why preserving this conservation law is important
2. Describe which gates preserve vs. violate this symmetry
3. Design an ansatz that stays in a fixed symmetry sector
4. Provide the complete Qiskit implementation"""

    def _build_output(self, conservation: dict, n_qubits: int) -> str:
        """Build the expected output."""
        return f"""## Conservation Law Analysis: {conservation["name"]}

### Physical Motivation

The {conservation["name"]} conservation law ({conservation["symbol"]}) is associated with the {conservation["operator"]}. Preserving this symmetry in our ansatz is crucial because:

1. **Physical relevance**: The true ground state lives in a specific symmetry sector
2. **Reduced search space**: Staying in one sector reduces variational complexity
3. **Numerical stability**: Symmetry-preserving circuits avoid spurious states

### Gate Analysis

**Gates that PRESERVE {conservation["name"]}:**
- RZZ: Commutes with total Z operators
- XX + YY (XY interaction): Preserves total magnetization
- SWAP: Preserves particle number
- Controlled rotations within symmetry sector

**Gates that VIOLATE {conservation["name"]}:**
- Single X or Y gates (flip individual spins)
- Arbitrary single-qubit rotations
- Gates that don't commute with the symmetry operator

### Symmetry-Preserving Ansatz Design

To ensure our circuit {conservation["constraint"]}:

1. **Initialize correctly**: Start in the target symmetry sector
2. **Use preserving gates**: Only apply gates that commute with {conservation["operator"]}
3. **Verify invariance**: Check that [G, {conservation["symbol"]}] = 0 for all gates G

### Implementation

```python
{get_symmetry_preserving_template(n_qubits, conservation["name"].replace(" ", "_"))}
```

This ansatz:
- Initializes in the half-filled sector (equal up/down spins)
- Uses XY-type interactions that preserve total magnetization
- Maintains the system in the correct symmetry sector throughout
"""


# ============================================================================
# Main Generator Class
# ============================================================================

class AugmentedDataGenerator:
    """
    Main generator for physics-augmented training data.

    Combines all three example types with configurable weights
    and saves to JSONL format.
    """

    def __init__(self, config: AugmentedDataConfig | None = None):
        """
        Initialize the generator.

        Args:
            config: Generation configuration
        """
        self.config = config or AugmentedDataConfig()
        self.rng = random.Random(self.config.seed)

        # Initialize type generators
        self.type1_gen = Type1Generator(self.rng)
        self.type2_gen = Type2Generator(self.rng)
        self.type3_gen = Type3Generator(self.rng)

        # Physics encoder for feature extraction
        self.encoder = PhysicsEncoder()

    def generate_example(
        self,
        example_type: ExampleType | None = None,
        physics_features: PhysicsFeatures | None = None,
    ) -> TrainingExample:
        """
        Generate a single training example.

        Args:
            example_type: Specific type to generate (random if None)
            physics_features: Optional physics features from encoder

        Returns:
            Generated TrainingExample
        """
        if example_type is None:
            # Random selection based on weights
            r = self.rng.random()
            if r < self.config.type1_weight:
                example_type = ExampleType.PHYSICS_TO_CIRCUIT
            elif r < self.config.type1_weight + self.config.type2_weight:
                example_type = ExampleType.SIMULATION_TO_INSIGHT
            else:
                example_type = ExampleType.CONSERVATION_TO_CONSTRAINT

        if example_type == ExampleType.PHYSICS_TO_CIRCUIT:
            return self.type1_gen.generate(physics_features)
        elif example_type == ExampleType.SIMULATION_TO_INSIGHT:
            return self.type2_gen.generate(physics_features)
        else:
            return self.type3_gen.generate(physics_features)

    def generate_batch(
        self,
        count: int,
        physics_features_list: list[PhysicsFeatures] | None = None,
    ) -> list[TrainingExample]:
        """
        Generate a batch of training examples.

        Args:
            count: Number of examples to generate
            physics_features_list: Optional list of physics features

        Returns:
            List of TrainingExample objects
        """
        examples = []

        for i in range(count):
            features = None
            if physics_features_list and i < len(physics_features_list):
                features = physics_features_list[i]

            example = self.generate_example(physics_features=features)

            # Quality filter
            if self.config.require_valid_syntax and not example.is_valid():
                # Retry once
                example = self.generate_example(physics_features=features)

            if len(example.output) >= self.config.min_output_length:
                examples.append(example)

        return examples

    def generate_from_trajectories(
        self,
        trajectories: np.ndarray | list[np.ndarray],
        dt: float = 1.0,
        dx: float = 1.0,
    ) -> list[TrainingExample]:
        """
        Generate examples from simulation trajectories.

        Args:
            trajectories: Array or list of trajectory arrays
            dt: Time step
            dx: Spatial step

        Returns:
            List of TrainingExample objects
        """
        if isinstance(trajectories, np.ndarray) and trajectories.ndim == 4:
            # Single trajectory
            trajectories = [trajectories]

        examples = []
        for traj in trajectories:
            features = self.encoder.encode(traj, dt, dx)
            example = self.generate_example(physics_features=features)
            examples.append(example)

        return examples

    def generate_dataset(
        self,
        target_count: int | None = None,
    ) -> tuple[list[TrainingExample], list[TrainingExample]]:
        """
        Generate full train/val dataset.

        Args:
            target_count: Number of examples (uses config default if None)

        Returns:
            Tuple of (train_examples, val_examples)
        """
        target = target_count or self.config.target_examples

        # Generate all examples
        examples = self.generate_batch(target)

        # Shuffle
        self.rng.shuffle(examples)

        # Split
        n_val = int(len(examples) * self.config.val_split)
        val_examples = examples[:n_val]
        train_examples = examples[n_val:]

        return train_examples, val_examples

    def save_dataset(
        self,
        train_examples: list[TrainingExample],
        val_examples: list[TrainingExample],
    ) -> dict:
        """
        Save dataset to JSONL files.

        Args:
            train_examples: Training examples
            val_examples: Validation examples

        Returns:
            Statistics dictionary
        """
        os.makedirs(self.config.output_dir, exist_ok=True)

        train_path = os.path.join(self.config.output_dir, self.config.train_file)
        val_path = os.path.join(self.config.output_dir, self.config.val_file)

        # Write train
        with open(train_path, "w") as f:
            for ex in train_examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

        # Write val
        with open(val_path, "w") as f:
            for ex in val_examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

        # Compute statistics
        type_counts = {}
        for ex in train_examples + val_examples:
            t = ex.example_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        stats = {
            "train_count": len(train_examples),
            "val_count": len(val_examples),
            "total_count": len(train_examples) + len(val_examples),
            "type_distribution": type_counts,
            "train_path": train_path,
            "val_path": val_path,
        }

        # Save stats
        stats_path = os.path.join(self.config.output_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        return stats

    def generate_and_save(
        self,
        target_count: int | None = None,
    ) -> dict:
        """
        Generate and save complete dataset.

        Args:
            target_count: Number of examples (uses config default if None)

        Returns:
            Statistics dictionary
        """
        train_examples, val_examples = self.generate_dataset(target_count)
        return self.save_dataset(train_examples, val_examples)


def load_augmented_dataset(
    split: str = "train",
    data_dir: str = "data/physics_augmented",
) -> list[TrainingExample]:
    """
    Load generated augmented dataset.

    Args:
        split: "train" or "val"
        data_dir: Directory containing data files

    Returns:
        List of TrainingExample objects
    """
    filename = "train.jsonl" if split == "train" else "val.jsonl"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    examples = []
    with open(filepath) as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append(TrainingExample.from_dict(data))

    return examples


def get_dataset_stats(data_dir: str = "data/physics_augmented") -> dict:
    """
    Get statistics for generated dataset.

    Args:
        data_dir: Directory containing data files

    Returns:
        Statistics dictionary
    """
    stats_path = os.path.join(data_dir, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)

    # Compute from files if stats not saved
    train = load_augmented_dataset("train", data_dir)
    val = load_augmented_dataset("val", data_dir)

    type_counts = {}
    for ex in train + val:
        t = ex.example_type.value
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "train_count": len(train),
        "val_count": len(val),
        "total_count": len(train) + len(val),
        "type_distribution": type_counts,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate physics-augmented training data")
    parser.add_argument("--count", type=int, default=100, help="Number of examples")
    parser.add_argument("--output-dir", type=str, default="data/physics_augmented")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = AugmentedDataConfig(
        output_dir=args.output_dir,
        target_examples=args.count,
        seed=args.seed,
    )

    generator = AugmentedDataGenerator(config)
    stats = generator.generate_and_save()

    print(f"Generated {stats['total_count']} examples")
    print(f"Train: {stats['train_count']}, Val: {stats['val_count']}")
    print(f"Type distribution: {stats['type_distribution']}")
    print(f"Saved to: {stats['train_path']}")
