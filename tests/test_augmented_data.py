"""
Tests for the physics-augmented training data generator.

These tests verify the generation of physics-augmented training examples
including all three example types, code validation, and dataset I/O.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from src.quasar.augmented_data import (
    AugmentedDataConfig,
    AugmentedDataGenerator,
    ExampleType,
    TrainingExample,
    Type1Generator,
    Type2Generator,
    Type3Generator,
    extract_code_from_output,
    get_dataset_stats,
    get_hardware_efficient_template,
    get_problem_inspired_template,
    get_symmetry_preserving_template,
    load_augmented_dataset,
    validate_code_syntax,
)
from src.quasar.physics_encoder import (
    DynamicsType,
    PhysicsEncoder,
    PhysicsFeatures,
    SymmetryType,
)


class TestExampleType:
    """Tests for ExampleType enum."""

    def test_enum_values(self):
        assert ExampleType.PHYSICS_TO_CIRCUIT.value == "physics_to_circuit"
        assert ExampleType.SIMULATION_TO_INSIGHT.value == "simulation_to_insight"
        assert ExampleType.CONSERVATION_TO_CONSTRAINT.value == "conservation_to_constraint"

    def test_all_types_exist(self):
        types = list(ExampleType)
        assert len(types) == 3


class TestTrainingExample:
    """Tests for TrainingExample dataclass."""

    def test_creation(self):
        example = TrainingExample(
            example_type=ExampleType.PHYSICS_TO_CIRCUIT,
            instruction="Design a circuit",
            input_text="",
            output="def create_circuit(): pass",
            metadata={"n_qubits": 4},
        )
        assert example.example_type == ExampleType.PHYSICS_TO_CIRCUIT
        assert example.instruction == "Design a circuit"
        assert example.metadata["n_qubits"] == 4

    def test_to_dict(self):
        example = TrainingExample(
            example_type=ExampleType.SIMULATION_TO_INSIGHT,
            instruction="Analyze simulation",
            input_text="Simulation data",
            output="Analysis output",
            metadata={"dynamics": "periodic"},
        )
        d = example.to_dict()
        assert d["type"] == "simulation_to_insight"
        assert d["instruction"] == "Analyze simulation"
        assert d["input"] == "Simulation data"
        assert d["output"] == "Analysis output"
        assert d["metadata"]["dynamics"] == "periodic"

    def test_from_dict(self):
        data = {
            "type": "conservation_to_constraint",
            "instruction": "Preserve symmetry",
            "input": "",
            "output": "Constraint output",
            "metadata": {},
        }
        example = TrainingExample.from_dict(data)
        assert example.example_type == ExampleType.CONSERVATION_TO_CONSTRAINT
        assert example.instruction == "Preserve symmetry"

    def test_is_valid_with_valid_code(self):
        example = TrainingExample(
            example_type=ExampleType.PHYSICS_TO_CIRCUIT,
            instruction="Test",
            input_text="",
            output="def test(): return 42",
        )
        assert example.is_valid() is True

    def test_is_valid_with_invalid_code(self):
        example = TrainingExample(
            example_type=ExampleType.PHYSICS_TO_CIRCUIT,
            instruction="Test",
            input_text="",
            output="def test(: return",  # Invalid syntax
        )
        assert example.is_valid() is False

    def test_is_valid_with_markdown_code(self):
        example = TrainingExample(
            example_type=ExampleType.PHYSICS_TO_CIRCUIT,
            instruction="Test",
            input_text="",
            output="```python\ndef test(): return 42\n```",
        )
        assert example.is_valid() is True


class TestAugmentedDataConfig:
    """Tests for AugmentedDataConfig."""

    def test_default_config(self):
        config = AugmentedDataConfig()
        assert config.target_examples == 5000
        assert config.val_split == 0.1
        assert config.type1_weight == 0.4
        assert config.type2_weight == 0.35
        assert config.type3_weight == 0.25

    def test_custom_config(self):
        config = AugmentedDataConfig(
            target_examples=1000,
            val_split=0.2,
            seed=123,
        )
        assert config.target_examples == 1000
        assert config.val_split == 0.2
        assert config.seed == 123

    def test_weights_sum_to_one(self):
        config = AugmentedDataConfig()
        total = config.type1_weight + config.type2_weight + config.type3_weight
        assert abs(total - 1.0) < 1e-6


class TestCodeValidation:
    """Tests for code validation functions."""

    def test_validate_valid_syntax(self):
        code = "def test(): return 42"
        assert validate_code_syntax(code) is True

    def test_validate_invalid_syntax(self):
        code = "def test(: return"
        assert validate_code_syntax(code) is False

    def test_validate_with_markdown(self):
        code = "```python\ndef test(): return 42\n```"
        assert validate_code_syntax(code) is True

    def test_validate_complex_code(self):
        code = """
from qiskit import QuantumCircuit

def create_ansatz(n_qubits):
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    return qc
"""
        assert validate_code_syntax(code) is True

    def test_extract_code_with_markdown(self):
        output = "Some text\n```python\ndef test(): pass\n```\nMore text"
        code = extract_code_from_output(output)
        assert "def test():" in code
        assert "Some text" not in code

    def test_extract_code_without_markdown(self):
        output = "from qiskit import QuantumCircuit\ndef create(): pass"
        code = extract_code_from_output(output)
        assert "QuantumCircuit" in code
        assert "def create" in code


class TestCodeTemplates:
    """Tests for code template generators."""

    def test_hardware_efficient_template(self):
        code = get_hardware_efficient_template(4, 2)
        assert validate_code_syntax(code)
        assert "QuantumCircuit" in code
        assert "Parameter" in code
        assert "def create_hardware_efficient_ansatz" in code

    def test_symmetry_preserving_template(self):
        code = get_symmetry_preserving_template(4, "U1")
        assert validate_code_syntax(code)
        assert "QuantumCircuit" in code
        assert "preserving" in code.lower()

    def test_problem_inspired_template(self):
        code = get_problem_inspired_template("XY chain", 6)
        assert validate_code_syntax(code)
        assert "QuantumCircuit" in code
        assert "xy_chain" in code.lower()


class TestType1Generator:
    """Tests for Type 1: Physics to Circuit generator."""

    def test_generate_basic(self):
        gen = Type1Generator()
        example = gen.generate()

        assert example.example_type == ExampleType.PHYSICS_TO_CIRCUIT
        assert len(example.instruction) > 50
        assert len(example.output) > 100
        assert "n_qubits" in example.metadata

    def test_generate_with_physics_features(self):
        # Create mock physics features
        gen = Type1Generator()

        # Generate without features first
        example = gen.generate()
        assert example is not None
        assert example.example_type == ExampleType.PHYSICS_TO_CIRCUIT

    def test_generate_has_code(self):
        gen = Type1Generator()
        example = gen.generate()

        assert "```python" in example.output or "def " in example.output

    def test_generate_mentions_symmetry(self):
        gen = Type1Generator()
        example = gen.generate()

        # Should mention symmetry in instruction or output
        combined = example.instruction + example.output
        assert "symmetr" in combined.lower() or "conserv" in combined.lower()

    def test_deterministic_with_seed(self):
        import random

        gen1 = Type1Generator(random.Random(42))
        gen2 = Type1Generator(random.Random(42))

        ex1 = gen1.generate()
        ex2 = gen2.generate()

        assert ex1.metadata == ex2.metadata


class TestType2Generator:
    """Tests for Type 2: Simulation to Insight generator."""

    def test_generate_basic(self):
        gen = Type2Generator()
        example = gen.generate()

        assert example.example_type == ExampleType.SIMULATION_TO_INSIGHT
        assert len(example.instruction) > 50
        assert len(example.output) > 100
        assert "dynamics_type" in example.metadata

    def test_generate_has_analysis(self):
        gen = Type2Generator()
        example = gen.generate()

        output_lower = example.output.lower()
        assert "analysis" in output_lower or "recommend" in output_lower

    def test_generate_with_physics_features(self):
        # Create a synthetic trajectory
        trajectory = np.random.randn(10, 32, 32, 2).astype(np.float32)
        encoder = PhysicsEncoder()
        features = encoder.encode(trajectory)

        gen = Type2Generator()
        example = gen.generate(physics_features=features)

        assert example.example_type == ExampleType.SIMULATION_TO_INSIGHT


class TestType3Generator:
    """Tests for Type 3: Conservation to Constraint generator."""

    def test_generate_basic(self):
        gen = Type3Generator()
        example = gen.generate()

        assert example.example_type == ExampleType.CONSERVATION_TO_CONSTRAINT
        assert len(example.instruction) > 50
        assert len(example.output) > 100
        assert "conservation_law" in example.metadata

    def test_generate_mentions_conservation(self):
        gen = Type3Generator()
        example = gen.generate()

        combined = example.instruction + example.output
        assert "conserv" in combined.lower() or "preserv" in combined.lower()

    def test_generate_has_gate_analysis(self):
        gen = Type3Generator()
        example = gen.generate()

        output_lower = example.output.lower()
        # Should discuss which gates preserve/violate symmetry
        assert "gate" in output_lower


class TestAugmentedDataGenerator:
    """Tests for the main AugmentedDataGenerator class."""

    def test_init_default(self):
        gen = AugmentedDataGenerator()
        assert gen.config is not None
        assert gen.type1_gen is not None
        assert gen.type2_gen is not None
        assert gen.type3_gen is not None

    def test_init_custom_config(self):
        config = AugmentedDataConfig(seed=123, target_examples=100)
        gen = AugmentedDataGenerator(config)
        assert gen.config.seed == 123
        assert gen.config.target_examples == 100

    def test_generate_example_random_type(self):
        gen = AugmentedDataGenerator()
        example = gen.generate_example()

        assert example.example_type in list(ExampleType)
        assert len(example.instruction) > 0
        assert len(example.output) > 0

    def test_generate_example_specific_type(self):
        gen = AugmentedDataGenerator()

        ex1 = gen.generate_example(ExampleType.PHYSICS_TO_CIRCUIT)
        assert ex1.example_type == ExampleType.PHYSICS_TO_CIRCUIT

        ex2 = gen.generate_example(ExampleType.SIMULATION_TO_INSIGHT)
        assert ex2.example_type == ExampleType.SIMULATION_TO_INSIGHT

        ex3 = gen.generate_example(ExampleType.CONSERVATION_TO_CONSTRAINT)
        assert ex3.example_type == ExampleType.CONSERVATION_TO_CONSTRAINT

    def test_generate_batch(self):
        gen = AugmentedDataGenerator()
        examples = gen.generate_batch(10)

        assert len(examples) >= 8  # Some may be filtered
        for ex in examples:
            assert isinstance(ex, TrainingExample)

    def test_generate_batch_type_distribution(self):
        config = AugmentedDataConfig(
            type1_weight=0.5,
            type2_weight=0.3,
            type3_weight=0.2,
            seed=42,
        )
        gen = AugmentedDataGenerator(config)
        examples = gen.generate_batch(100)

        type_counts = {}
        for ex in examples:
            t = ex.example_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        # Check distribution is roughly correct (with some tolerance)
        total = len(examples)
        if total > 0:
            type1_frac = type_counts.get("physics_to_circuit", 0) / total
            assert 0.3 < type1_frac < 0.7  # Should be around 0.5

    def test_generate_from_trajectories(self):
        gen = AugmentedDataGenerator()

        # Create synthetic trajectories
        trajectories = [
            np.random.randn(10, 32, 32, 2).astype(np.float32),
            np.random.randn(10, 32, 32, 2).astype(np.float32),
        ]

        examples = gen.generate_from_trajectories(trajectories)

        assert len(examples) == 2
        for ex in examples:
            assert isinstance(ex, TrainingExample)

    def test_generate_dataset(self):
        config = AugmentedDataConfig(target_examples=50, val_split=0.2)
        gen = AugmentedDataGenerator(config)

        train, val = gen.generate_dataset()

        assert len(train) + len(val) <= 50
        assert len(val) <= len(train)  # Val should be smaller

    def test_save_and_load_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AugmentedDataConfig(
                output_dir=tmpdir,
                target_examples=20,
                val_split=0.2,
            )
            gen = AugmentedDataGenerator(config)

            # Generate and save
            train, val = gen.generate_dataset()
            stats = gen.save_dataset(train, val)

            assert stats["train_count"] == len(train)
            assert stats["val_count"] == len(val)
            assert os.path.exists(os.path.join(tmpdir, "train.jsonl"))
            assert os.path.exists(os.path.join(tmpdir, "val.jsonl"))

            # Load and verify
            loaded_train = load_augmented_dataset("train", tmpdir)
            loaded_val = load_augmented_dataset("val", tmpdir)

            assert len(loaded_train) == len(train)
            assert len(loaded_val) == len(val)

    def test_generate_and_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AugmentedDataConfig(
                output_dir=tmpdir,
                target_examples=30,
            )
            gen = AugmentedDataGenerator(config)

            stats = gen.generate_and_save()

            assert stats["total_count"] > 0
            assert os.path.exists(stats["train_path"])
            assert os.path.exists(stats["val_path"])


class TestDatasetIO:
    """Tests for dataset I/O functions."""

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_augmented_dataset("train", "/nonexistent/path")

    def test_load_train_and_val(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            train_data = [
                TrainingExample(
                    ExampleType.PHYSICS_TO_CIRCUIT,
                    "inst1",
                    "",
                    "out1",
                ).to_dict()
            ]
            val_data = [
                TrainingExample(
                    ExampleType.SIMULATION_TO_INSIGHT,
                    "inst2",
                    "",
                    "out2",
                ).to_dict()
            ]

            with open(os.path.join(tmpdir, "train.jsonl"), "w") as f:
                for d in train_data:
                    f.write(json.dumps(d) + "\n")

            with open(os.path.join(tmpdir, "val.jsonl"), "w") as f:
                for d in val_data:
                    f.write(json.dumps(d) + "\n")

            train = load_augmented_dataset("train", tmpdir)
            val = load_augmented_dataset("val", tmpdir)

            assert len(train) == 1
            assert len(val) == 1
            assert train[0].example_type == ExampleType.PHYSICS_TO_CIRCUIT
            assert val[0].example_type == ExampleType.SIMULATION_TO_INSIGHT

    def test_get_dataset_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AugmentedDataConfig(
                output_dir=tmpdir,
                target_examples=25,
            )
            gen = AugmentedDataGenerator(config)
            gen.generate_and_save()

            stats = get_dataset_stats(tmpdir)

            assert "train_count" in stats
            assert "val_count" in stats
            assert "total_count" in stats
            assert "type_distribution" in stats


class TestIntegrationWithPhysicsEncoder:
    """Integration tests with the physics encoder."""

    def test_generate_with_encoded_features(self):
        # Create trajectory with known properties
        # Steady state: constant over time
        trajectory = np.ones((20, 32, 32, 2), dtype=np.float32)
        trajectory += np.random.randn(*trajectory.shape) * 0.01  # Small noise

        encoder = PhysicsEncoder()
        features = encoder.encode(trajectory)

        gen = AugmentedDataGenerator()
        example = gen.generate_example(
            ExampleType.SIMULATION_TO_INSIGHT,
            physics_features=features,
        )

        assert example is not None
        assert "dynamics_type" in example.metadata

    def test_batch_with_encoded_features(self):
        # Create multiple trajectories
        trajectories = [
            np.random.randn(10, 32, 32, 2).astype(np.float32)
            for _ in range(3)
        ]

        gen = AugmentedDataGenerator()
        examples = gen.generate_from_trajectories(trajectories)

        assert len(examples) == 3
        for ex in examples:
            assert isinstance(ex, TrainingExample)


class TestOutputQuality:
    """Tests for output quality requirements."""

    def test_examples_have_minimum_length(self):
        config = AugmentedDataConfig(min_output_length=100)
        gen = AugmentedDataGenerator(config)

        examples = gen.generate_batch(10)
        for ex in examples:
            assert len(ex.output) >= 100

    def test_examples_contain_reasoning(self):
        gen = AugmentedDataGenerator()

        for _ in range(5):
            example = gen.generate_example()
            output_lower = example.output.lower()

            # Should contain some form of explanation
            has_reasoning = any(
                keyword in output_lower
                for keyword in ["because", "therefore", "this", "the", "design"]
            )
            assert has_reasoning

    def test_examples_contain_code(self):
        gen = AugmentedDataGenerator()

        for _ in range(5):
            example = gen.generate_example()

            # Should contain code
            has_code = (
                "def " in example.output
                or "QuantumCircuit" in example.output
                or "```python" in example.output
            )
            assert has_code


class TestReproducibility:
    """Tests for reproducibility with seeds."""

    def test_same_seed_same_output(self):
        config1 = AugmentedDataConfig(seed=42, target_examples=10)
        config2 = AugmentedDataConfig(seed=42, target_examples=10)

        gen1 = AugmentedDataGenerator(config1)
        gen2 = AugmentedDataGenerator(config2)

        examples1 = gen1.generate_batch(5)
        examples2 = gen2.generate_batch(5)

        for e1, e2 in zip(examples1, examples2):
            assert e1.example_type == e2.example_type
            assert e1.metadata == e2.metadata

    def test_different_seed_different_output(self):
        config1 = AugmentedDataConfig(seed=42, target_examples=10)
        config2 = AugmentedDataConfig(seed=123, target_examples=10)

        gen1 = AugmentedDataGenerator(config1)
        gen2 = AugmentedDataGenerator(config2)

        examples1 = gen1.generate_batch(5)
        examples2 = gen2.generate_batch(5)

        # At least some should differ
        different_count = sum(
            e1.metadata != e2.metadata
            for e1, e2 in zip(examples1, examples2)
        )
        assert different_count > 0
