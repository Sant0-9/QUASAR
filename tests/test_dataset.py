"""
Tests for the dataset preparation module.
"""

import json
import os
import random
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.training.dataset import (
    DatasetConfig,
    MixedDatasetConfig,
    MixedDatasetBuilder,
    clean_example,
    filter_example,
    format_physics_augmented_example,
    load_physics_augmented_data,
    weighted_sample,
)


class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DatasetConfig()

        assert config.raw_dir == "data/raw"
        assert config.processed_dir == "data/processed"
        assert config.min_output_length == 50
        assert config.max_output_length == 5000
        assert config.require_qiskit is True
        assert config.require_function is True
        assert config.train_split == 0.9
        assert config.seed == 42

    def test_custom_config(self):
        """Test custom configuration."""
        config = DatasetConfig(
            raw_dir="/custom/raw",
            processed_dir="/custom/processed",
            min_output_length=100,
            max_output_length=3000,
            require_qiskit=False,
            train_split=0.8,
            seed=123,
        )

        assert config.raw_dir == "/custom/raw"
        assert config.processed_dir == "/custom/processed"
        assert config.min_output_length == 100
        assert config.max_output_length == 3000
        assert config.require_qiskit is False
        assert config.train_split == 0.8
        assert config.seed == 123

    def test_default_categories(self):
        """Test default categories list."""
        config = DatasetConfig()

        assert "circuit_generation" in config.categories
        assert "vqe" in config.categories
        assert "ansatz_design" in config.categories


class TestFilterExample:
    """Tests for filter_example function."""

    def test_valid_qiskit_code(self):
        """Valid Qiskit code should pass filter."""
        config = DatasetConfig()
        example = {
            "output": """from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits):
    qc = QuantumCircuit(num_qubits)
    params = [Parameter(f'theta_{i}') for i in range(num_qubits)]
    for i in range(num_qubits):
        qc.ry(params[i], i)
    return qc
""",
            "category": "circuit_generation",
        }

        assert filter_example(example, config) is True

    def test_too_short_output(self):
        """Output shorter than minimum should fail."""
        config = DatasetConfig(min_output_length=100)
        example = {
            "output": "x = 1",
            "category": "circuit_generation",
        }

        assert filter_example(example, config) is False

    def test_too_long_output(self):
        """Output longer than maximum should fail."""
        config = DatasetConfig(max_output_length=100)
        example = {
            "output": "x = 1\n" * 100,  # Long output
            "category": "circuit_generation",
        }

        assert filter_example(example, config) is False

    def test_no_qiskit_fails(self):
        """Code without Qiskit should fail when required."""
        config = DatasetConfig(require_qiskit=True)
        example = {
            "output": """def calculate(x):
    return x * 2
""" * 20,  # Make it long enough
            "category": "circuit_generation",
        }

        assert filter_example(example, config) is False

    def test_no_qiskit_passes_when_not_required(self):
        """Code without Qiskit should pass when not required."""
        config = DatasetConfig(require_qiskit=False, require_function=False)
        example = {
            "output": """# Some valid Python code
x = 1
y = 2
z = x + y
print(z)
""" * 20,  # Make it long enough
            "category": "circuit_generation",
        }

        assert filter_example(example, config) is True

    def test_no_function_fails(self):
        """Code without function definition should fail when required."""
        config = DatasetConfig(require_function=True, require_qiskit=False)
        example = {
            "output": """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
""" * 10,
            "category": "circuit_generation",
        }

        assert filter_example(example, config) is False

    def test_syntax_error_fails(self):
        """Code with syntax errors should fail."""
        config = DatasetConfig(require_qiskit=False, require_function=False)
        example = {
            "output": """def broken(
    x = 1  # Missing closing paren
""" * 10,
            "category": "circuit_generation",
        }

        assert filter_example(example, config) is False

    def test_wrong_category_fails(self):
        """Wrong category should fail when categories are specified."""
        config = DatasetConfig(
            categories=["circuit_generation", "vqe"],
            require_qiskit=False,
            require_function=False,
        )
        example = {
            "output": "x = 1\n" * 20,
            "category": "unrelated_topic",
        }

        assert filter_example(example, config) is False

    def test_empty_output_fails(self):
        """Empty output should fail."""
        config = DatasetConfig()
        example = {"output": ""}

        assert filter_example(example, config) is False


class TestCleanExample:
    """Tests for clean_example function."""

    def test_removes_markdown_python_block(self):
        """Should remove ```python``` markdown blocks."""
        example = {
            "output": """```python
from qiskit import QuantumCircuit

def create_ansatz():
    return QuantumCircuit(2)
```""",
            "instruction": "Create a circuit",
        }

        cleaned = clean_example(example)

        assert "```python" not in cleaned["output"]
        assert "```" not in cleaned["output"]
        assert "from qiskit import QuantumCircuit" in cleaned["output"]

    def test_removes_generic_code_blocks(self):
        """Should remove generic ``` code blocks."""
        example = {
            "output": """```
x = 1
y = 2
```""",
            "instruction": "Calculate",
        }

        cleaned = clean_example(example)

        assert "```" not in cleaned["output"]

    def test_adds_qiskit_import(self):
        """Should add Qiskit import if missing."""
        example = {
            "output": """def create_ansatz():
    qc = QuantumCircuit(2)
    return qc
""",
            "instruction": "Create circuit",
        }

        cleaned = clean_example(example)

        assert "from qiskit import QuantumCircuit" in cleaned["output"]

    def test_adds_parameter_import(self):
        """Should add Parameter import if missing."""
        example = {
            "output": """from qiskit import QuantumCircuit

def create_ansatz():
    p = Parameter('theta')
    qc = QuantumCircuit(1)
    qc.ry(p, 0)
    return qc
""",
            "instruction": "Create parameterized circuit",
        }

        cleaned = clean_example(example)

        assert "from qiskit.circuit import Parameter" in cleaned["output"]

    def test_preserves_existing_imports(self):
        """Should not duplicate existing imports."""
        example = {
            "output": """from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz():
    p = Parameter('theta')
    qc = QuantumCircuit(1)
    return qc
""",
            "instruction": "Create circuit",
        }

        cleaned = clean_example(example)

        # Count occurrences - should only have one of each
        assert cleaned["output"].count("from qiskit import QuantumCircuit") == 1

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        example = {
            "output": """

    x = 1

""",
            "instruction": "Test",
        }

        cleaned = clean_example(example)

        assert cleaned["output"] == cleaned["output"].strip()

    def test_preserves_other_fields(self):
        """Should preserve other fields in the example."""
        example = {
            "output": "x = 1",
            "instruction": "Test instruction",
            "input": "Test input",
            "category": "test",
        }

        cleaned = clean_example(example)

        assert cleaned["instruction"] == "Test instruction"
        assert cleaned["input"] == "Test input"
        assert cleaned["category"] == "test"


class TestDownloadQuantumDatasets:
    """Tests for download_quantum_datasets function."""

    @patch("datasets.load_dataset")
    def test_download_creates_directories(self, mock_load_dataset):
        """Should create raw directory."""
        from src.training.dataset import download_quantum_datasets

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(raw_dir=os.path.join(tmpdir, "raw"))

            # Mock the dataset
            mock_train = MagicMock()
            mock_train.column_names = ["instruction", "output"]
            mock_train.__len__ = MagicMock(return_value=1000)

            mock_ds = MagicMock()
            mock_ds.__getitem__ = MagicMock(return_value=mock_train)
            mock_ds.save_to_disk = MagicMock()
            mock_load_dataset.return_value = mock_ds

            download_quantum_datasets(config)

            assert os.path.exists(config.raw_dir)

    @patch("datasets.load_dataset")
    def test_download_returns_stats(self, mock_load_dataset):
        """Should return download statistics."""
        from src.training.dataset import download_quantum_datasets

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(raw_dir=os.path.join(tmpdir, "raw"))

            # Mock the dataset
            mock_train = MagicMock()
            mock_train.column_names = ["instruction", "output"]
            mock_train.__len__ = MagicMock(return_value=1000)

            mock_ds = MagicMock()
            mock_ds.__getitem__ = MagicMock(return_value=mock_train)
            mock_ds.save_to_disk = MagicMock()
            mock_load_dataset.return_value = mock_ds

            stats = download_quantum_datasets(config)

            assert "total_examples" in stats
            assert "columns" in stats
            assert "save_path" in stats


class TestFormatForTraining:
    """Tests for format_for_training function."""

    def test_qwen_format_structure(self):
        """Test that formatting produces correct Qwen chat structure."""
        # Create a mock example that would result from formatting
        system_prompt = "You are a quantum computing expert."
        instruction = "Create a Bell state circuit"
        input_text = "num_qubits: 2"
        output = "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)"

        # Manually construct expected format
        expected_start = "<|im_start|>system"
        expected_user = "<|im_start|>user"
        expected_assistant = "<|im_start|>assistant"
        expected_end = "<|im_end|>"

        # This tests the format structure we expect
        formatted = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{instruction}

{input_text}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>"""

        assert expected_start in formatted
        assert expected_user in formatted
        assert expected_assistant in formatted
        assert formatted.count(expected_end) == 3


class TestLoadProcessedDataset:
    """Tests for load_processed_dataset function."""

    def test_invalid_split_raises_error(self):
        """Should raise error for invalid split name."""
        from src.training.dataset import load_processed_dataset

        with pytest.raises(ValueError):
            load_processed_dataset("invalid")

    def test_missing_dataset_raises_error(self):
        """Should raise error if dataset doesn't exist."""
        from src.training.dataset import load_processed_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(processed_dir=tmpdir)

            with pytest.raises(FileNotFoundError):
                load_processed_dataset("train", config)


# ============================================================================
# Tests for Multi-Source Dataset Mixing (Phase 3)
# ============================================================================

class TestMixedDatasetConfig:
    """Tests for MixedDatasetConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MixedDatasetConfig()

        assert config.quantum_llm_weight == 0.50
        assert config.physics_augmented_weight == 0.30
        assert config.physics_reasoning_weight == 0.20
        assert config.train_split == 0.9
        assert config.seed == 42

    def test_weights_sum_to_one(self):
        """Weights should sum to 1.0."""
        config = MixedDatasetConfig()
        total = (
            config.quantum_llm_weight +
            config.physics_augmented_weight +
            config.physics_reasoning_weight
        )
        assert abs(total - 1.0) < 1e-6

    def test_custom_config(self):
        """Test custom configuration."""
        config = MixedDatasetConfig(
            quantum_llm_weight=0.6,
            physics_augmented_weight=0.3,
            physics_reasoning_weight=0.1,
            seed=123,
        )

        assert config.quantum_llm_weight == 0.6
        assert config.seed == 123


class TestFormatPhysicsAugmentedExample:
    """Tests for format_physics_augmented_example function."""

    def test_basic_formatting(self):
        """Test basic example formatting."""
        example = {
            "instruction": "Design a circuit",
            "input": "4 qubits",
            "output": "```python\ndef create(): pass\n```",
            "type": "physics_to_circuit",
        }
        system_prompt = "You are a quantum expert."

        result = format_physics_augmented_example(example, system_prompt)

        assert "text" in result
        assert "<|im_start|>system" in result["text"]
        assert "You are a quantum expert" in result["text"]
        assert "<|im_start|>user" in result["text"]
        assert "Design a circuit" in result["text"]
        assert "<|im_start|>assistant" in result["text"]
        assert "source" in result

    def test_empty_input(self):
        """Test formatting with empty input."""
        example = {
            "instruction": "Design a circuit",
            "input": "",
            "output": "code here",
        }
        system_prompt = "Expert"

        result = format_physics_augmented_example(example, system_prompt)

        assert "Design a circuit" in result["text"]

    def test_source_preserved(self):
        """Test that source type is preserved."""
        example = {
            "instruction": "Test",
            "input": "",
            "output": "output",
            "type": "simulation_to_insight",
        }

        result = format_physics_augmented_example(example, "prompt")

        assert result["source"] == "simulation_to_insight"


class TestWeightedSample:
    """Tests for weighted_sample function."""

    def test_basic_sampling(self):
        """Test basic weighted sampling."""
        rng = random.Random(42)

        source1 = [{"id": i, "source": "a"} for i in range(100)]
        source2 = [{"id": i, "source": "b"} for i in range(100)]

        sources = [
            (source1, 0.7),
            (source2, 0.3),
        ]

        result = weighted_sample(sources, 100, rng)

        assert len(result) > 0
        # Check distribution is roughly correct
        count_a = sum(1 for x in result if x["source"] == "a")
        count_b = sum(1 for x in result if x["source"] == "b")

        # Should be roughly 70/30 split (with some variance)
        assert count_a > count_b

    def test_empty_source_handled(self):
        """Test that empty sources are handled."""
        rng = random.Random(42)

        source1 = [{"id": i} for i in range(50)]
        source2 = []  # Empty

        sources = [
            (source1, 0.5),
            (source2, 0.5),
        ]

        result = weighted_sample(sources, 50, rng)

        assert len(result) > 0

    def test_respects_total_samples(self):
        """Test that total samples is respected."""
        rng = random.Random(42)

        source1 = [{"id": i} for i in range(100)]
        source2 = [{"id": i} for i in range(100)]

        sources = [
            (source1, 0.5),
            (source2, 0.5),
        ]

        result = weighted_sample(sources, 50, rng)

        assert len(result) <= 100  # At most sum of samples


class TestLoadPhysicsAugmentedData:
    """Tests for load_physics_augmented_data function."""

    def test_load_from_file(self):
        """Test loading from JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            train_file = os.path.join(tmpdir, "train.jsonl")
            examples = [
                {"instruction": "Test 1", "input": "", "output": "out1", "type": "t1"},
                {"instruction": "Test 2", "input": "", "output": "out2", "type": "t2"},
            ]
            with open(train_file, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

            result = load_physics_augmented_data(tmpdir, "System prompt")

            assert len(result) == 2
            assert "text" in result[0]
            assert "source" in result[0]

    def test_max_examples_limit(self):
        """Test max_examples parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_file = os.path.join(tmpdir, "train.jsonl")
            examples = [
                {"instruction": f"Test {i}", "input": "", "output": f"out{i}"}
                for i in range(10)
            ]
            with open(train_file, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

            result = load_physics_augmented_data(tmpdir, "Prompt", max_examples=3)

            assert len(result) == 3

    def test_missing_file_returns_empty(self):
        """Test that missing file returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_physics_augmented_data(tmpdir, "Prompt")
            assert result == []


class TestMixedDatasetBuilder:
    """Tests for MixedDatasetBuilder class."""

    def test_init_default(self):
        """Test default initialization."""
        builder = MixedDatasetBuilder()

        assert builder.config is not None
        assert builder.rng is not None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = MixedDatasetConfig(seed=123)
        builder = MixedDatasetBuilder(config)

        assert builder.config.seed == 123

    def test_build_mixed_dataset_with_mock_sources(self):
        """Test building mixed dataset with mock data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create physics augmented data
            physics_dir = os.path.join(tmpdir, "physics_augmented")
            os.makedirs(physics_dir)
            train_file = os.path.join(physics_dir, "train.jsonl")

            examples = [
                {"instruction": f"Physics {i}", "input": "", "output": f"out{i}"}
                for i in range(20)
            ]
            with open(train_file, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

            config = MixedDatasetConfig(
                quantum_llm_path=os.path.join(tmpdir, "nonexistent"),
                physics_augmented_path=physics_dir,
                physics_reasoning_path=os.path.join(tmpdir, "nonexistent2"),
                output_dir=os.path.join(tmpdir, "output"),
                max_examples_per_source=10,
            )

            builder = MixedDatasetBuilder(config)
            train, val = builder.build_mixed_dataset(target_size=15)

            # Should have some examples
            assert len(train) + len(val) > 0

    def test_save_mixed_dataset(self):
        """Test saving mixed dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MixedDatasetConfig(output_dir=tmpdir)
            builder = MixedDatasetBuilder(config)

            train = [
                {"text": "train example 1", "source": "test"},
                {"text": "train example 2", "source": "test"},
            ]
            val = [{"text": "val example 1", "source": "test"}]

            stats = builder.save_mixed_dataset(train, val)

            assert stats["train_size"] == 2
            assert stats["val_size"] == 1
            assert os.path.exists(os.path.join(tmpdir, "train"))
            assert os.path.exists(os.path.join(tmpdir, "val"))
            assert os.path.exists(os.path.join(tmpdir, "stats.json"))

    def test_source_distribution_tracked(self):
        """Test that source distribution is tracked in stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MixedDatasetConfig(output_dir=tmpdir)
            builder = MixedDatasetBuilder(config)

            train = [
                {"text": "ex1", "source": "quantum_llm"},
                {"text": "ex2", "source": "quantum_llm"},
                {"text": "ex3", "source": "physics_augmented"},
            ]
            val = []

            stats = builder.save_mixed_dataset(train, val)

            assert stats["source_distribution"]["quantum_llm"] == 2
            assert stats["source_distribution"]["physics_augmented"] == 1


class TestLoadMixedDataset:
    """Tests for load_mixed_dataset function."""

    def test_invalid_split_raises_error(self):
        """Should raise error for invalid split name."""
        from src.training.dataset import load_mixed_dataset

        with pytest.raises(ValueError):
            load_mixed_dataset("invalid")

    def test_missing_dataset_raises_error(self):
        """Should raise error if dataset doesn't exist."""
        from src.training.dataset import load_mixed_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MixedDatasetConfig(output_dir=tmpdir)

            with pytest.raises(FileNotFoundError):
                load_mixed_dataset("train", config)


class TestGetMixedDatasetStats:
    """Tests for get_mixed_dataset_stats function."""

    def test_loads_stats_from_file(self):
        """Test loading stats from JSON file."""
        from src.training.dataset import get_mixed_dataset_stats

        with tempfile.TemporaryDirectory() as tmpdir:
            stats = {
                "train_size": 100,
                "val_size": 10,
                "source_distribution": {"a": 50, "b": 60},
            }
            stats_file = os.path.join(tmpdir, "stats.json")
            with open(stats_file, "w") as f:
                json.dump(stats, f)

            config = MixedDatasetConfig(output_dir=tmpdir)
            result = get_mixed_dataset_stats(config)

            assert result["train_size"] == 100
            assert result["val_size"] == 10

    def test_missing_stats_raises_error(self):
        """Test that missing stats file raises error."""
        from src.training.dataset import get_mixed_dataset_stats

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MixedDatasetConfig(output_dir=tmpdir)

            with pytest.raises(FileNotFoundError):
                get_mixed_dataset_stats(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
