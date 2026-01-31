"""
Dataset preparation module for QuantumMind.

This module handles:
1. Downloading QuantumLLMInstruct dataset from HuggingFace
2. Filtering for circuit generation examples
3. Cleaning and validating data
4. Formatting for Qwen fine-tuning
5. Multi-source dataset mixing with weighted sampling

Usage:
    python -m src.training.dataset --download
    python -m src.training.dataset --prepare
    python -m src.training.dataset --mix
"""

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from tqdm import tqdm


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    min_output_length: int = 50
    max_output_length: int = 5000
    require_qiskit: bool = True
    require_function: bool = True
    train_split: float = 0.9
    seed: int = 42
    categories: list[str] = field(
        default_factory=lambda: [
            "circuit_generation",
            "ansatz_design",
            "vqe",
            "qaoa",
            "hamiltonian",
            "quantum_algorithm",
        ]
    )


def download_quantum_datasets(config: DatasetConfig | None = None) -> dict:
    """
    Download QuantumLLMInstruct dataset from HuggingFace.

    Args:
        config: Dataset configuration

    Returns:
        Dictionary with download statistics
    """
    from datasets import load_dataset

    config = config or DatasetConfig()
    os.makedirs(config.raw_dir, exist_ok=True)

    print("Downloading QuantumLLMInstruct dataset...")
    print("This may take a few minutes...")

    try:
        dataset = load_dataset("BoltzmannEntropy/QuantumLLMInstruct")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Make sure you have internet access and the datasets library installed.")
        raise

    # Save to disk
    save_path = os.path.join(config.raw_dir, "quantum_llm_instruct")
    dataset.save_to_disk(save_path)

    stats = {
        "total_examples": len(dataset["train"]),
        "columns": dataset["train"].column_names,
        "save_path": save_path,
    }

    print(f"Downloaded {stats['total_examples']} examples")
    print(f"Saved to: {save_path}")

    return stats


def filter_example(example: dict, config: DatasetConfig) -> bool:
    """
    Filter a single example based on configuration.

    Args:
        example: Dataset example
        config: Filter configuration

    Returns:
        True if example should be kept
    """
    output = example.get("output", "")

    # Length checks
    if len(output) < config.min_output_length:
        return False
    if len(output) > config.max_output_length:
        return False

    # Content checks
    if config.require_qiskit:
        if "QuantumCircuit" not in output and "from qiskit" not in output.lower():
            return False

    if config.require_function:
        if "def " not in output:
            return False

    # Category check (if available)
    if "category" in example:
        category = example["category"]
        if config.categories and category not in config.categories:
            return False

    # Syntax validation
    try:
        compile(output, "<string>", "exec")
    except SyntaxError:
        return False

    return True


def clean_example(example: dict) -> dict:
    """
    Clean and standardize a single example.

    Args:
        example: Raw example

    Returns:
        Cleaned example
    """
    output = example.get("output", "")

    # Remove markdown code blocks
    if "```python" in output:
        start = output.find("```python")
        end = output.find("```", start + 9)
        if start != -1 and end != -1:
            output = output[start + 9 : end].strip()
    elif "```" in output:
        output = output.replace("```", "").strip()

    # Ensure imports
    if "QuantumCircuit" in output and "from qiskit" not in output.lower():
        output = "from qiskit import QuantumCircuit\n" + output

    if "Parameter" in output and "from qiskit.circuit import Parameter" not in output:
        if "from qiskit import" in output:
            # Add to existing import
            output = output.replace(
                "from qiskit import QuantumCircuit",
                "from qiskit import QuantumCircuit\nfrom qiskit.circuit import Parameter",
            )
        else:
            output = "from qiskit.circuit import Parameter\n" + output

    return {
        **example,
        "output": output.strip(),
    }


def filter_dataset(
    config: DatasetConfig | None = None, verbose: bool = True
) -> dict:
    """
    Filter the raw dataset based on configuration.

    Args:
        config: Filter configuration
        verbose: Print progress

    Returns:
        Dictionary with filter statistics
    """
    from datasets import load_from_disk

    config = config or DatasetConfig()

    raw_path = os.path.join(config.raw_dir, "quantum_llm_instruct")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Run download first."
        )

    if verbose:
        print(f"Loading raw dataset from {raw_path}...")

    dataset = load_from_disk(raw_path)
    original_size = len(dataset["train"])

    if verbose:
        print(f"Original size: {original_size}")
        print("Filtering and cleaning...")

    # Filter
    filtered = dataset["train"].filter(
        lambda x: filter_example(x, config),
        desc="Filtering" if verbose else None,
    )

    # Clean
    cleaned = filtered.map(
        clean_example,
        desc="Cleaning" if verbose else None,
    )

    filtered_size = len(cleaned)

    # Save
    os.makedirs(config.processed_dir, exist_ok=True)
    save_path = os.path.join(config.processed_dir, "quantum_filtered")
    cleaned.save_to_disk(save_path)

    stats = {
        "original_size": original_size,
        "filtered_size": filtered_size,
        "retention_rate": filtered_size / original_size,
        "save_path": save_path,
    }

    if verbose:
        print(f"Filtered size: {filtered_size} ({stats['retention_rate']:.1%} retained)")
        print(f"Saved to: {save_path}")

    return stats


def format_for_training(
    config: DatasetConfig | None = None,
    system_prompt: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Format filtered dataset for Qwen fine-tuning.

    Args:
        config: Dataset configuration
        system_prompt: Custom system prompt (uses default if None)
        verbose: Print progress

    Returns:
        Dictionary with format statistics
    """
    from datasets import load_from_disk

    config = config or DatasetConfig()

    if system_prompt is None:
        system_prompt = """You are a quantum computing expert specializing in variational quantum algorithms. Your task is to generate efficient, trainable quantum circuits in Qiskit.

Key principles:
1. Always use Parameters for trainable values
2. Prefer shallow circuits over deep ones
3. Use local rotations followed by entanglement
4. Avoid patterns that cause barren plateaus
5. Match circuit structure to problem symmetry"""

    filtered_path = os.path.join(config.processed_dir, "quantum_filtered")
    if not os.path.exists(filtered_path):
        raise FileNotFoundError(
            f"Filtered dataset not found at {filtered_path}. Run filter first."
        )

    if verbose:
        print(f"Loading filtered dataset from {filtered_path}...")

    dataset = load_from_disk(filtered_path)

    def format_example(example):
        """Format a single example for Qwen chat format."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        # Combine instruction and input
        user_message = instruction
        if input_text:
            user_message += f"\n\n{input_text}"

        # Qwen chat format
        formatted = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>"""

        return {"text": formatted}

    if verbose:
        print("Formatting for training...")

    formatted = dataset.map(
        format_example,
        desc="Formatting" if verbose else None,
    )

    # Create train/val split
    split = formatted.train_test_split(
        test_size=1 - config.train_split,
        seed=config.seed,
    )

    # Save
    train_path = os.path.join(config.processed_dir, "train")
    val_path = os.path.join(config.processed_dir, "val")

    split["train"].save_to_disk(train_path)
    split["test"].save_to_disk(val_path)

    stats = {
        "train_size": len(split["train"]),
        "val_size": len(split["test"]),
        "train_path": train_path,
        "val_path": val_path,
        "system_prompt_length": len(system_prompt),
    }

    if verbose:
        print(f"Train size: {stats['train_size']}")
        print(f"Val size: {stats['val_size']}")
        print(f"Saved train to: {train_path}")
        print(f"Saved val to: {val_path}")

    # Save stats
    stats_path = os.path.join(config.processed_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def prepare_training_data(
    config: DatasetConfig | None = None, verbose: bool = True
) -> dict:
    """
    Run full data preparation pipeline.

    This downloads, filters, and formats the dataset.

    Args:
        config: Dataset configuration
        verbose: Print progress

    Returns:
        Dictionary with all statistics
    """
    config = config or DatasetConfig()

    # Check if already downloaded
    raw_path = os.path.join(config.raw_dir, "quantum_llm_instruct")
    if not os.path.exists(raw_path):
        if verbose:
            print("=" * 50)
            print("Step 1: Downloading dataset")
            print("=" * 50)
        download_stats = download_quantum_datasets(config)
    else:
        if verbose:
            print("Dataset already downloaded, skipping download.")
        download_stats = {"skipped": True}

    if verbose:
        print("\n" + "=" * 50)
        print("Step 2: Filtering dataset")
        print("=" * 50)
    filter_stats = filter_dataset(config, verbose)

    if verbose:
        print("\n" + "=" * 50)
        print("Step 3: Formatting for training")
        print("=" * 50)
    format_stats = format_for_training(config, verbose=verbose)

    return {
        "download": download_stats,
        "filter": filter_stats,
        "format": format_stats,
    }


def load_processed_dataset(split: str = "train", config: DatasetConfig | None = None):
    """
    Load processed dataset.

    Args:
        split: "train" or "val"
        config: Dataset configuration

    Returns:
        HuggingFace Dataset object
    """
    from datasets import load_from_disk

    config = config or DatasetConfig()

    if split not in ["train", "val"]:
        raise ValueError(f"split must be 'train' or 'val', got {split}")

    path = os.path.join(config.processed_dir, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. "
            "Run prepare_training_data first."
        )

    return load_from_disk(path)


def validate_dataset(config: DatasetConfig | None = None) -> dict:
    """
    Validate processed dataset quality.

    Args:
        config: Dataset configuration

    Returns:
        Validation statistics
    """
    config = config or DatasetConfig()

    train_data = load_processed_dataset("train", config)
    val_data = load_processed_dataset("val", config)

    stats = {
        "train_size": len(train_data),
        "val_size": len(val_data),
        "sample_lengths": [],
        "has_required_tags": 0,
    }

    # Check samples
    for i in range(min(100, len(train_data))):
        example = train_data[i]
        text = example["text"]
        stats["sample_lengths"].append(len(text))

        # Check for required format tags
        has_tags = (
            "<|im_start|>system" in text
            and "<|im_start|>user" in text
            and "<|im_start|>assistant" in text
        )
        if has_tags:
            stats["has_required_tags"] += 1

    stats["avg_length"] = sum(stats["sample_lengths"]) / len(stats["sample_lengths"])
    stats["tag_compliance"] = stats["has_required_tags"] / min(100, len(train_data))

    return stats


# ============================================================================
# Multi-Source Dataset Mixing (Phase 3)
# ============================================================================

@dataclass
class DataSourceConfig:
    """Configuration for a single data source."""

    name: str
    path: str
    weight: float
    format_fn: Callable[[dict], dict] | None = None


@dataclass
class MixedDatasetConfig:
    """Configuration for mixed dataset preparation."""

    # Data sources with weights
    quantum_llm_weight: float = 0.50  # QuantumLLMInstruct (code patterns)
    physics_augmented_weight: float = 0.30  # Physics-augmented (physics reasoning)
    physics_reasoning_weight: float = 0.20  # Physics reasoning (chain-of-thought)

    # Paths
    quantum_llm_path: str = "data/processed/train"
    physics_augmented_path: str = "data/physics_augmented"
    physics_reasoning_path: str = "data/physics_reasoning"
    output_dir: str = "data/mixed"

    # Settings
    train_split: float = 0.9
    seed: int = 42
    max_examples_per_source: int | None = None  # Limit per source (for testing)

    # System prompt for physics-augmented examples
    physics_system_prompt: str = """You are a quantum computing expert specializing in variational quantum algorithms and physics-informed circuit design.

Key principles:
1. Understand the physics of the problem before designing circuits
2. Respect symmetries and conservation laws in circuit structure
3. Match circuit depth to dynamics complexity
4. Use Parameters for trainable values
5. Avoid patterns that cause barren plateaus"""


def format_physics_augmented_example(
    example: dict,
    system_prompt: str,
) -> dict:
    """
    Format a physics-augmented example for training.

    Args:
        example: Raw example from physics_augmented dataset
        system_prompt: System prompt to use

    Returns:
        Formatted example with 'text' field
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    # Combine instruction and input
    user_message = instruction
    if input_text:
        user_message += f"\n\n{input_text}"

    # Qwen chat format
    formatted = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>"""

    return {"text": formatted, "source": example.get("type", "physics_augmented")}


def load_physics_augmented_data(
    path: str,
    system_prompt: str,
    max_examples: int | None = None,
) -> list[dict]:
    """
    Load and format physics-augmented dataset.

    Args:
        path: Path to physics_augmented directory
        system_prompt: System prompt to use
        max_examples: Optional limit on examples

    Returns:
        List of formatted examples
    """
    train_file = os.path.join(path, "train.jsonl")

    if not os.path.exists(train_file):
        print(f"Physics-augmented data not found at {train_file}")
        return []

    examples = []
    with open(train_file) as f:
        for line in f:
            if max_examples and len(examples) >= max_examples:
                break
            data = json.loads(line.strip())
            formatted = format_physics_augmented_example(data, system_prompt)
            examples.append(formatted)

    return examples


def load_physics_reasoning_data(
    path: str,
    system_prompt: str,
    max_examples: int | None = None,
) -> list[dict]:
    """
    Load physics reasoning (chain-of-thought) examples.

    These are manually curated or generated examples that
    demonstrate step-by-step physics reasoning.

    Args:
        path: Path to physics_reasoning directory
        system_prompt: System prompt to use
        max_examples: Optional limit on examples

    Returns:
        List of formatted examples
    """
    train_file = os.path.join(path, "train.jsonl")

    if not os.path.exists(train_file):
        print(f"Physics-reasoning data not found at {train_file}")
        return []

    examples = []
    with open(train_file) as f:
        for line in f:
            if max_examples and len(examples) >= max_examples:
                break
            data = json.loads(line.strip())

            # Format similar to physics-augmented
            formatted = format_physics_augmented_example(data, system_prompt)
            formatted["source"] = "physics_reasoning"
            examples.append(formatted)

    return examples


def weighted_sample(
    sources: list[tuple[list[dict], float]],
    total_samples: int,
    rng: random.Random,
) -> list[dict]:
    """
    Sample from multiple sources with given weights.

    Args:
        sources: List of (examples, weight) tuples
        total_samples: Total number of samples to draw
        rng: Random number generator

    Returns:
        Combined list of sampled examples
    """
    result = []

    # Calculate samples per source based on weights
    total_weight = sum(w for _, w in sources if len(_) > 0)
    if total_weight == 0:
        return result

    for examples, weight in sources:
        if len(examples) == 0:
            continue

        # Number of samples from this source
        n_samples = int(total_samples * weight / total_weight)
        n_samples = min(n_samples, len(examples))

        # Sample with replacement if needed
        if n_samples >= len(examples):
            sampled = examples.copy()
        else:
            sampled = rng.sample(examples, n_samples)

        result.extend(sampled)

    return result


class MixedDatasetBuilder:
    """
    Builder for mixed multi-source datasets.

    Combines QuantumLLMInstruct, physics-augmented, and physics-reasoning
    data with configurable weights.
    """

    def __init__(self, config: MixedDatasetConfig | None = None):
        """
        Initialize the builder.

        Args:
            config: Mixed dataset configuration
        """
        self.config = config or MixedDatasetConfig()
        self.rng = random.Random(self.config.seed)

    def load_all_sources(self) -> dict[str, list[dict]]:
        """
        Load all data sources.

        Returns:
            Dictionary mapping source name to list of examples
        """
        sources = {}

        # Load QuantumLLMInstruct (already formatted)
        try:
            from datasets import load_from_disk

            if os.path.exists(self.config.quantum_llm_path):
                quantum_llm = load_from_disk(self.config.quantum_llm_path)
                examples = [{"text": ex["text"], "source": "quantum_llm"} for ex in quantum_llm]
                if self.config.max_examples_per_source:
                    examples = examples[:self.config.max_examples_per_source]
                sources["quantum_llm"] = examples
                print(f"Loaded {len(examples)} QuantumLLMInstruct examples")
            else:
                print(f"QuantumLLMInstruct not found at {self.config.quantum_llm_path}")
                sources["quantum_llm"] = []
        except Exception as e:
            print(f"Error loading QuantumLLMInstruct: {e}")
            sources["quantum_llm"] = []

        # Load physics-augmented
        physics_aug = load_physics_augmented_data(
            self.config.physics_augmented_path,
            self.config.physics_system_prompt,
            self.config.max_examples_per_source,
        )
        sources["physics_augmented"] = physics_aug
        print(f"Loaded {len(physics_aug)} physics-augmented examples")

        # Load physics-reasoning
        physics_reason = load_physics_reasoning_data(
            self.config.physics_reasoning_path,
            self.config.physics_system_prompt,
            self.config.max_examples_per_source,
        )
        sources["physics_reasoning"] = physics_reason
        print(f"Loaded {len(physics_reason)} physics-reasoning examples")

        return sources

    def build_mixed_dataset(
        self,
        target_size: int | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Build mixed dataset with weighted sampling.

        Args:
            target_size: Target total size (uses all if None)

        Returns:
            Tuple of (train_examples, val_examples)
        """
        sources = self.load_all_sources()

        # Combine with weights
        source_weights = [
            (sources["quantum_llm"], self.config.quantum_llm_weight),
            (sources["physics_augmented"], self.config.physics_augmented_weight),
            (sources["physics_reasoning"], self.config.physics_reasoning_weight),
        ]

        # Calculate total available
        total_available = sum(len(s) for s, _ in source_weights)

        if target_size is None:
            target_size = total_available

        # Sample from sources
        combined = weighted_sample(source_weights, target_size, self.rng)

        # Shuffle
        self.rng.shuffle(combined)

        # Split
        n_val = int(len(combined) * (1 - self.config.train_split))
        val_examples = combined[:n_val]
        train_examples = combined[n_val:]

        return train_examples, val_examples

    def save_mixed_dataset(
        self,
        train_examples: list[dict],
        val_examples: list[dict],
    ) -> dict:
        """
        Save mixed dataset to disk.

        Args:
            train_examples: Training examples
            val_examples: Validation examples

        Returns:
            Statistics dictionary
        """
        from datasets import Dataset

        os.makedirs(self.config.output_dir, exist_ok=True)

        # Convert to HuggingFace datasets
        train_ds = Dataset.from_list(train_examples)
        val_ds = Dataset.from_list(val_examples)

        # Save
        train_path = os.path.join(self.config.output_dir, "train")
        val_path = os.path.join(self.config.output_dir, "val")

        train_ds.save_to_disk(train_path)
        val_ds.save_to_disk(val_path)

        # Compute source distribution
        source_counts = {}
        for ex in train_examples + val_examples:
            source = ex.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        stats = {
            "train_size": len(train_examples),
            "val_size": len(val_examples),
            "total_size": len(train_examples) + len(val_examples),
            "source_distribution": source_counts,
            "train_path": train_path,
            "val_path": val_path,
            "weights": {
                "quantum_llm": self.config.quantum_llm_weight,
                "physics_augmented": self.config.physics_augmented_weight,
                "physics_reasoning": self.config.physics_reasoning_weight,
            },
        }

        # Save stats
        stats_path = os.path.join(self.config.output_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\nMixed dataset saved:")
        print(f"  Train: {stats['train_size']} examples -> {train_path}")
        print(f"  Val: {stats['val_size']} examples -> {val_path}")
        print(f"  Source distribution: {source_counts}")

        return stats

    def build_and_save(self, target_size: int | None = None) -> dict:
        """
        Build and save mixed dataset.

        Args:
            target_size: Target total size

        Returns:
            Statistics dictionary
        """
        train, val = self.build_mixed_dataset(target_size)
        return self.save_mixed_dataset(train, val)


def load_mixed_dataset(split: str = "train", config: MixedDatasetConfig | None = None):
    """
    Load mixed dataset.

    Args:
        split: "train" or "val"
        config: Mixed dataset configuration

    Returns:
        HuggingFace Dataset object
    """
    from datasets import load_from_disk

    config = config or MixedDatasetConfig()

    if split not in ["train", "val"]:
        raise ValueError(f"split must be 'train' or 'val', got {split}")

    path = os.path.join(config.output_dir, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Mixed dataset not found at {path}. "
            "Run build_mixed_dataset first."
        )

    return load_from_disk(path)


def get_mixed_dataset_stats(config: MixedDatasetConfig | None = None) -> dict:
    """
    Get statistics for mixed dataset.

    Args:
        config: Mixed dataset configuration

    Returns:
        Statistics dictionary
    """
    config = config or MixedDatasetConfig()

    stats_path = os.path.join(config.output_dir, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)

    raise FileNotFoundError(f"Stats not found at {stats_path}")


def prepare_mixed_training_data(
    config: MixedDatasetConfig | None = None,
    target_size: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Prepare mixed training data from all sources.

    This is the main entry point for Phase 3 dataset preparation.

    Args:
        config: Mixed dataset configuration
        target_size: Target total size
        verbose: Print progress

    Returns:
        Statistics dictionary
    """
    config = config or MixedDatasetConfig()

    if verbose:
        print("=" * 50)
        print("Preparing Mixed Training Dataset")
        print("=" * 50)
        print(f"Weights: QuantumLLM={config.quantum_llm_weight:.0%}, "
              f"Physics-Aug={config.physics_augmented_weight:.0%}, "
              f"Physics-Reason={config.physics_reasoning_weight:.0%}")
        print()

    builder = MixedDatasetBuilder(config)
    stats = builder.build_and_save(target_size)

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset preparation for QuantumMind")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--filter", action="store_true", help="Filter dataset")
    parser.add_argument("--format", action="store_true", help="Format for training")
    parser.add_argument("--prepare", action="store_true", help="Run full pipeline")
    parser.add_argument("--validate", action="store_true", help="Validate dataset")
    parser.add_argument("--mix", action="store_true", help="Create mixed dataset")
    parser.add_argument("--mix-stats", action="store_true", help="Show mixed dataset stats")
    parser.add_argument("--target-size", type=int, default=None, help="Target size for mixed dataset")

    args = parser.parse_args()

    config = DatasetConfig()

    if args.download:
        download_quantum_datasets(config)
    elif args.filter:
        filter_dataset(config)
    elif args.format:
        format_for_training(config)
    elif args.prepare:
        prepare_training_data(config)
    elif args.validate:
        stats = validate_dataset(config)
        print("\nValidation Results:")
        print(f"  Train size: {stats['train_size']}")
        print(f"  Val size: {stats['val_size']}")
        print(f"  Avg sample length: {stats['avg_length']:.0f}")
        print(f"  Tag compliance: {stats['tag_compliance']:.1%}")
    elif args.mix:
        mixed_config = MixedDatasetConfig()
        stats = prepare_mixed_training_data(mixed_config, args.target_size)
        print("\nMixed Dataset Created:")
        print(f"  Total: {stats['total_size']} examples")
        print(f"  Distribution: {stats['source_distribution']}")
    elif args.mix_stats:
        try:
            stats = get_mixed_dataset_stats()
            print("\nMixed Dataset Stats:")
            print(f"  Train: {stats['train_size']}")
            print(f"  Val: {stats['val_size']}")
            print(f"  Distribution: {stats['source_distribution']}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
    else:
        parser.print_help()
