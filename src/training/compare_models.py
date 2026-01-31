"""
Model comparison script for evaluating physics-augmented fine-tuning.

Compares three models:
1. Base model (Qwen2.5-Coder-7B-Instruct)
2. Code-only fine-tuned (quantum-mind-v1)
3. Physics-augmented (quantum-mind-v2)

Usage:
    python -m src.training.compare_models
    python -m src.training.compare_models --output results/comparison.yaml
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import yaml


@dataclass
class ComparisonConfig:
    """Configuration for model comparison."""

    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    code_only_model: str = "models/checkpoints/quantum-mind-v1"
    physics_augmented_model: str = "models/checkpoints/quantum-mind-v2"
    num_prompts: int = 100
    output_path: str = "results/model_comparison.yaml"
    seed: int = 42


def load_validation_results(model_path: str) -> dict | None:
    """
    Load validation results for a model.

    Args:
        model_path: Path to model checkpoint

    Returns:
        Validation results or None if not found
    """
    results_path = os.path.join(model_path, "validation_results.json")

    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)

    return None


def run_validation_if_needed(
    model_path: str,
    is_base: bool,
    num_prompts: int,
) -> dict:
    """
    Run validation if results don't exist.

    Args:
        model_path: Path to model
        is_base: Whether this is a base model
        num_prompts: Number of prompts

    Returns:
        Validation results
    """
    from src.training.validate_model import (
        ValidationConfig,
        run_validation,
        save_validation_results,
    )

    # Check for existing results
    existing = load_validation_results(model_path)
    if existing and existing.get("num_prompts", 0) >= num_prompts:
        print(f"Using existing validation results for {model_path}")
        return existing

    # Run validation
    print(f"\nRunning validation for: {model_path}")
    config = ValidationConfig(
        model_path=model_path,
        is_base_model=is_base,
        num_prompts=num_prompts,
        output_dir=model_path if not is_base else "results/base_model",
    )

    results = run_validation(config)
    save_validation_results(results, config)

    return results


def compare_results(results: dict[str, dict]) -> dict:
    """
    Compare validation results across models.

    Args:
        results: Dictionary mapping model name to results

    Returns:
        Comparison summary
    """
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "models": list(results.keys()),
        "metrics": {},
        "by_category": {},
        "improvements": {},
    }

    # Collect metrics for each model
    metrics_to_compare = [
        "syntax_validity",
        "verification_pass",
        "bp_free",
        "has_code",
        "mentions_symmetry",
        "mentions_conservation",
        "explains_design",
    ]

    for metric in metrics_to_compare:
        comparison["metrics"][metric] = {}
        for model_name, model_results in results.items():
            if "rates" in model_results:
                comparison["metrics"][metric][model_name] = model_results["rates"].get(
                    metric, 0
                )

    # Compare by category
    categories = set()
    for model_results in results.values():
        if "by_category" in model_results:
            categories.update(model_results["by_category"].keys())

    for category in categories:
        comparison["by_category"][category] = {}
        for model_name, model_results in results.items():
            if "by_category" in model_results and category in model_results["by_category"]:
                cat_data = model_results["by_category"][category]
                n = cat_data.get("count", 1)
                comparison["by_category"][category][model_name] = {
                    "syntax_validity": cat_data.get("syntax_valid", 0) / n,
                    "verification_pass": cat_data.get("verification_passed", 0) / n,
                    "bp_free": cat_data.get("bp_free", 0) / n,
                }

    # Calculate improvements (physics-augmented vs others)
    if "physics_augmented" in results and "base" in results:
        base_rates = results["base"].get("rates", {})
        physics_rates = results["physics_augmented"].get("rates", {})

        for metric in metrics_to_compare:
            base_val = base_rates.get(metric, 0)
            physics_val = physics_rates.get(metric, 0)

            if base_val > 0:
                improvement = (physics_val - base_val) / base_val
            else:
                improvement = physics_val if physics_val > 0 else 0

            comparison["improvements"][f"vs_base_{metric}"] = improvement

    if "physics_augmented" in results and "code_only" in results:
        code_rates = results["code_only"].get("rates", {})
        physics_rates = results["physics_augmented"].get("rates", {})

        for metric in metrics_to_compare:
            code_val = code_rates.get(metric, 0)
            physics_val = physics_rates.get(metric, 0)

            if code_val > 0:
                improvement = (physics_val - code_val) / code_val
            else:
                improvement = physics_val if physics_val > 0 else 0

            comparison["improvements"][f"vs_code_only_{metric}"] = improvement

    return comparison


def print_comparison(comparison: dict):
    """Print comparison results."""
    print("\n" + "=" * 70)
    print("Model Comparison Results")
    print("=" * 70)

    models = comparison["models"]

    # Print metrics table
    print("\nOverall Metrics:")
    print("-" * 70)
    header = f"{'Metric':<25}" + "".join(f"{m:<15}" for m in models)
    print(header)
    print("-" * 70)

    for metric, values in comparison["metrics"].items():
        row = f"{metric:<25}"
        for model in models:
            val = values.get(model, 0)
            row += f"{val:<15.1%}"
        print(row)

    # Print improvements
    if comparison["improvements"]:
        print("\n" + "-" * 70)
        print("Physics-Augmented Improvements:")
        print("-" * 70)

        for key, val in comparison["improvements"].items():
            direction = "+" if val >= 0 else ""
            print(f"  {key}: {direction}{val:.1%}")

    # Print by category
    if comparison["by_category"]:
        print("\n" + "-" * 70)
        print("By Category (Verification Pass Rate):")
        print("-" * 70)

        for category in comparison["by_category"]:
            print(f"\n  {category}:")
            for model in models:
                if model in comparison["by_category"][category]:
                    rate = comparison["by_category"][category][model].get(
                        "verification_pass", 0
                    )
                    print(f"    {model}: {rate:.1%}")


def save_comparison(comparison: dict, output_path: str):
    """Save comparison results to file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(comparison, f, default_flow_style=False)

    print(f"\nComparison saved to: {output_path}")


def run_comparison(config: ComparisonConfig) -> dict:
    """
    Run full model comparison.

    Args:
        config: Comparison configuration

    Returns:
        Comparison results
    """
    print("=" * 70)
    print("QUASAR Model Comparison")
    print("=" * 70)
    print(f"\nModels to compare:")
    print(f"  1. Base: {config.base_model}")
    print(f"  2. Code-only: {config.code_only_model}")
    print(f"  3. Physics-augmented: {config.physics_augmented_model}")
    print()

    results = {}

    # Validate base model (if it exists or is accessible)
    try:
        results["base"] = run_validation_if_needed(
            config.base_model,
            is_base=True,
            num_prompts=config.num_prompts,
        )
    except Exception as e:
        print(f"Could not validate base model: {e}")
        results["base"] = {"rates": {}}

    # Validate code-only model (if exists)
    if os.path.exists(config.code_only_model):
        try:
            results["code_only"] = run_validation_if_needed(
                config.code_only_model,
                is_base=False,
                num_prompts=config.num_prompts,
            )
        except Exception as e:
            print(f"Could not validate code-only model: {e}")
            results["code_only"] = {"rates": {}}
    else:
        print(f"Code-only model not found at {config.code_only_model}")
        results["code_only"] = {"rates": {}}

    # Validate physics-augmented model
    if os.path.exists(config.physics_augmented_model):
        try:
            results["physics_augmented"] = run_validation_if_needed(
                config.physics_augmented_model,
                is_base=False,
                num_prompts=config.num_prompts,
            )
        except Exception as e:
            print(f"Could not validate physics-augmented model: {e}")
            results["physics_augmented"] = {"rates": {}}
    else:
        print(f"Physics-augmented model not found at {config.physics_augmented_model}")
        results["physics_augmented"] = {"rates": {}}

    # Compare results
    comparison = compare_results(results)

    return comparison


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare fine-tuned models")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Base model path",
    )
    parser.add_argument(
        "--code-only",
        type=str,
        default="models/checkpoints/quantum-mind-v1",
        help="Code-only fine-tuned model path",
    )
    parser.add_argument(
        "--physics-augmented",
        type=str,
        default="models/checkpoints/quantum-mind-v2",
        help="Physics-augmented model path",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=20,
        help="Number of prompts per model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/model_comparison.yaml",
        help="Output path for comparison results",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation, only compare existing results",
    )

    args = parser.parse_args()

    config = ComparisonConfig(
        base_model=args.base_model,
        code_only_model=args.code_only,
        physics_augmented_model=args.physics_augmented,
        num_prompts=args.num_prompts,
        output_path=args.output,
    )

    if args.skip_validation:
        # Load existing results only
        results = {}
        for name, path in [
            ("base", config.base_model),
            ("code_only", config.code_only_model),
            ("physics_augmented", config.physics_augmented_model),
        ]:
            existing = load_validation_results(path)
            if existing:
                results[name] = existing
            else:
                print(f"No existing results for {name}")
                results[name] = {"rates": {}}

        comparison = compare_results(results)
    else:
        comparison = run_comparison(config)

    # Print and save results
    print_comparison(comparison)
    save_comparison(comparison, config.output_path)


if __name__ == "__main__":
    main()
