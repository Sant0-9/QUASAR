"""
Model validation script for physics-augmented quantum circuit generation.

Validates fine-tuned models on a suite of test prompts, measuring:
- Syntax validity (code compiles)
- Verification pass rate (circuit passes all checks)
- Barren plateau-free rate
- Physics reasoning quality

Usage:
    python -m src.training.validate_model --model models/checkpoints/quantum-mind-v2
    python -m src.training.validate_model --model Qwen/Qwen2.5-Coder-7B-Instruct --base
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch
import yaml


@dataclass
class ValidationConfig:
    """Configuration for model validation."""

    model_path: str = "models/checkpoints/quantum-mind-v2"
    is_base_model: bool = False  # True if validating base model without LoRA
    num_prompts: int = 100
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    output_dir: str = "models/checkpoints/quantum-mind-v2"
    seed: int = 42


# Validation prompts organized by category
VALIDATION_PROMPTS = {
    "physics_to_circuit": [
        "Design an ansatz for a 4-qubit Heisenberg chain that preserves SU(2) symmetry. Explain your design choices.",
        "Create a hardware-efficient circuit for the transverse field Ising model on 6 qubits with linear connectivity.",
        "Design a variational circuit for the XY model that preserves total magnetization (U(1) symmetry).",
        "Create an ansatz for a 2D Hubbard model on a 2x2 lattice that respects particle number conservation.",
        "Design a quantum circuit for molecular hydrogen (H2) in the STO-3G basis using 4 qubits.",
    ],
    "simulation_to_insight": [
        "Given a simulation showing periodic oscillations with conserved total mass, what circuit design would you recommend?",
        "A physics simulation exhibits chaotic dynamics with no apparent symmetries. How should I design my ansatz?",
        "The simulation data shows steady-state behavior with translational symmetry. What does this imply for circuit design?",
        "I observe a system with reflection symmetry and conserved energy. How should the ansatz preserve these properties?",
        "My simulation shows quasi-periodic behavior. What circuit depth and structure would be appropriate?",
    ],
    "conservation_to_constraint": [
        "Given a system with U(1) conservation (total magnetization), what gates should be avoided in the ansatz?",
        "Design a circuit that strictly preserves particle number. What two-qubit gates are allowed?",
        "How do I ensure my ansatz respects parity (Z2) symmetry? Which gates preserve this symmetry?",
        "Create a symmetry-preserving ansatz for a system with SU(2) spin rotation symmetry.",
        "What circuit constraints apply when modeling a system with time-reversal symmetry?",
    ],
    "general_quantum": [
        "Create a 4-qubit hardware-efficient ansatz with 3 layers of rotations and entanglement.",
        "Design a VQE ansatz for finding the ground state of a simple Hamiltonian.",
        "Implement a parameterized quantum circuit that avoids barren plateaus.",
        "Create a circuit with alternating RY rotations and CNOT gates for 6 qubits.",
        "Design an ansatz suitable for NISQ devices with limited qubit connectivity.",
    ],
}


def load_model_for_inference(config: ValidationConfig):
    """
    Load model for inference.

    Args:
        config: Validation configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading model from: {config.model_path}")

    if config.is_base_model:
        # Load base model without LoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # Load fine-tuned model with LoRA adapters
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            config.model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path if not config.is_base_model else config.model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    config: ValidationConfig,
    system_prompt: str | None = None,
) -> str:
    """
    Generate a response from the model.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: User prompt
        config: Validation configuration
        system_prompt: Optional system prompt

    Returns:
        Generated response text
    """
    if system_prompt is None:
        system_prompt = """You are a quantum computing expert specializing in variational quantum algorithms and physics-informed circuit design. Your task is to generate efficient, trainable quantum circuits in Qiskit.

Key principles:
1. Understand the physics of the problem before designing circuits
2. Respect symmetries and conservation laws in circuit structure
3. Use Parameters for trainable values
4. Avoid patterns that cause barren plateaus"""

    # Format as Qwen chat
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]

    return response.strip()


def validate_syntax(code: str) -> bool:
    """
    Check if generated code is syntactically valid Python.

    Args:
        code: Generated code string

    Returns:
        True if code compiles
    """
    # Extract code from markdown
    if "```python" in code:
        start = code.find("```python") + 9
        end = code.find("```", start)
        if end > start:
            code = code[start:end].strip()
    elif "```" in code:
        code = code.replace("```", "").strip()

    # Try to compile
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def validate_circuit(response: str) -> dict:
    """
    Validate the generated circuit.

    Args:
        response: Model response

    Returns:
        Dictionary with validation results
    """
    from src.quantum.verifier import CircuitVerifier

    result = {
        "has_code": False,
        "syntax_valid": False,
        "verification_passed": False,
        "bp_free": False,
        "details": {},
    }

    # Check for code
    if "def " not in response and "QuantumCircuit" not in response:
        return result

    result["has_code"] = True

    # Check syntax
    result["syntax_valid"] = validate_syntax(response)
    if not result["syntax_valid"]:
        return result

    # Try to run circuit verification
    try:
        verifier = CircuitVerifier()
        verification = verifier.verify(response)
        result["verification_passed"] = verification.is_valid
        result["bp_free"] = not verification.details.get("has_barren_plateau", True)
        result["details"] = verification.details
    except Exception as e:
        result["details"]["error"] = str(e)

    return result


def evaluate_physics_reasoning(response: str, prompt_category: str) -> dict:
    """
    Evaluate the quality of physics reasoning in the response.

    Args:
        response: Model response
        prompt_category: Category of the prompt

    Returns:
        Dictionary with reasoning quality metrics
    """
    result = {
        "mentions_symmetry": False,
        "mentions_conservation": False,
        "explains_design": False,
        "reasoning_length": 0,
    }

    response_lower = response.lower()

    # Check for physics concepts
    symmetry_terms = ["symmetr", "invariant", "su(2)", "u(1)", "z2", "parity"]
    conservation_terms = ["conserv", "preserv", "commut", "sector"]
    design_terms = ["because", "therefore", "reason", "choice", "design"]

    result["mentions_symmetry"] = any(term in response_lower for term in symmetry_terms)
    result["mentions_conservation"] = any(
        term in response_lower for term in conservation_terms
    )
    result["explains_design"] = any(term in response_lower for term in design_terms)

    # Count reasoning text (non-code)
    lines = response.split("\n")
    reasoning_lines = [
        line for line in lines
        if not line.strip().startswith(("from ", "import ", "def ", "    ", "```"))
    ]
    result["reasoning_length"] = len(" ".join(reasoning_lines))

    return result


def run_validation(config: ValidationConfig) -> dict:
    """
    Run complete validation suite.

    Args:
        config: Validation configuration

    Returns:
        Validation results dictionary
    """
    import random

    print("=" * 60)
    print("Model Validation")
    print("=" * 60)

    # Set seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load model
    model, tokenizer = load_model_for_inference(config)

    # Collect all prompts
    all_prompts = []
    for category, prompts in VALIDATION_PROMPTS.items():
        for prompt in prompts:
            all_prompts.append((category, prompt))

    # Limit to config.num_prompts
    if len(all_prompts) > config.num_prompts:
        all_prompts = random.sample(all_prompts, config.num_prompts)

    # Run validation
    results = {
        "model_path": config.model_path,
        "is_base_model": config.is_base_model,
        "num_prompts": len(all_prompts),
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "syntax_valid": 0,
            "verification_passed": 0,
            "bp_free": 0,
            "has_code": 0,
            "mentions_symmetry": 0,
            "mentions_conservation": 0,
            "explains_design": 0,
        },
        "by_category": {},
        "examples": [],
    }

    print(f"\nValidating on {len(all_prompts)} prompts...")
    print()

    for i, (category, prompt) in enumerate(all_prompts):
        print(f"[{i+1}/{len(all_prompts)}] {category}: {prompt[:50]}...")

        # Generate response
        response = generate_response(model, tokenizer, prompt, config)

        # Validate circuit
        circuit_result = validate_circuit(response)

        # Evaluate reasoning
        reasoning_result = evaluate_physics_reasoning(response, category)

        # Update metrics
        if circuit_result["has_code"]:
            results["metrics"]["has_code"] += 1
        if circuit_result["syntax_valid"]:
            results["metrics"]["syntax_valid"] += 1
        if circuit_result["verification_passed"]:
            results["metrics"]["verification_passed"] += 1
        if circuit_result["bp_free"]:
            results["metrics"]["bp_free"] += 1
        if reasoning_result["mentions_symmetry"]:
            results["metrics"]["mentions_symmetry"] += 1
        if reasoning_result["mentions_conservation"]:
            results["metrics"]["mentions_conservation"] += 1
        if reasoning_result["explains_design"]:
            results["metrics"]["explains_design"] += 1

        # Track by category
        if category not in results["by_category"]:
            results["by_category"][category] = {
                "count": 0,
                "syntax_valid": 0,
                "verification_passed": 0,
                "bp_free": 0,
            }

        results["by_category"][category]["count"] += 1
        if circuit_result["syntax_valid"]:
            results["by_category"][category]["syntax_valid"] += 1
        if circuit_result["verification_passed"]:
            results["by_category"][category]["verification_passed"] += 1
        if circuit_result["bp_free"]:
            results["by_category"][category]["bp_free"] += 1

        # Save example (first 5)
        if len(results["examples"]) < 5:
            results["examples"].append({
                "category": category,
                "prompt": prompt,
                "response": response[:1000] + "..." if len(response) > 1000 else response,
                "circuit_result": circuit_result,
                "reasoning_result": reasoning_result,
            })

    # Calculate rates
    n = len(all_prompts)
    results["rates"] = {
        "syntax_validity": results["metrics"]["syntax_valid"] / n,
        "verification_pass": results["metrics"]["verification_passed"] / n,
        "bp_free": results["metrics"]["bp_free"] / n,
        "has_code": results["metrics"]["has_code"] / n,
        "mentions_symmetry": results["metrics"]["mentions_symmetry"] / n,
        "mentions_conservation": results["metrics"]["mentions_conservation"] / n,
        "explains_design": results["metrics"]["explains_design"] / n,
    }

    return results


def save_validation_results(results: dict, config: ValidationConfig):
    """
    Save validation results to file.

    Args:
        results: Validation results
        config: Validation configuration
    """
    os.makedirs(config.output_dir, exist_ok=True)

    # Save full results as JSON
    json_path = os.path.join(config.output_dir, "validation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save summary as YAML
    yaml_path = os.path.join(config.output_dir, "validation_log.yaml")
    summary = {
        "model_path": results["model_path"],
        "is_base_model": results["is_base_model"],
        "timestamp": results["timestamp"],
        "num_prompts": results["num_prompts"],
        "rates": results["rates"],
        "by_category": {
            cat: {
                "syntax_validity": data["syntax_valid"] / data["count"] if data["count"] > 0 else 0,
                "verification_pass": data["verification_passed"] / data["count"] if data["count"] > 0 else 0,
                "bp_free": data["bp_free"] / data["count"] if data["count"] > 0 else 0,
            }
            for cat, data in results["by_category"].items()
        },
    }

    with open(yaml_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    print(f"\nResults saved to:")
    print(f"  Full: {json_path}")
    print(f"  Summary: {yaml_path}")


def print_results(results: dict):
    """Print validation results summary."""
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)

    rates = results["rates"]
    print(f"\nOverall Metrics (n={results['num_prompts']}):")
    print(f"  Syntax validity:     {rates['syntax_validity']:.1%}")
    print(f"  Verification pass:   {rates['verification_pass']:.1%}")
    print(f"  BP-free rate:        {rates['bp_free']:.1%}")
    print(f"  Has code:            {rates['has_code']:.1%}")
    print(f"  Mentions symmetry:   {rates['mentions_symmetry']:.1%}")
    print(f"  Mentions conserv.:   {rates['mentions_conservation']:.1%}")
    print(f"  Explains design:     {rates['explains_design']:.1%}")

    print("\nBy Category:")
    for cat, data in results["by_category"].items():
        n = data["count"]
        print(f"  {cat}:")
        print(f"    Syntax: {data['syntax_valid']/n:.1%}, "
              f"Verify: {data['verification_passed']/n:.1%}, "
              f"BP-free: {data['bp_free']/n:.1%}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate fine-tuned model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/checkpoints/quantum-mind-v2",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="Validate base model (no LoRA adapters)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=20,
        help="Number of prompts to validate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )

    args = parser.parse_args()

    config = ValidationConfig(
        model_path=args.model,
        is_base_model=args.base,
        num_prompts=args.num_prompts,
        output_dir=args.output_dir or args.model,
    )

    # Run validation
    results = run_validation(config)

    # Print results
    print_results(results)

    # Save results
    save_validation_results(results, config)


if __name__ == "__main__":
    main()
