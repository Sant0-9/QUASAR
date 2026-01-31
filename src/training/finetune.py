"""
Fine-tuning script for physics-augmented quantum circuit generation.

This script fine-tunes Qwen2.5-Coder-7B-Instruct using QLoRA on a mixed
dataset of QuantumLLMInstruct and physics-augmented examples.

Usage:
    # Basic training
    python -m src.training.finetune

    # With custom config
    python -m src.training.finetune --config configs/training.yaml

    # Resume from checkpoint
    python -m src.training.finetune --resume models/checkpoints/quantum-mind-v2/checkpoint-500

Requirements:
    - NVIDIA GPU with 24GB+ VRAM (A100 recommended)
    - transformers, peft, bitsandbytes, trl installed
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    task_type: str = "CAUSAL_LM"


@dataclass
class QuantizationConfig:
    """4-bit quantization configuration."""

    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


@dataclass
class FinetuneConfig:
    """Complete fine-tuning configuration."""

    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Quantization
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Dataset
    dataset_dir: str = "data/mixed"

    # Output
    output_dir: str = "models/checkpoints/quantum-mind-v2"
    logging_dir: str = "logs/training"

    # Resume
    resume_from_checkpoint: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "FinetuneConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        config = cls()

        # Model settings
        if "model" in data:
            config.model_name = data["model"].get("name", config.model_name)
            config.trust_remote_code = data["model"].get(
                "trust_remote_code", config.trust_remote_code
            )
            config.torch_dtype = data["model"].get("torch_dtype", config.torch_dtype)

        # LoRA settings
        if "lora" in data:
            config.lora = LoRAConfig(**data["lora"])

        # Quantization settings
        if "quantization" in data:
            config.quantization = QuantizationConfig(**data["quantization"])

        # Training settings
        if "training" in data:
            config.training = TrainingConfig(**data["training"])

        # Dataset
        if "dataset" in data:
            config.dataset_dir = data["dataset"].get(
                "mixed_dataset_dir", config.dataset_dir
            )

        # Output
        if "output" in data:
            config.output_dir = data["output"].get("dir", config.output_dir)
            config.logging_dir = data["output"].get("logging_dir", config.logging_dir)

        return config


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def load_model_and_tokenizer(config: FinetuneConfig):
    """
    Load model with 4-bit quantization and tokenizer.

    Args:
        config: Fine-tuning configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading model: {config.model_name}")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_compute_dtype=get_torch_dtype(config.quantization.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
        torch_dtype=get_torch_dtype(config.torch_dtype),
    )

    # Enable gradient checkpointing
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Model loaded: {model.num_parameters():,} parameters")

    return model, tokenizer


def prepare_model_for_training(model, config: FinetuneConfig):
    """
    Add LoRA adapters to the model.

    Args:
        model: Base model
        config: Fine-tuning configuration

    Returns:
        Model with LoRA adapters
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        task_type=config.lora.task_type,
        bias="none",
    )

    # Add adapters
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model


def load_dataset(config: FinetuneConfig):
    """
    Load training and validation datasets.

    Args:
        config: Fine-tuning configuration

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    from datasets import load_from_disk

    train_path = os.path.join(config.dataset_dir, "train")
    val_path = os.path.join(config.dataset_dir, "val")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training dataset not found at {train_path}. "
            "Run dataset preparation first."
        )

    train_dataset = load_from_disk(train_path)
    eval_dataset = None

    if os.path.exists(val_path):
        eval_dataset = load_from_disk(val_path)
        print(f"Loaded datasets: train={len(train_dataset)}, val={len(eval_dataset)}")
    else:
        print(f"Loaded training dataset: {len(train_dataset)} examples")

    return train_dataset, eval_dataset


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: FinetuneConfig,
):
    """
    Create the SFT trainer.

    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Validation dataset
        config: Fine-tuning configuration

    Returns:
        SFTTrainer instance
    """
    from transformers import TrainingArguments
    from trl import SFTTrainer

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        optim=config.training.optim,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end if eval_dataset else False,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        logging_dir=config.logging_dir,
        report_to=["tensorboard"],
        gradient_checkpointing=config.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=config.training.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    return trainer


def save_training_log(config: FinetuneConfig, trainer, metrics: dict):
    """
    Save training log with configuration and metrics.

    Args:
        config: Fine-tuning configuration
        trainer: Trained trainer
        metrics: Training metrics
    """
    log_path = os.path.join(config.output_dir, "training_log.yaml")

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "model": config.model_name,
        "lora": {
            "r": config.lora.r,
            "alpha": config.lora.alpha,
            "dropout": config.lora.dropout,
        },
        "training": {
            "epochs": config.training.num_train_epochs,
            "learning_rate": config.training.learning_rate,
            "batch_size": config.training.per_device_train_batch_size,
            "gradient_accumulation": config.training.gradient_accumulation_steps,
        },
        "metrics": metrics,
    }

    with open(log_path, "w") as f:
        yaml.dump(log_data, f, default_flow_style=False)

    print(f"Training log saved to: {log_path}")


def run_training(config: FinetuneConfig) -> dict:
    """
    Run the complete fine-tuning pipeline.

    Args:
        config: Fine-tuning configuration

    Returns:
        Training metrics
    """
    print("=" * 60)
    print("QUASAR Fine-Tuning")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")

    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print()

    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Add LoRA adapters
    model = prepare_model_for_training(model, config)

    # Load datasets
    train_dataset, eval_dataset = load_dataset(config)

    # Create trainer
    trainer = create_trainer(
        model, tokenizer, train_dataset, eval_dataset, config
    )

    # Train
    print("\nStarting training...")
    print(f"Output: {config.output_dir}")
    print()

    train_result = trainer.train(
        resume_from_checkpoint=config.resume_from_checkpoint
    )

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    # Get metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    if eval_dataset:
        metrics["eval_samples"] = len(eval_dataset)

    # Save training log
    save_training_log(config, trainer, metrics)

    print("\nTraining complete!")
    print(f"Final loss: {metrics.get('train_loss', 'N/A')}")
    print(f"Model saved to: {config.output_dir}")

    return metrics


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune QUASAR model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Override dataset directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without training",
    )

    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        config = FinetuneConfig.from_yaml(args.config)
    else:
        print(f"Config not found at {args.config}, using defaults")
        config = FinetuneConfig()

    # Apply overrides
    if args.resume:
        config.resume_from_checkpoint = args.resume
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.dataset_dir:
        config.dataset_dir = args.dataset_dir

    # Dry run
    if args.dry_run:
        print("Configuration:")
        print(f"  Model: {config.model_name}")
        print(f"  LoRA r={config.lora.r}, alpha={config.lora.alpha}")
        print(f"  Learning rate: {config.training.learning_rate}")
        print(f"  Epochs: {config.training.num_train_epochs}")
        print(f"  Batch size: {config.training.per_device_train_batch_size}")
        print(f"  Gradient accum: {config.training.gradient_accumulation_steps}")
        print(f"  Dataset: {config.dataset_dir}")
        print(f"  Output: {config.output_dir}")
        return

    # Run training
    metrics = run_training(config)

    return metrics


if __name__ == "__main__":
    main()
