#!/bin/bash
# QUASAR Fine-tuning Setup Script for Brev GPU Instance
# Run this script on a fresh Brev A100 instance

set -e  # Exit on error

echo "=========================================="
echo "QUASAR Fine-tuning Setup"
echo "=========================================="

# Configuration
REPO_URL="https://github.com/Sant0-9/quantum-mind.git"
PROJECT_DIR="quantum-mind"
VENV_DIR="venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Clone repo
echo ""
log_info "Step 1/6: Cloning repository..."
if [ -d "$PROJECT_DIR" ]; then
    log_warn "Directory $PROJECT_DIR exists, pulling latest..."
    cd "$PROJECT_DIR"
    git pull
else
    git clone "$REPO_URL"
    cd "$PROJECT_DIR"
fi

# Step 2: Create virtual environment
echo ""
log_info "Step 2/6: Setting up Python environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Step 3: Install dependencies
echo ""
log_info "Step 3/6: Installing dependencies..."
pip install --upgrade pip

# Core ML dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Training dependencies
pip install transformers datasets peft trl accelerate bitsandbytes

# Additional dependencies
pip install scipy h5py pyyaml qiskit qiskit-aer the-well

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Step 4: Generate physics-augmented dataset from The Well
echo ""
log_info "Step 4/6: Generating physics-augmented dataset from The Well..."

python << 'PYTHON_SCRIPT'
import os
import sys

print("Generating physics-augmented data from The Well simulations...")

try:
    from src.quasar.well_loader import WellLoader, WellConfig
    from src.quasar.augmented_data import AugmentedDataGenerator, AugmentedDataConfig
    import random

    # Config for Well data (streaming from HuggingFace)
    well_config = WellConfig(
        dataset_name='shear_flow',
        base_path='hf://datasets/polymathic-ai/',
        split='train',
        batch_size=1,
        restrict_samples=200,  # Use 200 real trajectories
    )

    aug_config = AugmentedDataConfig(
        output_dir='data/physics_augmented',
        target_examples=5000,
    )

    print("Loading Well data (streaming from HuggingFace)...")
    loader = WellLoader(well_config)
    print(f"Loaded {len(loader)} trajectories")

    print("Generating physics-augmented examples...")
    generator = AugmentedDataGenerator(aug_config)

    all_examples = []
    for i, batch in enumerate(loader):
        traj = batch.input_fields[0, :, :, :, 0].numpy()
        examples = generator.generate_from_trajectories([traj], dt=0.01, dx=1.0)
        all_examples.extend(examples)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1} trajectories, {len(all_examples)} examples")

    print(f"Generated {len(all_examples)} examples from real Well data")

    # Generate additional synthetic examples
    remaining = 5000 - len(all_examples)
    if remaining > 0:
        print(f"Generating {remaining} additional synthetic examples...")
        synthetic = generator.generate_batch(remaining)
        all_examples.extend(synthetic)

    # Split and save
    random.shuffle(all_examples)
    n_val = int(len(all_examples) * 0.1)
    train = all_examples[n_val:]
    val = all_examples[:n_val]

    stats = generator.save_dataset(train, val)
    print(f"Saved: {stats['train_count']} train, {stats['val_count']} val")

except Exception as e:
    print(f"Error generating Well data: {e}")
    print("Falling back to synthetic-only generation...")

    from src.quasar.augmented_data import AugmentedDataGenerator, AugmentedDataConfig

    aug_config = AugmentedDataConfig(
        output_dir='data/physics_augmented',
        target_examples=5000,
    )

    generator = AugmentedDataGenerator(aug_config)
    stats = generator.generate_and_save()
    print(f"Generated {stats['total_count']} synthetic examples")

print("Physics-augmented data generation complete!")
PYTHON_SCRIPT

# Step 5: Prepare mixed dataset
echo ""
log_info "Step 5/6: Preparing mixed training dataset..."

python -m src.training.dataset --prepare || log_warn "Prepare step had issues, continuing..."
python -m src.training.dataset --mix

# Verify dataset
python << 'VERIFY_SCRIPT'
from datasets import load_from_disk
import os

train_path = "data/mixed/train"
val_path = "data/mixed/val"

if not os.path.exists(train_path):
    raise FileNotFoundError(f"Training dataset not found at {train_path}")

train = load_from_disk(train_path)
val = load_from_disk(val_path) if os.path.exists(val_path) else None

print(f"Dataset ready:")
print(f"  Train: {len(train)} examples")
if val:
    print(f"  Val: {len(val)} examples")
print(f"  Columns: {train.column_names}")
print(f"  Sample text length: {len(train[0]['text'])} chars")
VERIFY_SCRIPT

# Step 6: Run fine-tuning
echo ""
log_info "Step 6/6: Starting fine-tuning..."
echo ""
echo "=========================================="
echo "Fine-tuning Qwen2.5-Coder-7B-Instruct"
echo "Method: QLoRA (4-bit)"
echo "Expected time: 2-4 hours"
echo "=========================================="
echo ""

python -m src.training.finetune

echo ""
echo "=========================================="
echo "FINE-TUNING COMPLETE!"
echo "=========================================="
echo ""
echo "Model saved to: models/checkpoints/quasar-v2/"
echo ""
echo "To validate the model, run:"
echo "  python -m src.training.validate_model"
echo ""
