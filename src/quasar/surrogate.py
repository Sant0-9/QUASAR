"""
Surrogate model for fast circuit quality prediction.

Predicts VQE energy error from circuit and Hamiltonian encodings,
enabling 100x faster circuit exploration by filtering candidates
before expensive VQE runs.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from src.quasar.circuit_encoder import CircuitEncoder, CircuitEncoderConfig


class Trainability(str, Enum):
    """Circuit trainability classification."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SurrogateConfig:
    """Configuration for surrogate model."""

    # Encoder dimensions
    circuit_embedding_dim: int = 256
    hamiltonian_embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 3

    # Hamiltonian encoder
    max_qubits: int = 12
    max_terms: int = 100

    # Training
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Trainability thresholds
    high_trainability_threshold: float = 0.3
    low_trainability_threshold: float = 0.7


@dataclass
class SurrogatePrediction:
    """Prediction result from surrogate model."""

    predicted_error: float  # Predicted energy error
    trainability: Trainability  # High/Medium/Low classification
    confidence: float  # Model confidence (0-1)
    inference_time_ms: float = 0.0  # Time taken for prediction


@dataclass
class SurrogateTrainingExample:
    """Single training example for surrogate model."""

    circuit: QuantumCircuit
    hamiltonian: SparsePauliOp
    energy_error: float
    converged: bool = True
    metadata: dict = field(default_factory=dict)


class HamiltonianEncoder:
    """
    Encodes Hamiltonians (SparsePauliOp) into fixed-size vectors.

    Extracts features including:
    - Term structure (Pauli types and coefficients)
    - Interaction graph (qubit connectivity)
    - Spectral properties
    """

    def __init__(self, config: SurrogateConfig | None = None):
        """
        Initialize the Hamiltonian encoder.

        Args:
            config: Surrogate configuration
        """
        self.config = config or SurrogateConfig()
        self._setup_dimensions()

    def _setup_dimensions(self):
        """Calculate feature dimensions."""
        # Pauli type counts per qubit (I, X, Y, Z per qubit)
        self.pauli_dim = self.config.max_qubits * 4

        # Global features
        self.global_dim = 10

        # Interaction features (adjacency)
        self.interaction_dim = self.config.max_qubits * (self.config.max_qubits - 1) // 2

        # Coefficient statistics
        self.coeff_dim = 10

        self.raw_dim = self.pauli_dim + self.global_dim + self.interaction_dim + self.coeff_dim

    def encode(self, hamiltonian: SparsePauliOp) -> np.ndarray:
        """
        Encode a Hamiltonian into a fixed-size vector.

        Args:
            hamiltonian: SparsePauliOp to encode

        Returns:
            Dense embedding vector of shape (hamiltonian_embedding_dim,)
        """
        num_qubits = hamiltonian.num_qubits
        paulis = hamiltonian.paulis
        coeffs = np.array(hamiltonian.coeffs)

        features = []

        # 1. Pauli type counts per qubit
        pauli_counts = np.zeros((self.config.max_qubits, 4), dtype=np.float32)
        for pauli in paulis:
            pauli_str = str(pauli)
            for q, p in enumerate(reversed(pauli_str)):  # Qiskit uses little-endian
                if q >= self.config.max_qubits:
                    break
                idx = {"I": 0, "X": 1, "Y": 2, "Z": 3}.get(p, 0)
                pauli_counts[q, idx] += 1

        # Normalize by number of terms
        num_terms = len(paulis)
        if num_terms > 0:
            pauli_counts /= num_terms
        features.extend(pauli_counts.flatten())

        # 2. Global features
        global_feats = [
            num_qubits / self.config.max_qubits,
            num_terms / self.config.max_terms,
            np.log1p(num_terms) / np.log1p(self.config.max_terms),
        ]

        # Count different Pauli types
        x_count = sum(1 for p in paulis if "X" in str(p))
        y_count = sum(1 for p in paulis if "Y" in str(p))
        z_count = sum(1 for p in paulis if "Z" in str(p))
        identity_count = sum(1 for p in paulis if str(p) == "I" * num_qubits)

        global_feats.extend([
            x_count / max(1, num_terms),
            y_count / max(1, num_terms),
            z_count / max(1, num_terms),
            identity_count / max(1, num_terms),
        ])

        # Locality (average number of non-identity Paulis per term)
        locality = np.mean([sum(1 for c in str(p) if c != "I") for p in paulis]) if num_terms > 0 else 0
        global_feats.append(locality / num_qubits)

        # Two-body vs many-body ratio
        two_body = sum(1 for p in paulis if sum(1 for c in str(p) if c != "I") == 2)
        global_feats.append(two_body / max(1, num_terms))

        # Pad global features
        while len(global_feats) < self.global_dim:
            global_feats.append(0.0)
        features.extend(global_feats[:self.global_dim])

        # 3. Interaction features (which qubits interact)
        interaction = np.zeros((self.config.max_qubits, self.config.max_qubits), dtype=np.float32)
        for pauli in paulis:
            pauli_str = str(pauli)
            active_qubits = [q for q, p in enumerate(reversed(pauli_str)) if p != "I"]
            for i in range(len(active_qubits)):
                for j in range(i + 1, len(active_qubits)):
                    q1, q2 = active_qubits[i], active_qubits[j]
                    if q1 < self.config.max_qubits and q2 < self.config.max_qubits:
                        interaction[q1, q2] += 1
                        interaction[q2, q1] += 1

        # Normalize and flatten upper triangle
        if num_terms > 0:
            interaction /= num_terms
        for i in range(self.config.max_qubits):
            for j in range(i + 1, self.config.max_qubits):
                features.append(interaction[i, j])

        # 4. Coefficient statistics
        coeff_real = np.real(coeffs)
        coeff_imag = np.imag(coeffs)

        coeff_feats = [
            np.mean(np.abs(coeff_real)),
            np.std(np.abs(coeff_real)),
            np.min(coeff_real) if len(coeff_real) > 0 else 0,
            np.max(coeff_real) if len(coeff_real) > 0 else 0,
            np.mean(np.abs(coeff_imag)),
            np.std(np.abs(coeff_imag)),
            float(np.sum(np.abs(coeff_imag) > 1e-10) > 0),  # Has imaginary parts
            np.sum(np.abs(coeffs)),  # Total weight
            np.max(np.abs(coeffs)) / (np.sum(np.abs(coeffs)) + 1e-10),  # Dominance
            len(set(np.round(np.abs(coeff_real), 6))),  # Unique coefficient count
        ]
        features.extend(coeff_feats[:self.coeff_dim])

        # Convert and pad/truncate to embedding dim
        embedding = np.array(features, dtype=np.float32)
        target_dim = self.config.hamiltonian_embedding_dim

        if len(embedding) < target_dim:
            embedding = np.pad(embedding, (0, target_dim - len(embedding)))
        elif len(embedding) > target_dim:
            embedding = embedding[:target_dim]

        return embedding

    def encode_batch(self, hamiltonians: list[SparsePauliOp]) -> np.ndarray:
        """
        Encode a batch of Hamiltonians.

        Args:
            hamiltonians: List of SparsePauliOp

        Returns:
            Embeddings array of shape (batch_size, hamiltonian_embedding_dim)
        """
        return np.stack([self.encode(h) for h in hamiltonians])


class SurrogateModel(nn.Module):
    """
    Neural network surrogate for VQE energy prediction.

    Takes circuit and Hamiltonian encodings and predicts:
    - Energy error (regression)
    - Trainability (classification)
    - Confidence (0-1)
    """

    def __init__(self, config: SurrogateConfig | None = None):
        """
        Initialize the surrogate model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config or SurrogateConfig()

        # Input dimension
        input_dim = self.config.circuit_embedding_dim + self.config.hamiltonian_embedding_dim

        # Shared encoder
        layers = []
        current_dim = input_dim

        for i in range(self.config.num_layers):
            layers.extend([
                nn.Linear(current_dim, self.config.hidden_dim),
                nn.LayerNorm(self.config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
            ])
            current_dim = self.config.hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Prediction heads
        self.energy_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.trainability_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3 classes: high, medium, low
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        circuit_embedding: torch.Tensor,
        hamiltonian_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            circuit_embedding: Circuit encodings (batch, circuit_dim)
            hamiltonian_embedding: Hamiltonian encodings (batch, ham_dim)

        Returns:
            Tuple of (energy_error, trainability_logits, confidence)
        """
        # Concatenate embeddings
        x = torch.cat([circuit_embedding, hamiltonian_embedding], dim=-1)

        # Shared encoding
        h = self.encoder(x)

        # Predictions
        energy = self.energy_head(h).squeeze(-1)
        trainability = self.trainability_head(h)
        confidence = self.confidence_head(h).squeeze(-1)

        return energy, trainability, confidence

    def predict_energy(
        self,
        circuit_embedding: torch.Tensor,
        hamiltonian_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict energy error only."""
        energy, _, _ = self.forward(circuit_embedding, hamiltonian_embedding)
        return energy


class SurrogateEvaluator:
    """
    High-level interface for surrogate-based circuit evaluation.

    Combines encoders and model for end-to-end prediction.
    """

    def __init__(
        self,
        config: SurrogateConfig | None = None,
        model_path: str | Path | None = None,
        device: str | None = None,
    ):
        """
        Initialize the surrogate evaluator.

        Args:
            config: Surrogate configuration
            model_path: Path to saved model weights
            device: Device to use (cpu/cuda)
        """
        self.config = config or SurrogateConfig()

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize encoders
        circuit_config = CircuitEncoderConfig(
            embedding_dim=self.config.circuit_embedding_dim,
            max_qubits=self.config.max_qubits,
        )
        self.circuit_encoder = CircuitEncoder(circuit_config)
        self.hamiltonian_encoder = HamiltonianEncoder(self.config)

        # Initialize model
        self.model = SurrogateModel(self.config)
        self.model.to(self.device)
        self.model.eval()

        # Load weights if provided
        if model_path is not None:
            self.load(model_path)

        # Training data buffer for active learning
        self.training_buffer: list[SurrogateTrainingExample] = []

    def predict(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
    ) -> SurrogatePrediction:
        """
        Predict circuit quality.

        Args:
            circuit: Quantum circuit
            hamiltonian: Target Hamiltonian

        Returns:
            SurrogatePrediction with energy error, trainability, confidence
        """
        start_time = time.time()

        # Encode inputs
        circuit_emb = self.circuit_encoder.encode(circuit)
        ham_emb = self.hamiltonian_encoder.encode(hamiltonian)

        # Convert to tensors
        circuit_tensor = torch.from_numpy(circuit_emb).unsqueeze(0).to(self.device)
        ham_tensor = torch.from_numpy(ham_emb).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            energy, trainability_logits, confidence = self.model(circuit_tensor, ham_tensor)

        # Convert outputs
        predicted_error = float(energy.item())
        confidence_val = float(confidence.item())

        # Classify trainability
        trainability_probs = F.softmax(trainability_logits, dim=-1)
        trainability_idx = int(torch.argmax(trainability_probs, dim=-1).item())
        trainability = [Trainability.HIGH, Trainability.MEDIUM, Trainability.LOW][trainability_idx]

        inference_time = (time.time() - start_time) * 1000

        return SurrogatePrediction(
            predicted_error=predicted_error,
            trainability=trainability,
            confidence=confidence_val,
            inference_time_ms=inference_time,
        )

    def predict_batch(
        self,
        circuits: list[QuantumCircuit],
        hamiltonian: SparsePauliOp,
    ) -> list[SurrogatePrediction]:
        """
        Predict quality for multiple circuits.

        Args:
            circuits: List of quantum circuits
            hamiltonian: Target Hamiltonian (same for all)

        Returns:
            List of SurrogatePrediction
        """
        start_time = time.time()

        # Encode circuits
        circuit_embs = self.circuit_encoder.encode_batch(circuits)
        ham_emb = self.hamiltonian_encoder.encode(hamiltonian)

        # Repeat Hamiltonian embedding for batch
        ham_embs = np.tile(ham_emb, (len(circuits), 1))

        # Convert to tensors
        circuit_tensor = torch.from_numpy(circuit_embs).to(self.device)
        ham_tensor = torch.from_numpy(ham_embs).to(self.device)

        # Predict
        with torch.no_grad():
            energy, trainability_logits, confidence = self.model(circuit_tensor, ham_tensor)

        # Convert outputs
        predictions = []
        trainability_probs = F.softmax(trainability_logits, dim=-1)
        trainability_indices = torch.argmax(trainability_probs, dim=-1)

        total_time = (time.time() - start_time) * 1000
        per_circuit_time = total_time / len(circuits)

        for i in range(len(circuits)):
            trainability_idx = int(trainability_indices[i].item())
            trainability = [Trainability.HIGH, Trainability.MEDIUM, Trainability.LOW][trainability_idx]

            predictions.append(SurrogatePrediction(
                predicted_error=float(energy[i].item()),
                trainability=trainability,
                confidence=float(confidence[i].item()),
                inference_time_ms=per_circuit_time,
            ))

        return predictions

    def score_circuits(
        self,
        circuits: list[QuantumCircuit],
        hamiltonian: SparsePauliOp,
    ) -> list[tuple[int, float]]:
        """
        Score and rank circuits by predicted quality.

        Args:
            circuits: List of quantum circuits
            hamiltonian: Target Hamiltonian

        Returns:
            List of (circuit_index, score) sorted by score (lower is better)
        """
        predictions = self.predict_batch(circuits, hamiltonian)

        # Score = predicted_error (lower is better)
        # Adjust by confidence: lower confidence -> worse score
        scores = []
        for i, pred in enumerate(predictions):
            # Penalize low confidence predictions
            adjusted_score = pred.predicted_error / (pred.confidence + 0.1)
            scores.append((i, adjusted_score))

        # Sort by score (lower is better)
        scores.sort(key=lambda x: x[1])
        return scores

    def select_top_k(
        self,
        circuits: list[QuantumCircuit],
        hamiltonian: SparsePauliOp,
        k: int = 10,
        confidence_threshold: float = 0.5,
    ) -> list[int]:
        """
        Select top-K circuits for VQE evaluation.

        Args:
            circuits: List of candidate circuits
            hamiltonian: Target Hamiltonian
            k: Number of circuits to select
            confidence_threshold: Minimum confidence to consider

        Returns:
            Indices of selected circuits
        """
        predictions = self.predict_batch(circuits, hamiltonian)

        # Filter by confidence and trainability
        valid_indices = []
        for i, pred in enumerate(predictions):
            if pred.confidence >= confidence_threshold:
                if pred.trainability != Trainability.LOW:
                    valid_indices.append((i, pred.predicted_error, pred.confidence))

        # Sort by predicted error
        valid_indices.sort(key=lambda x: x[1])

        # Select top k
        selected = [idx for idx, _, _ in valid_indices[:k]]

        # If not enough, add remaining by score regardless of trainability
        if len(selected) < k:
            remaining = [
                (i, pred.predicted_error)
                for i, pred in enumerate(predictions)
                if i not in selected and pred.confidence >= confidence_threshold
            ]
            remaining.sort(key=lambda x: x[1])
            for idx, _ in remaining:
                if len(selected) >= k:
                    break
                selected.append(idx)

        return selected

    def add_training_example(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        energy_error: float,
        converged: bool = True,
    ):
        """
        Add a training example for active learning.

        Args:
            circuit: The circuit that was evaluated
            hamiltonian: The Hamiltonian used
            energy_error: Actual energy error from VQE
            converged: Whether VQE converged
        """
        example = SurrogateTrainingExample(
            circuit=circuit,
            hamiltonian=hamiltonian,
            energy_error=energy_error,
            converged=converged,
        )
        self.training_buffer.append(example)

    def update_from_buffer(self, epochs: int = 10, batch_size: int = 32):
        """
        Update model with accumulated training examples.

        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        if len(self.training_buffer) < batch_size:
            return  # Not enough examples

        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Prepare data
        circuit_embs = []
        ham_embs = []
        targets = []

        for example in self.training_buffer:
            circuit_embs.append(self.circuit_encoder.encode(example.circuit))
            ham_embs.append(self.hamiltonian_encoder.encode(example.hamiltonian))
            targets.append(example.energy_error)

        circuit_tensor = torch.from_numpy(np.stack(circuit_embs)).to(self.device)
        ham_tensor = torch.from_numpy(np.stack(ham_embs)).to(self.device)
        target_tensor = torch.tensor(targets, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(circuit_tensor, ham_tensor, target_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for c_batch, h_batch, t_batch in loader:
                optimizer.zero_grad()
                energy, _, _ = self.model(c_batch, h_batch)
                loss = F.mse_loss(energy, t_batch)
                loss.backward()
                optimizer.step()

        self.model.eval()
        self.training_buffer.clear()

    def save(self, path: str | Path):
        """
        Save model weights.

        Args:
            path: Path to save weights
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }, path)

    def load(self, path: str | Path):
        """
        Load model weights.

        Args:
            path: Path to weights file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()


class SurrogateTrainer:
    """
    Trainer for the surrogate model.

    Handles bootstrap data generation and model training.
    """

    def __init__(
        self,
        config: SurrogateConfig | None = None,
        device: str | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            config: Surrogate configuration
            device: Device to use
        """
        self.config = config or SurrogateConfig()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize components
        circuit_config = CircuitEncoderConfig(
            embedding_dim=self.config.circuit_embedding_dim,
            max_qubits=self.config.max_qubits,
        )
        self.circuit_encoder = CircuitEncoder(circuit_config)
        self.hamiltonian_encoder = HamiltonianEncoder(self.config)
        self.model = SurrogateModel(self.config).to(self.device)

    def prepare_dataset(
        self,
        examples: list[SurrogateTrainingExample],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare dataset from training examples.

        Args:
            examples: List of training examples

        Returns:
            Tuple of (circuit_embeddings, ham_embeddings, energy_targets, trainability_targets)
        """
        circuit_embs = []
        ham_embs = []
        energy_targets = []
        trainability_targets = []

        for example in examples:
            circuit_embs.append(self.circuit_encoder.encode(example.circuit))
            ham_embs.append(self.hamiltonian_encoder.encode(example.hamiltonian))
            energy_targets.append(example.energy_error)

            # Classify trainability based on energy error
            if example.energy_error < self.config.high_trainability_threshold:
                trainability_targets.append(0)  # HIGH
            elif example.energy_error < self.config.low_trainability_threshold:
                trainability_targets.append(1)  # MEDIUM
            else:
                trainability_targets.append(2)  # LOW

        return (
            torch.from_numpy(np.stack(circuit_embs)),
            torch.from_numpy(np.stack(ham_embs)),
            torch.tensor(energy_targets, dtype=torch.float32),
            torch.tensor(trainability_targets, dtype=torch.long),
        )

    def train(
        self,
        train_examples: list[SurrogateTrainingExample],
        val_examples: list[SurrogateTrainingExample] | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
    ) -> dict:
        """
        Train the surrogate model.

        Args:
            train_examples: Training examples
            val_examples: Validation examples
            epochs: Number of epochs
            batch_size: Batch size
            patience: Early stopping patience

        Returns:
            Training history
        """
        # Prepare data
        train_data = self.prepare_dataset(train_examples)
        train_dataset = torch.utils.data.TensorDataset(*train_data)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        if val_examples:
            val_data = self.prepare_dataset(val_examples)
            val_dataset = torch.utils.data.TensorDataset(*val_data)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        else:
            val_loader = None

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_r2": []}
        best_val_loss = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0

            for c_batch, h_batch, e_batch, t_batch in train_loader:
                c_batch = c_batch.to(self.device)
                h_batch = h_batch.to(self.device)
                e_batch = e_batch.to(self.device)
                t_batch = t_batch.to(self.device)

                optimizer.zero_grad()
                energy, trainability_logits, confidence = self.model(c_batch, h_batch)

                # Combined loss
                energy_loss = F.mse_loss(energy, e_batch)
                trainability_loss = F.cross_entropy(trainability_logits, t_batch)
                loss = energy_loss + 0.1 * trainability_loss

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validate
            if val_loader:
                val_loss, val_r2 = self._validate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_r2"].append(val_r2)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

        return history

    def _validate(
        self,
        val_loader: torch.utils.data.DataLoader,
    ) -> tuple[float, float]:
        """Run validation and return loss and R^2."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for c_batch, h_batch, e_batch, t_batch in val_loader:
                c_batch = c_batch.to(self.device)
                h_batch = h_batch.to(self.device)
                e_batch = e_batch.to(self.device)
                t_batch = t_batch.to(self.device)

                energy, trainability_logits, _ = self.model(c_batch, h_batch)
                energy_loss = F.mse_loss(energy, e_batch)
                trainability_loss = F.cross_entropy(trainability_logits, t_batch)
                loss = energy_loss + 0.1 * trainability_loss

                total_loss += loss.item()
                all_preds.extend(energy.cpu().numpy())
                all_targets.extend(e_batch.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        # Calculate R^2
        preds = np.array(all_preds)
        targets = np.array(all_targets)
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        return avg_loss, r2

    def get_evaluator(self) -> SurrogateEvaluator:
        """
        Get an evaluator with the trained model.

        Returns:
            SurrogateEvaluator with trained weights
        """
        evaluator = SurrogateEvaluator(self.config, device=str(self.device))
        evaluator.model.load_state_dict(self.model.state_dict())
        return evaluator


def create_surrogate_evaluator(
    model_path: str | Path | None = None,
    config: SurrogateConfig | None = None,
) -> SurrogateEvaluator:
    """
    Convenience function to create a surrogate evaluator.

    Args:
        model_path: Optional path to saved model
        config: Optional configuration

    Returns:
        SurrogateEvaluator ready for predictions
    """
    return SurrogateEvaluator(config=config, model_path=model_path)
