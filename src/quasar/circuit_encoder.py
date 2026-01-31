"""
Circuit encoder for quantum circuits.

Converts quantum circuits into fixed-size dense vectors for the
surrogate model. Encodes gate types, circuit structure, entanglement
patterns, and connectivity information.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit


# Gate categories for encoding
SINGLE_QUBIT_GATES = frozenset({
    "x", "y", "z", "h", "s", "sdg", "t", "tdg",
    "rx", "ry", "rz", "r", "u", "u1", "u2", "u3", "p",
})

TWO_QUBIT_GATES = frozenset({
    "cx", "cy", "cz", "swap", "iswap", "ecr",
    "crx", "cry", "crz", "cp", "rxx", "ryy", "rzz", "rzx",
})

ROTATION_GATES = frozenset({
    "rx", "ry", "rz", "r", "u", "u1", "u2", "u3", "p",
    "crx", "cry", "crz", "cp", "rxx", "ryy", "rzz", "rzx",
})

# Ordered gate list for one-hot encoding
GATE_TYPES = [
    # Single-qubit non-parametric
    "x", "y", "z", "h", "s", "sdg", "t", "tdg",
    # Single-qubit rotations
    "rx", "ry", "rz", "u", "u3", "p",
    # Two-qubit gates
    "cx", "cy", "cz", "swap", "iswap",
    "crx", "cry", "crz",
    "rxx", "ryy", "rzz",
    # Other
    "other",
]

GATE_TO_IDX = {gate: idx for idx, gate in enumerate(GATE_TYPES)}


@dataclass
class CircuitEncoderConfig:
    """Configuration for circuit encoder."""

    embedding_dim: int = 256  # Output embedding dimension
    max_qubits: int = 12  # Maximum qubits to support
    max_depth: int = 100  # Maximum circuit depth
    max_gates: int = 500  # Maximum total gates
    use_learned_encoder: bool = False  # Use neural network for encoding


@dataclass
class CircuitFeatures:
    """Raw features extracted from a circuit."""

    num_qubits: int
    depth: int
    num_params: int
    total_gates: int
    single_qubit_gates: int
    two_qubit_gates: int
    gate_counts: dict[str, int]
    entanglement_density: float  # Two-qubit gates / (n*(n-1)/2)
    qubit_degrees: list[int]  # Number of two-qubit gates per qubit
    adjacency: np.ndarray  # Qubit connectivity matrix
    layer_structure: list[dict]  # Gate types per layer
    param_per_qubit: list[int]  # Parameters per qubit


class CircuitEncoder:
    """
    Encodes quantum circuits into fixed-size dense vectors.

    The encoder extracts structural features from the circuit including:
    - Gate type distribution
    - Circuit depth and size
    - Parameter count and distribution
    - Entanglement pattern and connectivity
    - Layer-wise structure

    Example:
        >>> encoder = CircuitEncoder()
        >>> circuit = QuantumCircuit(4)
        >>> circuit.h(0)
        >>> circuit.cx(0, 1)
        >>> embedding = encoder.encode(circuit)
        >>> print(embedding.shape)
        (256,)
    """

    def __init__(self, config: CircuitEncoderConfig | None = None):
        """
        Initialize the circuit encoder.

        Args:
            config: Encoder configuration
        """
        self.config = config or CircuitEncoderConfig()
        self._setup_dimensions()

    def _setup_dimensions(self):
        """Calculate feature dimensions for embedding."""
        # Gate type histogram: len(GATE_TYPES) features
        self.gate_hist_dim = len(GATE_TYPES)

        # Global features: depth, params, gates, density, etc.
        self.global_dim = 10

        # Per-qubit features: degree, param count
        self.per_qubit_dim = self.config.max_qubits * 2

        # Adjacency features: flattened upper triangle
        self.adj_dim = self.config.max_qubits * (self.config.max_qubits - 1) // 2

        # Layer structure: simplified representation
        self.layer_dim = 20

        # Total raw features
        self.raw_dim = (
            self.gate_hist_dim
            + self.global_dim
            + self.per_qubit_dim
            + self.adj_dim
            + self.layer_dim
        )

    def extract_features(self, circuit: QuantumCircuit) -> CircuitFeatures:
        """
        Extract raw features from a circuit.

        Args:
            circuit: Quantum circuit to analyze

        Returns:
            CircuitFeatures with extracted information
        """
        num_qubits = circuit.num_qubits
        depth = circuit.depth()
        num_params = len(circuit.parameters)

        # Count gates
        ops = circuit.count_ops()
        total_gates = sum(v for k, v in ops.items() if k not in ["barrier", "measure"])
        single_qubit = sum(ops.get(g, 0) for g in SINGLE_QUBIT_GATES)
        two_qubit = sum(ops.get(g, 0) for g in TWO_QUBIT_GATES)

        # Entanglement density
        max_pairs = num_qubits * (num_qubits - 1) / 2
        entanglement_density = two_qubit / max_pairs if max_pairs > 0 else 0.0

        # Per-qubit analysis
        qubit_degrees = [0] * num_qubits
        param_per_qubit = [0] * num_qubits
        adjacency = np.zeros((num_qubits, num_qubits), dtype=np.float32)

        for instruction in circuit.data:
            gate_name = instruction.operation.name
            qubits = [circuit.qubits.index(q) for q in instruction.qubits]

            # Count parameters per qubit
            if len(instruction.operation.params) > 0:
                for q in qubits:
                    param_per_qubit[q] += len(instruction.operation.params)

            # Track connectivity for two-qubit gates
            if len(qubits) == 2:
                q1, q2 = qubits
                qubit_degrees[q1] += 1
                qubit_degrees[q2] += 1
                adjacency[q1, q2] += 1
                adjacency[q2, q1] += 1

        # Layer structure (simplified)
        layer_structure = self._analyze_layers(circuit)

        return CircuitFeatures(
            num_qubits=num_qubits,
            depth=depth,
            num_params=num_params,
            total_gates=total_gates,
            single_qubit_gates=single_qubit,
            two_qubit_gates=two_qubit,
            gate_counts=dict(ops),
            entanglement_density=entanglement_density,
            qubit_degrees=qubit_degrees,
            adjacency=adjacency,
            layer_structure=layer_structure,
            param_per_qubit=param_per_qubit,
        )

    def _analyze_layers(self, circuit: QuantumCircuit) -> list[dict]:
        """
        Analyze circuit layer by layer.

        Returns list of dicts with gate counts per layer.
        """
        # Use Qiskit's depth analysis
        layers = []
        try:
            # Get layers from circuit
            dag = None
            try:
                from qiskit.converters import circuit_to_dag
                dag = circuit_to_dag(circuit)
            except ImportError:
                pass

            if dag is not None:
                for layer in dag.layers():
                    layer_ops = {}
                    for node in layer["graph"].op_nodes():
                        gate = node.op.name
                        layer_ops[gate] = layer_ops.get(gate, 0) + 1
                    layers.append(layer_ops)
        except Exception:
            # Fallback: single layer with all gates
            layers = [dict(circuit.count_ops())]

        return layers[:20]  # Limit to first 20 layers

    def encode(self, circuit: QuantumCircuit) -> np.ndarray:
        """
        Encode a circuit into a fixed-size dense vector.

        Args:
            circuit: Quantum circuit to encode

        Returns:
            Dense embedding vector of shape (embedding_dim,)
        """
        features = self.extract_features(circuit)
        return self._features_to_embedding(features)

    def encode_raw(self, circuit: QuantumCircuit) -> np.ndarray:
        """
        Encode a circuit into raw features (before padding to embedding_dim).

        Used by LearnedCircuitEncoder to get consistent raw features.

        Args:
            circuit: Quantum circuit to encode

        Returns:
            Raw feature vector of shape (raw_dim,)
        """
        features = self.extract_features(circuit)
        return self._features_to_raw(features)

    def _features_to_raw(self, features: CircuitFeatures) -> np.ndarray:
        """Convert extracted features to raw vector without padding."""
        embedding = []

        # 1. Gate type histogram (normalized)
        gate_hist = np.zeros(len(GATE_TYPES), dtype=np.float32)
        for gate, count in features.gate_counts.items():
            if gate in GATE_TO_IDX:
                gate_hist[GATE_TO_IDX[gate]] = count
            elif gate not in ["barrier", "measure"]:
                gate_hist[GATE_TO_IDX["other"]] += count

        if features.total_gates > 0:
            gate_hist = gate_hist / features.total_gates
        embedding.extend(gate_hist)

        # 2. Global features (normalized)
        global_feats = [
            features.num_qubits / self.config.max_qubits,
            features.depth / self.config.max_depth,
            features.num_params / max(1, features.total_gates),
            features.total_gates / self.config.max_gates,
            features.single_qubit_gates / max(1, features.total_gates),
            features.two_qubit_gates / max(1, features.total_gates),
            features.entanglement_density,
            np.std(features.qubit_degrees) / (np.mean(features.qubit_degrees) + 1e-6),
            np.log1p(features.depth) / np.log1p(self.config.max_depth),
            np.log1p(features.total_gates) / np.log1p(self.config.max_gates),
        ]
        embedding.extend(global_feats)

        # 3. Per-qubit features
        degrees = features.qubit_degrees[:self.config.max_qubits]
        degrees = degrees + [0] * (self.config.max_qubits - len(degrees))
        degrees_norm = [d / max(1, sum(features.qubit_degrees)) for d in degrees]
        embedding.extend(degrees_norm)

        params_per_q = features.param_per_qubit[:self.config.max_qubits]
        params_per_q = params_per_q + [0] * (self.config.max_qubits - len(params_per_q))
        params_norm = [p / max(1, features.num_params) for p in params_per_q]
        embedding.extend(params_norm)

        # 4. Adjacency features
        adj_flat = []
        for i in range(self.config.max_qubits):
            for j in range(i + 1, self.config.max_qubits):
                if i < features.num_qubits and j < features.num_qubits:
                    val = features.adjacency[i, j] / max(1, features.two_qubit_gates)
                else:
                    val = 0.0
                adj_flat.append(val)
        embedding.extend(adj_flat)

        # 5. Layer structure features
        layer_feats = self._encode_layer_structure(features.layer_structure)
        embedding.extend(layer_feats)

        return np.array(embedding, dtype=np.float32)

    def _features_to_embedding(self, features: CircuitFeatures) -> np.ndarray:
        """Convert extracted features to dense embedding."""
        embedding = []

        # 1. Gate type histogram (normalized)
        gate_hist = np.zeros(len(GATE_TYPES), dtype=np.float32)
        for gate, count in features.gate_counts.items():
            if gate in GATE_TO_IDX:
                gate_hist[GATE_TO_IDX[gate]] = count
            elif gate not in ["barrier", "measure"]:
                gate_hist[GATE_TO_IDX["other"]] += count

        # Normalize by total gates
        if features.total_gates > 0:
            gate_hist = gate_hist / features.total_gates
        embedding.extend(gate_hist)

        # 2. Global features (normalized)
        global_feats = [
            features.num_qubits / self.config.max_qubits,
            features.depth / self.config.max_depth,
            features.num_params / max(1, features.total_gates),  # Param density
            features.total_gates / self.config.max_gates,
            features.single_qubit_gates / max(1, features.total_gates),
            features.two_qubit_gates / max(1, features.total_gates),
            features.entanglement_density,
            np.std(features.qubit_degrees) / (np.mean(features.qubit_degrees) + 1e-6),  # CV of degrees
            np.log1p(features.depth) / np.log1p(self.config.max_depth),
            np.log1p(features.total_gates) / np.log1p(self.config.max_gates),
        ]
        embedding.extend(global_feats)

        # 3. Per-qubit features (padded to max_qubits)
        degrees = features.qubit_degrees[:self.config.max_qubits]
        degrees = degrees + [0] * (self.config.max_qubits - len(degrees))
        degrees_norm = [d / max(1, sum(features.qubit_degrees)) for d in degrees]
        embedding.extend(degrees_norm)

        params_per_q = features.param_per_qubit[:self.config.max_qubits]
        params_per_q = params_per_q + [0] * (self.config.max_qubits - len(params_per_q))
        params_norm = [p / max(1, features.num_params) for p in params_per_q]
        embedding.extend(params_norm)

        # 4. Adjacency features (upper triangle, padded)
        adj_flat = []
        for i in range(self.config.max_qubits):
            for j in range(i + 1, self.config.max_qubits):
                if i < features.num_qubits and j < features.num_qubits:
                    # Normalize by max two-qubit gates
                    val = features.adjacency[i, j] / max(1, features.two_qubit_gates)
                else:
                    val = 0.0
                adj_flat.append(val)
        embedding.extend(adj_flat)

        # 5. Layer structure features
        layer_feats = self._encode_layer_structure(features.layer_structure)
        embedding.extend(layer_feats)

        # Convert to numpy and pad/truncate to embedding_dim
        embedding = np.array(embedding, dtype=np.float32)

        if len(embedding) < self.config.embedding_dim:
            embedding = np.pad(embedding, (0, self.config.embedding_dim - len(embedding)))
        elif len(embedding) > self.config.embedding_dim:
            embedding = embedding[:self.config.embedding_dim]

        return embedding

    def _encode_layer_structure(self, layers: list[dict]) -> list[float]:
        """Encode layer structure into fixed-size vector."""
        feats = []

        # Summary statistics
        num_layers = len(layers)
        feats.append(num_layers / 20)  # Normalized layer count

        # Average gates per layer
        if num_layers > 0:
            gates_per_layer = [sum(layer.values()) for layer in layers]
            feats.append(np.mean(gates_per_layer) / 10)
            feats.append(np.std(gates_per_layer) / (np.mean(gates_per_layer) + 1e-6))
        else:
            feats.extend([0, 0])

        # Two-qubit gate pattern (where do they appear?)
        two_q_pattern = []
        for layer in layers[:10]:
            has_two_q = any(g in TWO_QUBIT_GATES for g in layer.keys())
            two_q_pattern.append(1.0 if has_two_q else 0.0)
        two_q_pattern.extend([0.0] * (10 - len(two_q_pattern)))
        feats.extend(two_q_pattern)

        # Rotation gate pattern
        rot_pattern = []
        for layer in layers[:5]:
            has_rot = any(g in ROTATION_GATES for g in layer.keys())
            rot_pattern.append(1.0 if has_rot else 0.0)
        rot_pattern.extend([0.0] * (5 - len(rot_pattern)))
        feats.extend(rot_pattern)

        # Pad to fixed size
        while len(feats) < 20:
            feats.append(0.0)

        return feats[:20]

    def encode_batch(self, circuits: list[QuantumCircuit]) -> np.ndarray:
        """
        Encode a batch of circuits.

        Args:
            circuits: List of quantum circuits

        Returns:
            Embeddings array of shape (batch_size, embedding_dim)
        """
        embeddings = [self.encode(circuit) for circuit in circuits]
        return np.stack(embeddings)

    def encode_to_tensor(self, circuit: QuantumCircuit) -> torch.Tensor:
        """
        Encode circuit to PyTorch tensor.

        Args:
            circuit: Quantum circuit

        Returns:
            Tensor of shape (embedding_dim,)
        """
        embedding = self.encode(circuit)
        return torch.from_numpy(embedding)

    def encode_batch_to_tensor(self, circuits: list[QuantumCircuit]) -> torch.Tensor:
        """
        Encode batch of circuits to PyTorch tensor.

        Args:
            circuits: List of quantum circuits

        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        embeddings = self.encode_batch(circuits)
        return torch.from_numpy(embeddings)


class LearnedCircuitEncoder(nn.Module):
    """
    Neural network-based circuit encoder.

    Uses an MLP to transform raw features into learned embeddings.
    Can be fine-tuned with surrogate model training.
    """

    def __init__(self, config: CircuitEncoderConfig | None = None):
        """
        Initialize the learned encoder.

        Args:
            config: Encoder configuration
        """
        super().__init__()
        self.config = config or CircuitEncoderConfig()
        self.base_encoder = CircuitEncoder(self.config)

        # MLP to transform features
        raw_dim = self.base_encoder.raw_dim
        hidden_dim = 512

        self.encoder = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.config.embedding_dim),
            nn.LayerNorm(self.config.embedding_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through learned encoder.

        Args:
            features: Raw features tensor (batch, raw_dim)

        Returns:
            Embeddings tensor (batch, embedding_dim)
        """
        return self.encoder(features)

    def encode_circuit(self, circuit: QuantumCircuit) -> torch.Tensor:
        """
        Encode a single circuit.

        Args:
            circuit: Quantum circuit

        Returns:
            Embedding tensor (embedding_dim,)
        """
        # Get raw features from base encoder (before padding)
        raw = self.base_encoder.encode_raw(circuit)
        raw_tensor = torch.from_numpy(raw).unsqueeze(0)

        with torch.no_grad():
            embedding = self.forward(raw_tensor)

        return embedding.squeeze(0)

    def encode_batch(self, circuits: list[QuantumCircuit]) -> torch.Tensor:
        """
        Encode a batch of circuits.

        Args:
            circuits: List of quantum circuits

        Returns:
            Embeddings tensor (batch, embedding_dim)
        """
        raw_list = [self.base_encoder.encode_raw(circuit) for circuit in circuits]
        raw = np.stack(raw_list)
        raw_tensor = torch.from_numpy(raw)

        with torch.no_grad():
            embeddings = self.forward(raw_tensor)

        return embeddings


def encode_circuit(
    circuit: QuantumCircuit,
    config: CircuitEncoderConfig | None = None,
) -> np.ndarray:
    """
    Convenience function to encode a single circuit.

    Args:
        circuit: Quantum circuit
        config: Optional encoder configuration

    Returns:
        Dense embedding vector of shape (embedding_dim,)
    """
    encoder = CircuitEncoder(config)
    return encoder.encode(circuit)


def get_circuit_embedding(
    circuit: QuantumCircuit,
    embedding_dim: int = 256,
) -> np.ndarray:
    """
    Convenience function to get circuit embedding.

    Args:
        circuit: Quantum circuit
        embedding_dim: Desired embedding dimension

    Returns:
        Dense embedding vector of shape (embedding_dim,)
    """
    config = CircuitEncoderConfig(embedding_dim=embedding_dim)
    encoder = CircuitEncoder(config)
    return encoder.encode(circuit)


def compute_circuit_similarity(
    circuit1: QuantumCircuit,
    circuit2: QuantumCircuit,
    config: CircuitEncoderConfig | None = None,
) -> float:
    """
    Compute similarity between two circuits based on embeddings.

    Uses cosine similarity between embeddings.

    Args:
        circuit1: First quantum circuit
        circuit2: Second quantum circuit
        config: Optional encoder configuration

    Returns:
        Similarity score in range [-1, 1]
    """
    encoder = CircuitEncoder(config)
    emb1 = encoder.encode(circuit1)
    emb2 = encoder.encode(circuit2)

    # Cosine similarity
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    return float(np.dot(emb1, emb2) / (norm1 * norm2))
