"""
Tests for the circuit encoder module.

These tests verify the encoding of quantum circuits into fixed-size
dense vectors, including gate type encoding, structural features,
entanglement patterns, and similarity computation.
"""

import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from src.quasar.circuit_encoder import (
    CircuitEncoder,
    CircuitEncoderConfig,
    CircuitFeatures,
    LearnedCircuitEncoder,
    GATE_TYPES,
    GATE_TO_IDX,
    SINGLE_QUBIT_GATES,
    TWO_QUBIT_GATES,
    compute_circuit_similarity,
    encode_circuit,
    get_circuit_embedding,
)


class TestCircuitEncoderConfig:
    """Tests for CircuitEncoderConfig."""

    def test_default_config(self):
        config = CircuitEncoderConfig()
        assert config.embedding_dim == 256
        assert config.max_qubits == 12
        assert config.max_depth == 100

    def test_custom_config(self):
        config = CircuitEncoderConfig(embedding_dim=128, max_qubits=8)
        assert config.embedding_dim == 128
        assert config.max_qubits == 8


class TestGateConstants:
    """Tests for gate type constants."""

    def test_single_qubit_gates(self):
        assert "h" in SINGLE_QUBIT_GATES
        assert "rx" in SINGLE_QUBIT_GATES
        assert "cx" not in SINGLE_QUBIT_GATES

    def test_two_qubit_gates(self):
        assert "cx" in TWO_QUBIT_GATES
        assert "cz" in TWO_QUBIT_GATES
        assert "h" not in TWO_QUBIT_GATES

    def test_gate_index_mapping(self):
        assert "cx" in GATE_TO_IDX
        assert "ry" in GATE_TO_IDX
        assert GATE_TO_IDX["other"] == len(GATE_TYPES) - 1


class TestCircuitFeatures:
    """Tests for CircuitFeatures dataclass."""

    def test_creation(self):
        features = CircuitFeatures(
            num_qubits=4,
            depth=5,
            num_params=10,
            total_gates=15,
            single_qubit_gates=12,
            two_qubit_gates=3,
            gate_counts={"rx": 4, "ry": 4, "rz": 4, "cx": 3},
            entanglement_density=0.5,
            qubit_degrees=[2, 2, 1, 1],
            adjacency=np.zeros((4, 4)),
            layer_structure=[{"rx": 4}, {"cx": 3}],
            param_per_qubit=[3, 3, 2, 2],
        )
        assert features.num_qubits == 4
        assert features.depth == 5
        assert features.total_gates == 15


class TestCircuitEncoderBasic:
    """Basic tests for CircuitEncoder."""

    @pytest.fixture
    def encoder(self):
        return CircuitEncoder()

    @pytest.fixture
    def simple_circuit(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        return qc

    @pytest.fixture
    def parameterized_circuit(self):
        qc = QuantumCircuit(4)
        for i in range(4):
            p = Parameter(f"theta_{i}")
            qc.ry(p, i)
        qc.cx(0, 1)
        qc.cx(2, 3)
        return qc

    def test_encoder_creation(self, encoder):
        assert encoder.config.embedding_dim == 256

    def test_encode_returns_numpy_array(self, encoder, simple_circuit):
        embedding = encoder.encode(simple_circuit)
        assert isinstance(embedding, np.ndarray)

    def test_embedding_shape(self, encoder, simple_circuit):
        embedding = encoder.encode(simple_circuit)
        assert embedding.shape == (256,)

    def test_embedding_dtype(self, encoder, simple_circuit):
        embedding = encoder.encode(simple_circuit)
        assert embedding.dtype == np.float32

    def test_custom_embedding_dim(self, simple_circuit):
        config = CircuitEncoderConfig(embedding_dim=128)
        encoder = CircuitEncoder(config)
        embedding = encoder.encode(simple_circuit)
        assert embedding.shape == (128,)

    def test_extract_features(self, encoder, simple_circuit):
        features = encoder.extract_features(simple_circuit)
        assert isinstance(features, CircuitFeatures)
        assert features.num_qubits == 4
        assert features.total_gates == 4  # 1 H + 3 CX


class TestCircuitEncoderGateCounts:
    """Tests for gate counting in encoder."""

    @pytest.fixture
    def encoder(self):
        return CircuitEncoder()

    def test_single_qubit_gate_count(self, encoder):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.x(0)
        qc.y(1)
        features = encoder.extract_features(qc)
        assert features.single_qubit_gates == 4
        assert features.two_qubit_gates == 0

    def test_two_qubit_gate_count(self, encoder):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cz(0, 2)
        features = encoder.extract_features(qc)
        assert features.two_qubit_gates == 3
        assert features.single_qubit_gates == 0

    def test_rotation_gates_count(self, encoder):
        qc = QuantumCircuit(2)
        p1 = Parameter("a")
        p2 = Parameter("b")
        qc.rx(p1, 0)
        qc.ry(p2, 1)
        qc.rz(0.5, 0)
        features = encoder.extract_features(qc)
        assert features.num_params == 2


class TestCircuitEncoderConnectivity:
    """Tests for connectivity analysis."""

    @pytest.fixture
    def encoder(self):
        return CircuitEncoder()

    def test_linear_connectivity(self, encoder):
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        features = encoder.extract_features(qc)
        # Check adjacency matrix
        assert features.adjacency[0, 1] == 1
        assert features.adjacency[1, 2] == 1
        assert features.adjacency[2, 3] == 1
        assert features.adjacency[0, 3] == 0

    def test_qubit_degrees(self, encoder):
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        features = encoder.extract_features(qc)
        # Qubit 0 has degree 3, others have degree 1
        assert features.qubit_degrees[0] == 3
        assert features.qubit_degrees[1] == 1
        assert features.qubit_degrees[2] == 1
        assert features.qubit_degrees[3] == 1

    def test_entanglement_density(self, encoder):
        # Full connectivity
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        features = encoder.extract_features(qc)
        # 3 gates out of 3 possible pairs
        assert features.entanglement_density == pytest.approx(1.0)


class TestCircuitEncoderEmbeddings:
    """Tests for embedding properties."""

    @pytest.fixture
    def encoder(self):
        return CircuitEncoder()

    def test_embedding_is_finite(self, encoder):
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.h(i)
        for i in range(3):
            qc.cx(i, i + 1)
        embedding = encoder.encode(qc)
        assert np.all(np.isfinite(embedding))

    def test_embedding_normalized_range(self, encoder):
        qc = QuantumCircuit(4)
        for i in range(4):
            p = Parameter(f"p_{i}")
            qc.ry(p, i)
        qc.cx(0, 1)
        qc.cx(2, 3)
        embedding = encoder.encode(qc)
        # Most features should be in [0, 1] range
        assert np.all(embedding >= -0.1)  # Allow small negative
        assert np.all(embedding <= 2.0)  # Allow some overflow


class TestSimilarCircuitsSimilarEmbeddings:
    """Test that similar circuits produce similar embeddings."""

    @pytest.fixture
    def encoder(self):
        return CircuitEncoder()

    def test_identical_circuits_same_embedding(self, encoder):
        qc1 = QuantumCircuit(4)
        qc1.h(0)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit(4)
        qc2.h(0)
        qc2.cx(0, 1)

        emb1 = encoder.encode(qc1)
        emb2 = encoder.encode(qc2)

        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_similar_circuits_close_embeddings(self, encoder):
        # Two circuits with same structure, different params
        qc1 = QuantumCircuit(4)
        for i in range(4):
            qc1.ry(Parameter(f"a_{i}"), i)
        qc1.cx(0, 1)
        qc1.cx(2, 3)

        qc2 = QuantumCircuit(4)
        for i in range(4):
            qc2.ry(Parameter(f"b_{i}"), i)
        qc2.cx(0, 1)
        qc2.cx(2, 3)

        emb1 = encoder.encode(qc1)
        emb2 = encoder.encode(qc2)

        similarity = compute_circuit_similarity(qc1, qc2)
        assert similarity > 0.99  # Very similar


class TestDifferentCircuitsDifferentEmbeddings:
    """Test that different circuits produce different embeddings."""

    @pytest.fixture
    def encoder(self):
        return CircuitEncoder()

    def test_different_gate_types_different_embedding(self, encoder):
        qc1 = QuantumCircuit(4)
        for i in range(4):
            qc1.rx(Parameter(f"p_{i}"), i)

        qc2 = QuantumCircuit(4)
        for i in range(4):
            qc2.rz(Parameter(f"p_{i}"), i)

        emb1 = encoder.encode(qc1)
        emb2 = encoder.encode(qc2)

        # Embeddings should be different
        assert not np.allclose(emb1, emb2)

    def test_different_connectivity_different_embedding(self, encoder):
        # Linear connectivity
        qc1 = QuantumCircuit(4)
        qc1.cx(0, 1)
        qc1.cx(1, 2)
        qc1.cx(2, 3)

        # Star connectivity
        qc2 = QuantumCircuit(4)
        qc2.cx(0, 1)
        qc2.cx(0, 2)
        qc2.cx(0, 3)

        emb1 = encoder.encode(qc1)
        emb2 = encoder.encode(qc2)

        assert not np.allclose(emb1, emb2)

    def test_different_depths_different_embedding(self, encoder):
        # Shallow circuit
        qc1 = QuantumCircuit(4)
        qc1.h(0)
        qc1.cx(0, 1)

        # Deep circuit
        qc2 = QuantumCircuit(4)
        for _ in range(5):
            qc2.h(0)
            qc2.cx(0, 1)
            qc2.cx(1, 2)
            qc2.cx(2, 3)

        emb1 = encoder.encode(qc1)
        emb2 = encoder.encode(qc2)

        assert not np.allclose(emb1, emb2)


class TestCircuitEncoderBatch:
    """Tests for batch encoding."""

    @pytest.fixture
    def encoder(self):
        return CircuitEncoder()

    @pytest.fixture
    def circuit_batch(self):
        circuits = []
        for n_layers in [1, 2, 3, 4]:
            qc = QuantumCircuit(4)
            for _ in range(n_layers):
                for i in range(4):
                    qc.ry(Parameter(f"p_{len(qc.parameters)}"), i)
                for i in range(3):
                    qc.cx(i, i + 1)
            circuits.append(qc)
        return circuits

    def test_batch_encode_shape(self, encoder, circuit_batch):
        embeddings = encoder.encode_batch(circuit_batch)
        assert embeddings.shape == (4, 256)

    def test_batch_encode_tensor(self, encoder, circuit_batch):
        embeddings = encoder.encode_batch_to_tensor(circuit_batch)
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (4, 256)

    def test_single_to_tensor(self, encoder, circuit_batch):
        embedding = encoder.encode_to_tensor(circuit_batch[0])
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (256,)


class TestLearnedCircuitEncoder:
    """Tests for LearnedCircuitEncoder."""

    @pytest.fixture
    def learned_encoder(self):
        return LearnedCircuitEncoder()

    @pytest.fixture
    def simple_circuit(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        return qc

    def test_creation(self, learned_encoder):
        assert learned_encoder.config.embedding_dim == 256

    def test_forward_shape(self, learned_encoder):
        # Create random input matching raw dim
        raw_dim = learned_encoder.base_encoder.raw_dim
        x = torch.randn(8, raw_dim)
        output = learned_encoder(x)
        assert output.shape == (8, 256)

    def test_encode_circuit(self, learned_encoder, simple_circuit):
        embedding = learned_encoder.encode_circuit(simple_circuit)
        assert embedding.shape == (256,)

    def test_encode_batch(self, learned_encoder):
        circuits = [QuantumCircuit(4) for _ in range(5)]
        for qc in circuits:
            qc.h(0)
            qc.cx(0, 1)

        embeddings = learned_encoder.encode_batch(circuits)
        assert embeddings.shape == (5, 256)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_encode_circuit(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        embedding = encode_circuit(qc)
        assert embedding.shape == (256,)

    def test_get_circuit_embedding(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        embedding = get_circuit_embedding(qc)
        assert embedding.shape == (256,)

    def test_get_circuit_embedding_custom_dim(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        embedding = get_circuit_embedding(qc, embedding_dim=128)
        assert embedding.shape == (128,)

    def test_compute_similarity_same(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        similarity = compute_circuit_similarity(qc, qc)
        assert similarity == pytest.approx(1.0)

    def test_compute_similarity_different(self):
        qc1 = QuantumCircuit(4)
        qc1.h(0)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit(4)
        for i in range(4):
            qc2.x(i)

        similarity = compute_circuit_similarity(qc1, qc2)
        # Different circuits should have lower similarity
        assert similarity < 0.95


class TestCircuitEncoderEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def encoder(self):
        return CircuitEncoder()

    def test_empty_circuit(self, encoder):
        qc = QuantumCircuit(4)
        embedding = encoder.encode(qc)
        assert embedding.shape == (256,)
        assert np.all(np.isfinite(embedding))

    def test_single_qubit_circuit(self, encoder):
        qc = QuantumCircuit(1)
        qc.h(0)
        embedding = encoder.encode(qc)
        assert embedding.shape == (256,)

    def test_large_circuit(self, encoder):
        qc = QuantumCircuit(8)
        for _ in range(20):
            for i in range(8):
                qc.ry(Parameter(f"p_{len(qc.parameters)}"), i)
            for i in range(7):
                qc.cx(i, i + 1)
        embedding = encoder.encode(qc)
        assert embedding.shape == (256,)
        assert np.all(np.isfinite(embedding))

    def test_circuit_with_barriers(self, encoder):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)
        qc.barrier()
        embedding = encoder.encode(qc)
        assert embedding.shape == (256,)

    def test_circuit_at_max_qubits(self, encoder):
        qc = QuantumCircuit(12)  # max_qubits default
        for i in range(12):
            qc.h(i)
        embedding = encoder.encode(qc)
        assert embedding.shape == (256,)

    def test_circuit_exceeds_max_qubits(self):
        config = CircuitEncoderConfig(max_qubits=4)
        encoder = CircuitEncoder(config)
        qc = QuantumCircuit(6)  # Exceeds max_qubits
        for i in range(6):
            qc.h(i)
        # Should still work, just truncate/pad features
        embedding = encoder.encode(qc)
        assert embedding.shape == (256,)


class TestCircuitEncoderRobustness:
    """Tests for encoder robustness."""

    @pytest.fixture
    def encoder(self):
        return CircuitEncoder()

    def test_all_baseline_circuits(self, encoder):
        """Test encoding all baseline circuit types."""
        from src.evaluation.baselines import (
            hardware_efficient_ansatz,
            efficient_su2_ansatz,
            real_amplitudes_ansatz,
            excitation_preserving_ansatz,
        )

        baselines = [
            hardware_efficient_ansatz(4, 2),
            efficient_su2_ansatz(4, 2),
            real_amplitudes_ansatz(4, 2),
            excitation_preserving_ansatz(4, 2),
        ]

        for qc in baselines:
            embedding = encoder.encode(qc)
            assert embedding.shape == (256,)
            assert np.all(np.isfinite(embedding))

    def test_random_circuits_different_embeddings(self, encoder):
        """Test that random circuits get different embeddings."""
        from src.evaluation.baselines import hardware_efficient_ansatz

        embeddings = []
        for depth in range(1, 5):
            qc = hardware_efficient_ansatz(4, depth, entanglement="linear")
            embeddings.append(encoder.encode(qc))

        # Check that embeddings are distinct
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Different depths should give different embeddings
                assert not np.allclose(embeddings[i], embeddings[j])
