"""
Tests for the surrogate model module.

These tests verify the surrogate model for predicting circuit quality
including Hamiltonian encoding, model architecture, training, and inference.
"""

import time

import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from src.quantum.hamiltonians import xy_chain, heisenberg_chain, transverse_ising
from src.evaluation.baselines import hardware_efficient_ansatz, efficient_su2_ansatz
from src.quasar.surrogate import (
    HamiltonianEncoder,
    SurrogateConfig,
    SurrogateEvaluator,
    SurrogateModel,
    SurrogatePrediction,
    SurrogateTrainer,
    SurrogateTrainingExample,
    Trainability,
    create_surrogate_evaluator,
)


class TestSurrogateConfig:
    """Tests for SurrogateConfig."""

    def test_default_config(self):
        config = SurrogateConfig()
        assert config.circuit_embedding_dim == 256
        assert config.hamiltonian_embedding_dim == 256
        assert config.hidden_dim == 512

    def test_custom_config(self):
        config = SurrogateConfig(hidden_dim=256, num_layers=2)
        assert config.hidden_dim == 256
        assert config.num_layers == 2


class TestTrainability:
    """Tests for Trainability enum."""

    def test_enum_values(self):
        assert Trainability.HIGH.value == "high"
        assert Trainability.MEDIUM.value == "medium"
        assert Trainability.LOW.value == "low"


class TestSurrogatePrediction:
    """Tests for SurrogatePrediction dataclass."""

    def test_creation(self):
        pred = SurrogatePrediction(
            predicted_error=0.1,
            trainability=Trainability.HIGH,
            confidence=0.95,
            inference_time_ms=5.0,
        )
        assert pred.predicted_error == 0.1
        assert pred.trainability == Trainability.HIGH
        assert pred.confidence == 0.95


class TestHamiltonianEncoder:
    """Tests for HamiltonianEncoder."""

    @pytest.fixture
    def encoder(self):
        return HamiltonianEncoder()

    @pytest.fixture
    def xy_ham(self):
        return xy_chain(4).operator

    @pytest.fixture
    def heisenberg_ham(self):
        return heisenberg_chain(4).operator

    def test_encoder_creation(self, encoder):
        assert encoder.config.hamiltonian_embedding_dim == 256

    def test_encode_returns_numpy(self, encoder, xy_ham):
        embedding = encoder.encode(xy_ham)
        assert isinstance(embedding, np.ndarray)

    def test_embedding_shape(self, encoder, xy_ham):
        embedding = encoder.encode(xy_ham)
        assert embedding.shape == (256,)

    def test_embedding_dtype(self, encoder, xy_ham):
        embedding = encoder.encode(xy_ham)
        assert embedding.dtype == np.float32

    def test_encode_different_hamiltonians(self, encoder, xy_ham, heisenberg_ham):
        emb_xy = encoder.encode(xy_ham)
        emb_heis = encoder.encode(heisenberg_ham)
        # Different Hamiltonians should give different embeddings
        assert not np.allclose(emb_xy, emb_heis)

    def test_encode_same_hamiltonian(self, encoder, xy_ham):
        emb1 = encoder.encode(xy_ham)
        emb2 = encoder.encode(xy_ham)
        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_encode_batch(self, encoder):
        hams = [xy_chain(4).operator, heisenberg_chain(4).operator, transverse_ising(4).operator]
        embeddings = encoder.encode_batch(hams)
        assert embeddings.shape == (3, 256)

    def test_embedding_is_finite(self, encoder):
        for n_qubits in [2, 4, 6]:
            ham = xy_chain(n_qubits).operator
            embedding = encoder.encode(ham)
            assert np.all(np.isfinite(embedding))


class TestSurrogateModel:
    """Tests for SurrogateModel architecture."""

    @pytest.fixture
    def model(self):
        return SurrogateModel()

    @pytest.fixture
    def batch_data(self):
        batch_size = 8
        circuit_emb = torch.randn(batch_size, 256)
        ham_emb = torch.randn(batch_size, 256)
        return circuit_emb, ham_emb

    def test_model_creation(self, model):
        assert isinstance(model, nn.Module)

    def test_forward_shapes(self, model, batch_data):
        circuit_emb, ham_emb = batch_data
        energy, trainability, confidence = model(circuit_emb, ham_emb)

        assert energy.shape == (8,)
        assert trainability.shape == (8, 3)
        assert confidence.shape == (8,)

    def test_predict_energy(self, model, batch_data):
        circuit_emb, ham_emb = batch_data
        energy = model.predict_energy(circuit_emb, ham_emb)
        assert energy.shape == (8,)

    def test_confidence_range(self, model, batch_data):
        circuit_emb, ham_emb = batch_data
        _, _, confidence = model(circuit_emb, ham_emb)
        assert torch.all(confidence >= 0)
        assert torch.all(confidence <= 1)

    def test_model_trainable(self, model, batch_data):
        circuit_emb, ham_emb = batch_data
        target = torch.randn(8)

        optimizer = torch.optim.Adam(model.parameters())
        optimizer.zero_grad()

        energy, _, _ = model(circuit_emb, ham_emb)
        loss = torch.nn.functional.mse_loss(energy, target)
        loss.backward()
        optimizer.step()

        # Check gradients were computed
        assert any(p.grad is not None for p in model.parameters())


class TestSurrogateEvaluator:
    """Tests for SurrogateEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return SurrogateEvaluator()

    @pytest.fixture
    def simple_circuit(self):
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.ry(Parameter(f"p_{i}"), i)
        qc.cx(0, 1)
        qc.cx(2, 3)
        return qc

    @pytest.fixture
    def xy_ham(self):
        return xy_chain(4).operator

    def test_evaluator_creation(self, evaluator):
        assert evaluator.config is not None
        assert evaluator.circuit_encoder is not None
        assert evaluator.hamiltonian_encoder is not None

    def test_predict_returns_prediction(self, evaluator, simple_circuit, xy_ham):
        pred = evaluator.predict(simple_circuit, xy_ham)
        assert isinstance(pred, SurrogatePrediction)

    def test_prediction_has_all_fields(self, evaluator, simple_circuit, xy_ham):
        pred = evaluator.predict(simple_circuit, xy_ham)
        assert isinstance(pred.predicted_error, float)
        assert isinstance(pred.trainability, Trainability)
        assert isinstance(pred.confidence, float)
        assert pred.inference_time_ms > 0

    def test_confidence_in_range(self, evaluator, simple_circuit, xy_ham):
        pred = evaluator.predict(simple_circuit, xy_ham)
        assert 0 <= pred.confidence <= 1

    def test_predict_batch(self, evaluator, xy_ham):
        circuits = [hardware_efficient_ansatz(4, d) for d in range(1, 4)]
        predictions = evaluator.predict_batch(circuits, xy_ham)
        assert len(predictions) == 3
        assert all(isinstance(p, SurrogatePrediction) for p in predictions)

    def test_score_circuits(self, evaluator, xy_ham):
        circuits = [hardware_efficient_ansatz(4, d) for d in range(1, 5)]
        scores = evaluator.score_circuits(circuits, xy_ham)
        assert len(scores) == 4
        # Should be sorted
        assert all(scores[i][1] <= scores[i+1][1] for i in range(len(scores)-1))

    def test_select_top_k(self, evaluator, xy_ham):
        circuits = [hardware_efficient_ansatz(4, d) for d in range(1, 6)]
        selected = evaluator.select_top_k(circuits, xy_ham, k=3)
        assert len(selected) <= 3
        assert all(0 <= idx < 5 for idx in selected)


class TestSurrogateEvaluatorInferenceTime:
    """Tests for inference time requirements."""

    @pytest.fixture
    def evaluator(self):
        return SurrogateEvaluator()

    def test_inference_time_under_50ms(self, evaluator):
        circuit = hardware_efficient_ansatz(4, 2)
        ham = xy_chain(4).operator

        # Warm up
        evaluator.predict(circuit, ham)

        # Measure
        times = []
        for _ in range(10):
            pred = evaluator.predict(circuit, ham)
            times.append(pred.inference_time_ms)

        avg_time = np.mean(times)
        assert avg_time < 50, f"Inference time {avg_time}ms exceeds 50ms limit"

    def test_batch_inference_efficient(self, evaluator):
        circuits = [hardware_efficient_ansatz(4, d) for d in [1, 2, 3]]
        ham = xy_chain(4).operator

        # Warm up
        evaluator.predict_batch(circuits, ham)

        start = time.time()
        predictions = evaluator.predict_batch(circuits, ham)
        elapsed = (time.time() - start) * 1000

        per_circuit = elapsed / len(circuits)
        assert per_circuit < 30, f"Per-circuit time {per_circuit}ms too slow"


class TestSurrogateEvaluatorActiveLearning:
    """Tests for active learning functionality."""

    @pytest.fixture
    def evaluator(self):
        return SurrogateEvaluator()

    def test_add_training_example(self, evaluator):
        circuit = hardware_efficient_ansatz(4, 2)
        ham = xy_chain(4).operator

        evaluator.add_training_example(circuit, ham, energy_error=0.1)
        assert len(evaluator.training_buffer) == 1

    def test_multiple_training_examples(self, evaluator):
        ham = xy_chain(4).operator

        for i in range(5):
            circuit = hardware_efficient_ansatz(4, i + 1)
            evaluator.add_training_example(circuit, ham, energy_error=0.1 * i)

        assert len(evaluator.training_buffer) == 5

    def test_update_from_buffer_clears_buffer(self, evaluator):
        ham = xy_chain(4).operator

        # Add enough examples
        for i in range(50):
            circuit = hardware_efficient_ansatz(4, (i % 3) + 1)
            evaluator.add_training_example(circuit, ham, energy_error=0.1 * (i % 5))

        evaluator.update_from_buffer(epochs=1)
        assert len(evaluator.training_buffer) == 0


class TestSurrogateTrainer:
    """Tests for SurrogateTrainer."""

    @pytest.fixture
    def trainer(self):
        return SurrogateTrainer()

    @pytest.fixture
    def training_examples(self):
        examples = []
        for n_layers in [1, 2, 3]:
            for ham_fn in [xy_chain, heisenberg_chain, transverse_ising]:
                circuit = hardware_efficient_ansatz(4, n_layers)
                ham = ham_fn(4).operator
                # Simulate energy error based on depth
                error = 0.5 / n_layers + np.random.randn() * 0.05
                examples.append(SurrogateTrainingExample(
                    circuit=circuit,
                    hamiltonian=ham,
                    energy_error=max(0, error),
                ))
        return examples

    def test_trainer_creation(self, trainer):
        assert trainer.config is not None
        assert trainer.model is not None

    def test_prepare_dataset(self, trainer, training_examples):
        c_emb, h_emb, e_targets, t_targets = trainer.prepare_dataset(training_examples)

        assert c_emb.shape == (len(training_examples), 256)
        assert h_emb.shape == (len(training_examples), 256)
        assert len(e_targets) == len(training_examples)
        assert len(t_targets) == len(training_examples)

    def test_train_runs(self, trainer, training_examples):
        # Split into train/val
        train = training_examples[:6]
        val = training_examples[6:]

        history = trainer.train(train, val, epochs=2, batch_size=2)

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) > 0

    def test_get_evaluator(self, trainer, training_examples):
        trainer.train(training_examples, epochs=1, batch_size=4)
        evaluator = trainer.get_evaluator()

        assert isinstance(evaluator, SurrogateEvaluator)

        # Should be able to make predictions
        circuit = hardware_efficient_ansatz(4, 2)
        ham = xy_chain(4).operator
        pred = evaluator.predict(circuit, ham)
        assert isinstance(pred, SurrogatePrediction)


class TestSurrogateModelSaveLoad:
    """Tests for save/load functionality."""

    @pytest.fixture
    def evaluator(self):
        return SurrogateEvaluator()

    def test_save_load_roundtrip(self, evaluator, tmp_path):
        # Make a prediction before saving
        circuit = hardware_efficient_ansatz(4, 2)
        ham = xy_chain(4).operator
        pred_before = evaluator.predict(circuit, ham)

        # Save
        model_path = tmp_path / "surrogate.pt"
        evaluator.save(model_path)

        # Load into new evaluator
        new_evaluator = SurrogateEvaluator()
        new_evaluator.load(model_path)

        # Predictions should match
        pred_after = new_evaluator.predict(circuit, ham)
        assert abs(pred_before.predicted_error - pred_after.predicted_error) < 1e-5


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_surrogate_evaluator(self):
        evaluator = create_surrogate_evaluator()
        assert isinstance(evaluator, SurrogateEvaluator)

    def test_create_with_config(self):
        config = SurrogateConfig(hidden_dim=256)
        evaluator = create_surrogate_evaluator(config=config)
        assert evaluator.config.hidden_dim == 256


class TestSurrogateEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def evaluator(self):
        return SurrogateEvaluator()

    def test_empty_circuit(self, evaluator):
        circuit = QuantumCircuit(4)  # No gates
        ham = xy_chain(4).operator
        pred = evaluator.predict(circuit, ham)
        assert isinstance(pred, SurrogatePrediction)

    def test_single_qubit_system(self, evaluator):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        ham = xy_chain(2).operator
        pred = evaluator.predict(circuit, ham)
        assert isinstance(pred, SurrogatePrediction)

    def test_large_circuit(self, evaluator):
        circuit = hardware_efficient_ansatz(8, 5)
        ham = xy_chain(8).operator
        pred = evaluator.predict(circuit, ham)
        assert isinstance(pred, SurrogatePrediction)
        assert np.isfinite(pred.predicted_error)


class TestSurrogateRobustness:
    """Tests for model robustness."""

    @pytest.fixture
    def evaluator(self):
        return SurrogateEvaluator()

    def test_all_hamiltonians(self, evaluator):
        """Test with all supported Hamiltonians."""
        circuit = hardware_efficient_ansatz(4, 2)

        for ham_fn in [xy_chain, heisenberg_chain, transverse_ising]:
            ham = ham_fn(4).operator
            pred = evaluator.predict(circuit, ham)
            assert isinstance(pred, SurrogatePrediction)
            assert np.isfinite(pred.predicted_error)

    def test_all_baselines(self, evaluator):
        """Test with all baseline circuits."""
        ham = xy_chain(4).operator

        from src.evaluation.baselines import (
            real_amplitudes_ansatz,
            excitation_preserving_ansatz,
        )

        baselines = [
            hardware_efficient_ansatz(4, 2),
            efficient_su2_ansatz(4, 2),
            real_amplitudes_ansatz(4, 2),
            excitation_preserving_ansatz(4, 2),
        ]

        for circuit in baselines:
            pred = evaluator.predict(circuit, ham)
            assert isinstance(pred, SurrogatePrediction)

    def test_deterministic_predictions(self, evaluator):
        """Test that predictions are deterministic in eval mode."""
        circuit = hardware_efficient_ansatz(4, 2)
        ham = xy_chain(4).operator

        pred1 = evaluator.predict(circuit, ham)
        pred2 = evaluator.predict(circuit, ham)

        assert pred1.predicted_error == pytest.approx(pred2.predicted_error)
        assert pred1.confidence == pytest.approx(pred2.confidence)


# Import nn for TestSurrogateModel
import torch.nn as nn
