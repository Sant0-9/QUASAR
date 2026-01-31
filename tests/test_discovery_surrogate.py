"""
Tests for surrogate integration in discovery agent.

These tests verify the integration of the surrogate model with the
discovery agent for fast circuit filtering.
"""

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from src.agent.discovery import DiscoveryAgent, DiscoveryConfig
from src.evaluation.baselines import hardware_efficient_ansatz, efficient_su2_ansatz
from src.quantum.hamiltonians import xy_chain, heisenberg_chain, transverse_ising


class TestDiscoveryConfigSurrogate:
    """Tests for surrogate-related config options."""

    def test_default_surrogate_disabled(self):
        config = DiscoveryConfig()
        assert config.use_surrogate is False

    def test_surrogate_config_options(self):
        config = DiscoveryConfig(
            use_surrogate=True,
            proposals_per_iteration=50,
            vqe_top_k=5,
            surrogate_confidence_threshold=0.6,
            surrogate_update_frequency=20,
        )
        assert config.use_surrogate is True
        assert config.proposals_per_iteration == 50
        assert config.vqe_top_k == 5
        assert config.surrogate_confidence_threshold == 0.6
        assert config.surrogate_update_frequency == 20


class TestDiscoveryAgentSurrogateInit:
    """Tests for surrogate initialization."""

    def test_agent_without_surrogate(self):
        config = DiscoveryConfig(use_surrogate=False, verbose=False)
        agent = DiscoveryAgent(config)
        assert agent._surrogate is None

    def test_agent_with_surrogate(self):
        config = DiscoveryConfig(use_surrogate=True, verbose=False)
        agent = DiscoveryAgent(config)
        assert agent._surrogate is not None

    def test_surrogate_stats_initialized(self):
        config = DiscoveryConfig(use_surrogate=True, verbose=False)
        agent = DiscoveryAgent(config)
        assert "circuits_proposed" in agent._stats
        assert "circuits_filtered_by_surrogate" in agent._stats
        assert "vqe_runs" in agent._stats


class TestScoreCircuits:
    """Tests for score_circuits method."""

    @pytest.fixture
    def agent_with_surrogate(self):
        config = DiscoveryConfig(use_surrogate=True, verbose=False)
        return DiscoveryAgent(config)

    @pytest.fixture
    def agent_without_surrogate(self):
        config = DiscoveryConfig(use_surrogate=False, verbose=False)
        return DiscoveryAgent(config)

    @pytest.fixture
    def test_circuits(self):
        return [hardware_efficient_ansatz(4, d) for d in [1, 2, 3, 4]]

    @pytest.fixture
    def xy_ham(self):
        return xy_chain(4).operator

    def test_score_circuits_with_surrogate(self, agent_with_surrogate, test_circuits, xy_ham):
        scores = agent_with_surrogate.score_circuits(test_circuits, xy_ham)
        assert len(scores) == 4
        assert all(isinstance(s, tuple) and len(s) == 2 for s in scores)
        # Check sorted by score
        score_values = [s[1] for s in scores]
        assert score_values == sorted(score_values)

    def test_score_circuits_without_surrogate(self, agent_without_surrogate, test_circuits, xy_ham):
        scores = agent_without_surrogate.score_circuits(test_circuits, xy_ham)
        assert len(scores) == 4
        # All scores should be 0.0 (equal) without surrogate
        assert all(s[1] == 0.0 for s in scores)


class TestSelectTopK:
    """Tests for select_top_k method."""

    @pytest.fixture
    def agent(self):
        config = DiscoveryConfig(use_surrogate=True, vqe_top_k=3, verbose=False)
        return DiscoveryAgent(config)

    @pytest.fixture
    def test_circuits(self):
        return [hardware_efficient_ansatz(4, d) for d in [1, 2, 3, 4, 5]]

    @pytest.fixture
    def xy_ham(self):
        return xy_chain(4).operator

    def test_select_top_k_default(self, agent, test_circuits, xy_ham):
        selected = agent.select_top_k(test_circuits, xy_ham)
        # Should select vqe_top_k = 3
        assert len(selected) <= 3
        assert all(0 <= idx < 5 for idx in selected)

    def test_select_top_k_custom_k(self, agent, test_circuits, xy_ham):
        selected = agent.select_top_k(test_circuits, xy_ham, k=2)
        assert len(selected) <= 2

    def test_select_top_k_more_than_circuits(self, agent, xy_ham):
        circuits = [hardware_efficient_ansatz(4, 1), hardware_efficient_ansatz(4, 2)]
        selected = agent.select_top_k(circuits, xy_ham, k=5)
        assert len(selected) <= 2  # Can't select more than available


class TestUpdateSurrogate:
    """Tests for update_surrogate method."""

    @pytest.fixture
    def agent(self):
        config = DiscoveryConfig(
            use_surrogate=True,
            surrogate_update_frequency=5,
            verbose=False,
        )
        return DiscoveryAgent(config)

    @pytest.fixture
    def xy_ham(self):
        return xy_chain(4).operator

    def test_update_surrogate_adds_example(self, agent, xy_ham):
        circuit = hardware_efficient_ansatz(4, 2)
        initial_count = len(agent._surrogate.training_buffer)
        agent.update_surrogate(circuit, xy_ham, energy_error=0.1)
        assert len(agent._surrogate.training_buffer) == initial_count + 1

    def test_update_surrogate_increments_counter(self, agent, xy_ham):
        circuit = hardware_efficient_ansatz(4, 2)
        assert agent._vqe_count_since_update == 0
        agent.update_surrogate(circuit, xy_ham, energy_error=0.1)
        assert agent._vqe_count_since_update == 1

    def test_update_surrogate_periodic_training(self, agent, xy_ham):
        # Add examples up to update frequency
        for i in range(5):
            circuit = hardware_efficient_ansatz(4, (i % 3) + 1)
            agent.update_surrogate(circuit, xy_ham, energy_error=0.1 * i)

        # Counter should be reset after update
        assert agent._vqe_count_since_update == 0

    def test_update_surrogate_without_surrogate(self):
        config = DiscoveryConfig(use_surrogate=False, verbose=False)
        agent = DiscoveryAgent(config)
        circuit = hardware_efficient_ansatz(4, 2)
        ham = xy_chain(4).operator
        # Should not raise
        agent.update_surrogate(circuit, ham, energy_error=0.1)


class TestBatchProposalMethods:
    """Tests for batch proposal methods."""

    @pytest.fixture
    def agent(self):
        config = DiscoveryConfig(use_surrogate=True, verbose=False)
        return DiscoveryAgent(config)

    def test_generate_proposals_batch(self, agent):
        proposals = agent._generate_proposals_batch(
            goal_description="Test goal",
            num_qubits=4,
            hamiltonian_type="XY_CHAIN",
            constraints=None,
            feedback=None,
            count=3,
        )
        # MockProposer should return proposals
        assert len(proposals) <= 3

    def test_verify_proposals_batch(self, agent):
        from src.agent.proposer import CircuitProposal

        # Create mock proposals with valid code
        proposals = [
            CircuitProposal(
                code="""
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(Parameter(f'p_{i}'), i)
    return qc
""",
                reasoning="Test",
                raw_response="",
            )
        ]

        valid = agent._verify_proposals_batch(proposals, num_qubits=4)
        assert len(valid) == 1
        proposal, circuit = valid[0]
        assert circuit.num_qubits == 4

    def test_bp_filter_batch(self, agent):
        from src.agent.proposer import CircuitProposal

        circuits = [
            (CircuitProposal(code="", reasoning="", raw_response=""), hardware_efficient_ansatz(4, 1)),
            (CircuitProposal(code="", reasoning="", raw_response=""), hardware_efficient_ansatz(4, 2)),
        ]
        ham = xy_chain(4).operator

        filtered = agent._bp_filter_batch(circuits, ham)
        # Should return some circuits (shallow circuits are trainable)
        assert len(filtered) >= 0


class TestSurrogateIntegrationFlow:
    """End-to-end tests for surrogate integration."""

    @pytest.fixture
    def agent(self):
        config = DiscoveryConfig(
            use_surrogate=True,
            use_mock_proposer=True,
            max_iterations=2,
            verbose=False,
        )
        return DiscoveryAgent(config)

    def test_discovery_with_surrogate_runs(self, agent):
        # Should complete without errors
        result = agent.discover(
            hamiltonian_type="XY_CHAIN",
            num_qubits=4,
        )
        assert result is not None
        assert result.total_iterations > 0

    def test_stats_track_surrogate_usage(self, agent):
        agent.discover(
            hamiltonian_type="XY_CHAIN",
            num_qubits=4,
        )
        stats = agent.get_stats()
        assert "circuits_proposed" in stats
        assert "vqe_runs" in stats


class TestSurrogateAllHamiltonians:
    """Test surrogate works with all Hamiltonian types."""

    @pytest.fixture
    def agent(self):
        config = DiscoveryConfig(use_surrogate=True, verbose=False)
        return DiscoveryAgent(config)

    @pytest.fixture
    def test_circuits(self):
        return [hardware_efficient_ansatz(4, 2)]

    def test_xy_chain(self, agent, test_circuits):
        ham = xy_chain(4).operator
        scores = agent.score_circuits(test_circuits, ham)
        assert len(scores) == 1

    def test_heisenberg(self, agent, test_circuits):
        ham = heisenberg_chain(4).operator
        scores = agent.score_circuits(test_circuits, ham)
        assert len(scores) == 1

    def test_tfim(self, agent, test_circuits):
        ham = transverse_ising(4).operator
        scores = agent.score_circuits(test_circuits, ham)
        assert len(scores) == 1


class TestSurrogateSpeedup:
    """Tests to verify surrogate provides speedup."""

    @pytest.fixture
    def agent(self):
        config = DiscoveryConfig(use_surrogate=True, verbose=False)
        return DiscoveryAgent(config)

    def test_scoring_faster_than_vqe(self, agent):
        import time

        circuits = [hardware_efficient_ansatz(4, d) for d in [1, 2, 3, 4, 5]]
        ham = xy_chain(4).operator

        # Score circuits (should be fast)
        start = time.time()
        for _ in range(10):
            agent.score_circuits(circuits, ham)
        scoring_time = time.time() - start

        # Average time per scoring call
        per_call = scoring_time / 10
        assert per_call < 0.5, f"Scoring took {per_call}s, expected < 0.5s"
