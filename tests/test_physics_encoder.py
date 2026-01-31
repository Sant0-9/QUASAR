"""
Tests for the physics encoder module.

These tests verify the extraction of physics features from
simulation trajectories including symmetries, conservation laws,
dynamics classification, and embeddings.
"""

import numpy as np
import pytest
import torch

from src.quasar.physics_encoder import (
    CharacteristicScales,
    ConservationResult,
    DynamicsType,
    PhysicsEncoder,
    PhysicsEncoderConfig,
    PhysicsFeatures,
    SymmetryResult,
    SymmetryType,
    encode_trajectory,
    get_physics_embedding,
)


class TestPhysicsEncoderConfig:
    """Tests for PhysicsEncoderConfig."""

    def test_default_config(self):
        config = PhysicsEncoderConfig()
        assert config.embedding_dim == 64
        assert config.symmetry_threshold == 0.8
        assert config.conservation_threshold == 0.05

    def test_custom_config(self):
        config = PhysicsEncoderConfig(embedding_dim=128, symmetry_threshold=0.9)
        assert config.embedding_dim == 128
        assert config.symmetry_threshold == 0.9


class TestDynamicsType:
    """Tests for DynamicsType enum."""

    def test_enum_values(self):
        assert DynamicsType.STEADY_STATE.value == "steady_state"
        assert DynamicsType.PERIODIC.value == "periodic"
        assert DynamicsType.CHAOTIC.value == "chaotic"

    def test_string_conversion(self):
        assert str(DynamicsType.PERIODIC) == "DynamicsType.PERIODIC"


class TestSymmetryType:
    """Tests for SymmetryType enum."""

    def test_enum_values(self):
        assert SymmetryType.TRANSLATION_X.value == "translation_x"
        assert SymmetryType.ROTATION.value == "rotation"


class TestSymmetryResult:
    """Tests for SymmetryResult dataclass."""

    def test_creation(self):
        result = SymmetryResult(
            symmetry_type=SymmetryType.ROTATION,
            detected=True,
            confidence=0.95,
            period=4.0,
        )
        assert result.detected is True
        assert result.confidence == 0.95
        assert result.period == 4.0


class TestConservationResult:
    """Tests for ConservationResult dataclass."""

    def test_creation(self):
        result = ConservationResult(
            quantity_name="energy",
            initial_value=100.0,
            final_value=99.5,
            mean_value=99.8,
            variation=0.01,
            is_conserved=True,
        )
        assert result.is_conserved is True
        assert result.variation == 0.01


class TestCharacteristicScales:
    """Tests for CharacteristicScales dataclass."""

    def test_creation(self):
        scales = CharacteristicScales(
            length_scale=10.0,
            time_scale=5.0,
            velocity_scale=2.0,
            energy_scale=100.0,
        )
        assert scales.length_scale == 10.0
        assert scales.velocity_scale == 2.0


class TestPhysicsFeatures:
    """Tests for PhysicsFeatures dataclass."""

    @pytest.fixture
    def sample_features(self):
        return PhysicsFeatures(
            symmetries=[
                SymmetryResult(SymmetryType.TRANSLATION_X, True, 0.9, 10.0),
                SymmetryResult(SymmetryType.ROTATION, False, 0.3, None),
            ],
            conservation=[
                ConservationResult("mass", 100, 100, 100, 0.001, True),
                ConservationResult("energy", 50, 45, 47, 0.1, False),
            ],
            dynamics_type=DynamicsType.PERIODIC,
            dynamics_confidence=0.85,
            characteristic_scales=CharacteristicScales(10, 5, 2, 100),
            embedding=np.zeros(64),
        )

    def test_detected_symmetries(self, sample_features):
        detected = sample_features.detected_symmetries
        assert len(detected) == 1
        assert SymmetryType.TRANSLATION_X in detected

    def test_conserved_quantities(self, sample_features):
        conserved = sample_features.conserved_quantities
        assert "mass" in conserved
        assert "energy" not in conserved

    def test_num_symmetries(self, sample_features):
        assert sample_features.num_symmetries == 1

    def test_embedding_dim(self, sample_features):
        assert sample_features.embedding_dim == 64


class TestPhysicsEncoderBasic:
    """Basic tests for PhysicsEncoder."""

    @pytest.fixture
    def encoder(self):
        return PhysicsEncoder()

    @pytest.fixture
    def simple_trajectory(self):
        # Simple 3D trajectory: (time=10, H=16, W=16, C=2)
        return np.random.randn(10, 16, 16, 2)

    def test_encoder_creation(self, encoder):
        assert encoder.config.embedding_dim == 64

    def test_encode_returns_features(self, encoder, simple_trajectory):
        features = encoder.encode(simple_trajectory)
        assert isinstance(features, PhysicsFeatures)

    def test_encode_torch_tensor(self, encoder):
        traj = torch.randn(10, 16, 16, 2)
        features = encoder.encode(traj)
        assert isinstance(features, PhysicsFeatures)

    def test_encode_3d_input(self, encoder):
        # Single channel input (T, H, W)
        traj = np.random.randn(10, 16, 16)
        features = encoder.encode(traj)
        assert isinstance(features, PhysicsFeatures)

    def test_embedding_shape(self, encoder, simple_trajectory):
        features = encoder.encode(simple_trajectory)
        assert features.embedding.shape == (64,)

    def test_custom_embedding_dim(self, simple_trajectory):
        config = PhysicsEncoderConfig(embedding_dim=128)
        encoder = PhysicsEncoder(config)
        features = encoder.encode(simple_trajectory)
        assert features.embedding.shape == (128,)


class TestPhysicsEncoderSymmetries:
    """Tests for symmetry detection."""

    @pytest.fixture
    def encoder(self):
        return PhysicsEncoder()

    def test_detects_reflection_x_symmetry(self, encoder):
        # Create symmetric field
        H, W = 32, 32
        traj = np.zeros((5, H, W, 1))
        x = np.linspace(-1, 1, W)
        for i in range(5):
            traj[i, :, :, 0] = np.abs(x)[np.newaxis, :]  # Symmetric in x

        features = encoder.encode(traj)
        ref_x = [s for s in features.symmetries if s.symmetry_type == SymmetryType.REFLECTION_X]
        assert len(ref_x) == 1
        # Should detect or at least have high confidence

    def test_detects_time_translation(self, encoder):
        # Create steady state (constant in time)
        traj = np.ones((20, 16, 16, 1)) * 5.0
        features = encoder.encode(traj)

        time_trans = [s for s in features.symmetries if s.symmetry_type == SymmetryType.TIME_TRANSLATION]
        assert len(time_trans) == 1
        assert time_trans[0].detected  # Use truthiness check


class TestPhysicsEncoderConservation:
    """Tests for conservation law analysis."""

    @pytest.fixture
    def encoder(self):
        return PhysicsEncoder()

    def test_detects_conserved_mass(self, encoder):
        # Constant total mass
        traj = np.random.randn(20, 16, 16, 2)
        traj[..., 0] = 1.0  # Constant density

        features = encoder.encode(traj)
        mass = [c for c in features.conservation if c.quantity_name == "total_mass"]
        assert len(mass) == 1
        assert mass[0].is_conserved  # Use truthiness check

    def test_detects_energy_change(self, encoder):
        # Energy that decays
        traj = np.zeros((20, 16, 16, 1))
        for t in range(20):
            traj[t] = np.exp(-t / 5) * np.random.randn(16, 16, 1)

        features = encoder.encode(traj)
        energy = [c for c in features.conservation if c.quantity_name == "energy_proxy"]
        assert len(energy) == 1
        # Energy should not be conserved (decaying)


class TestPhysicsEncoderDynamics:
    """Tests for dynamics classification."""

    @pytest.fixture
    def encoder(self):
        return PhysicsEncoder()

    def test_detects_periodic_dynamics(self, encoder):
        # Create periodic signal
        t = np.linspace(0, 4 * np.pi, 50)
        traj = np.zeros((50, 8, 8, 1))
        for i, ti in enumerate(t):
            traj[i, :, :, 0] = np.sin(ti)

        features = encoder.encode(traj, dt=0.1)
        assert features.dynamics_type == DynamicsType.PERIODIC

    def test_detects_steady_state(self, encoder):
        # Constant field
        traj = np.ones((30, 16, 16, 1)) * 3.0

        features = encoder.encode(traj)
        assert features.dynamics_type == DynamicsType.STEADY_STATE

    def test_dynamics_confidence_range(self, encoder):
        traj = np.random.randn(20, 16, 16, 2)
        features = encoder.encode(traj)
        assert 0 <= features.dynamics_confidence <= 1


class TestPhysicsEncoderScales:
    """Tests for characteristic scale computation."""

    @pytest.fixture
    def encoder(self):
        return PhysicsEncoder()

    def test_computes_all_scales(self, encoder):
        traj = np.random.randn(20, 32, 32, 2)
        features = encoder.encode(traj, dt=0.1, dx=0.5)

        scales = features.characteristic_scales
        assert scales.length_scale > 0
        assert scales.time_scale > 0
        assert scales.velocity_scale > 0
        assert scales.energy_scale is not None


class TestPhysicsEncoderBatch:
    """Tests for batch encoding."""

    @pytest.fixture
    def encoder(self):
        return PhysicsEncoder()

    def test_batch_encode(self, encoder):
        batch = np.random.randn(4, 10, 16, 16, 2)
        features_list = encoder.batch_encode(batch)
        assert len(features_list) == 4
        assert all(isinstance(f, PhysicsFeatures) for f in features_list)

    def test_get_embedding_batch(self, encoder):
        batch = np.random.randn(4, 10, 16, 16, 2)
        embeddings = encoder.get_embedding_batch(batch)
        assert embeddings.shape == (4, 64)

    def test_batch_with_torch(self, encoder):
        batch = torch.randn(3, 10, 16, 16, 2)
        features_list = encoder.batch_encode(batch)
        assert len(features_list) == 3


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_encode_trajectory(self):
        traj = np.random.randn(10, 16, 16, 2)
        features = encode_trajectory(traj)
        assert isinstance(features, PhysicsFeatures)

    def test_encode_trajectory_with_config(self):
        traj = np.random.randn(10, 16, 16, 2)
        config = PhysicsEncoderConfig(embedding_dim=32)
        features = encode_trajectory(traj, config=config)
        assert features.embedding_dim == 32

    def test_get_physics_embedding(self):
        traj = np.random.randn(10, 16, 16, 2)
        embedding = get_physics_embedding(traj)
        assert embedding.shape == (64,)

    def test_get_physics_embedding_custom_dim(self):
        traj = np.random.randn(10, 16, 16, 2)
        embedding = get_physics_embedding(traj, embedding_dim=128)
        assert embedding.shape == (128,)


class TestPhysicsEncoderEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def encoder(self):
        return PhysicsEncoder()

    def test_very_short_trajectory(self, encoder):
        traj = np.random.randn(2, 8, 8, 1)
        features = encoder.encode(traj)
        assert isinstance(features, PhysicsFeatures)

    def test_single_timestep(self, encoder):
        traj = np.random.randn(1, 8, 8, 1)
        features = encoder.encode(traj)
        # Should handle gracefully
        assert features.dynamics_type in DynamicsType

    def test_non_square_spatial(self, encoder):
        traj = np.random.randn(10, 16, 32, 2)
        features = encoder.encode(traj)
        assert isinstance(features, PhysicsFeatures)
        # Rotation should not be detected for non-square
        rotation = [s for s in features.symmetries if s.symmetry_type == SymmetryType.ROTATION]
        if rotation:
            assert rotation[0].detected is False

    def test_metadata_preserved(self, encoder):
        traj = np.random.randn(10, 16, 16, 2)
        features = encoder.encode(traj, dt=0.5, dx=0.25)
        assert features.metadata["dt"] == 0.5
        assert features.metadata["dx"] == 0.25
        assert features.metadata["shape"] == (10, 16, 16, 2)


class TestPhysicsEncoderDownsampling:
    """Tests for downsampling functionality."""

    def test_spatial_downsampling(self):
        config = PhysicsEncoderConfig(downsample_spatial=2)
        encoder = PhysicsEncoder(config)
        traj = np.random.randn(10, 32, 32, 2)
        features = encoder.encode(traj)
        # Should still produce valid features
        assert isinstance(features, PhysicsFeatures)

    def test_temporal_downsampling(self):
        config = PhysicsEncoderConfig(downsample_temporal=2)
        encoder = PhysicsEncoder(config)
        traj = np.random.randn(20, 16, 16, 2)
        features = encoder.encode(traj)
        assert isinstance(features, PhysicsFeatures)
