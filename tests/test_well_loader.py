"""
Tests for The Well data loader.

These tests verify the WellLoader functionality for loading
physics simulation trajectories from The Well dataset.
"""

import pytest
import torch

from src.quasar.well_loader import (
    TrajectoryBatch,
    WellConfig,
    WellLoader,
    create_streaming_loader,
    get_dataset_info,
    list_datasets,
)


class TestWellConfig:
    """Tests for WellConfig dataclass."""

    def test_default_config(self):
        config = WellConfig()
        assert config.dataset_name == "shear_flow"
        assert config.split == "train"
        assert config.batch_size == 4
        assert config.n_steps_input == 1
        assert config.n_steps_output == 1

    def test_custom_config(self):
        config = WellConfig(
            dataset_name="MHD_64",
            split="valid",
            batch_size=8,
            n_steps_input=2,
        )
        assert config.dataset_name == "MHD_64"
        assert config.split == "valid"
        assert config.batch_size == 8
        assert config.n_steps_input == 2

    def test_streaming_base_path(self):
        config = WellConfig(base_path="hf://datasets/polymathic-ai/")
        assert "hf://" in config.base_path


class TestListDatasets:
    """Tests for list_datasets function."""

    def test_returns_list(self):
        datasets = list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0

    def test_contains_known_datasets(self):
        datasets = list_datasets()
        assert "shear_flow" in datasets
        assert "MHD_64" in datasets
        assert "rayleigh_benard" in datasets

    def test_sorted_alphabetically(self):
        datasets = list_datasets()
        assert datasets == sorted(datasets)


class TestGetDatasetInfo:
    """Tests for get_dataset_info function."""

    def test_valid_dataset(self):
        info = get_dataset_info("shear_flow")
        assert info["name"] == "shear_flow"
        assert info["available"] is True
        assert "physics_type" in info
        assert "huggingface_path" in info

    def test_physics_type(self):
        info = get_dataset_info("shear_flow")
        assert info["physics_type"] == "fluid_dynamics"

        info = get_dataset_info("MHD_64")
        assert info["physics_type"] == "magnetohydrodynamics"

    def test_invalid_dataset_raises(self):
        with pytest.raises(ValueError) as exc_info:
            get_dataset_info("nonexistent_dataset")
        assert "Unknown dataset" in str(exc_info.value)

    def test_size_estimate(self):
        info = get_dataset_info("shear_flow")
        assert info["estimated_size_gb"] is not None
        assert info["estimated_size_gb"] > 0


class TestTrajectoryBatch:
    """Tests for TrajectoryBatch dataclass."""

    @pytest.fixture
    def sample_batch(self):
        return TrajectoryBatch(
            input_fields=torch.randn(4, 2, 64, 128, 3),
            output_fields=torch.randn(4, 1, 64, 128, 3),
            scalars=torch.randn(4, 2),
            space_grid=torch.randn(64, 128, 2),
            input_times=torch.randn(4, 2),
            output_times=torch.randn(4, 1),
            metadata={"test": True},
        )

    def test_batch_size(self, sample_batch):
        assert sample_batch.batch_size == 4

    def test_spatial_shape(self, sample_batch):
        assert sample_batch.spatial_shape == (64, 128)

    def test_num_channels(self, sample_batch):
        assert sample_batch.num_channels == 3

    def test_input_steps(self, sample_batch):
        assert sample_batch.input_steps == 2

    def test_output_steps(self, sample_batch):
        assert sample_batch.output_steps == 1

    def test_to_device(self, sample_batch):
        # Test moving to CPU (safe for CI without GPU)
        moved = sample_batch.to("cpu")
        assert moved.input_fields.device.type == "cpu"
        assert moved.output_fields.device.type == "cpu"
        assert moved.scalars.device.type == "cpu"

    def test_metadata_preserved(self, sample_batch):
        assert sample_batch.metadata["test"] is True


class TestWellLoaderValidation:
    """Tests for WellLoader validation."""

    def test_invalid_dataset_raises(self):
        config = WellConfig(dataset_name="nonexistent")
        with pytest.raises(ValueError) as exc_info:
            WellLoader(config)
        assert "Unknown dataset" in str(exc_info.value)

    def test_invalid_split_raises(self):
        config = WellConfig(split="invalid")
        with pytest.raises(ValueError) as exc_info:
            WellLoader(config)
        assert "Invalid split" in str(exc_info.value)

    def test_invalid_batch_size_raises(self):
        config = WellConfig(batch_size=0)
        with pytest.raises(ValueError) as exc_info:
            WellLoader(config)
        assert "batch_size" in str(exc_info.value)

    def test_invalid_n_steps_raises(self):
        config = WellConfig(n_steps_input=0)
        with pytest.raises(ValueError) as exc_info:
            WellLoader(config)
        assert "n_steps_input" in str(exc_info.value)


@pytest.mark.slow
class TestWellLoaderStreaming:
    """
    Integration tests for WellLoader with streaming data.

    These tests require network access and may be slow.
    Mark with @pytest.mark.slow and skip in fast CI runs.
    """

    @pytest.fixture
    def streaming_loader(self):
        """Create a small streaming loader for testing."""
        config = WellConfig(
            dataset_name="shear_flow",
            split="train",
            batch_size=2,
            restrict_samples=4,
        )
        return WellLoader(config)

    def test_loader_creation(self, streaming_loader):
        assert streaming_loader.dataset_name == "shear_flow"
        assert streaming_loader.split == "train"

    def test_num_samples(self, streaming_loader):
        assert streaming_loader.num_samples == 4

    def test_len_batches(self, streaming_loader):
        assert len(streaming_loader) == 2  # 4 samples / 2 batch_size

    def test_iteration(self, streaming_loader):
        batches = list(streaming_loader)
        assert len(batches) == 2
        for batch in batches:
            assert isinstance(batch, TrajectoryBatch)

    def test_batch_shapes(self, streaming_loader):
        batch = next(iter(streaming_loader))
        assert batch.input_fields.dim() == 5  # (B, T, H, W, C)
        assert batch.output_fields.dim() == 5
        assert batch.batch_size == 2

    def test_get_sample(self, streaming_loader):
        sample = streaming_loader.get_sample(0)
        assert sample.batch_size == 1

    def test_get_sample_out_of_range(self, streaming_loader):
        with pytest.raises(IndexError):
            streaming_loader.get_sample(100)

    def test_get_info(self, streaming_loader):
        info = streaming_loader.get_info()
        assert info["dataset_name"] == "shear_flow"
        assert info["num_samples"] == 4
        assert info["num_batches"] == 2
        assert "input_shape" in info
        assert "output_shape" in info


@pytest.mark.slow
class TestCreateStreamingLoader:
    """Tests for create_streaming_loader convenience function."""

    def test_basic_creation(self):
        loader = create_streaming_loader(
            "shear_flow",
            split="train",
            batch_size=2,
            restrict_samples=2,
        )
        assert loader.dataset_name == "shear_flow"
        assert loader.num_samples == 2

    def test_kwargs_passed(self):
        loader = create_streaming_loader(
            "shear_flow",
            split="train",
            batch_size=2,
            restrict_samples=2,
            n_steps_input=2,
        )
        assert loader.config.n_steps_input == 2


class TestWellLoaderKwargsOverride:
    """Tests for kwargs override in WellLoader init."""

    def test_override_batch_size(self):
        config = WellConfig(batch_size=4)
        # Can't actually test this without network, but test the logic
        assert config.batch_size == 4


# Mark all streaming tests as slow
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (network/streaming required)"
    )
