"""
The Well data loader for physics simulations.

This module provides a PyTorch-compatible interface to The Well datasets,
enabling efficient streaming of physics simulation trajectories for
training physics-augmented models.

The Well is a 15TB collection of physics simulations from PolymathicAI,
containing datasets across fluid dynamics, MHD, biological systems, and more.

Reference: https://github.com/PolymathicAI/the_well
"""

from dataclasses import dataclass, field
from typing import Iterator, Literal

import torch
from torch.utils.data import DataLoader

from the_well.data import WELL_DATASETS, WellDataset


@dataclass
class TrajectoryBatch:
    """
    Container for a batch of physics simulation trajectories.

    Attributes:
        input_fields: Input state fields, shape (batch, time, height, width, channels)
        output_fields: Output/next state fields, shape (batch, time, height, width, channels)
        scalars: Constant simulation parameters, shape (batch, num_scalars)
        space_grid: Spatial coordinates, shape (height, width, dims) or (batch, height, width, dims)
        input_times: Input timesteps, shape (batch, input_steps)
        output_times: Output timesteps, shape (batch, output_steps)
        metadata: Additional metadata dict
    """

    input_fields: torch.Tensor
    output_fields: torch.Tensor
    scalars: torch.Tensor
    space_grid: torch.Tensor
    input_times: torch.Tensor
    output_times: torch.Tensor
    metadata: dict = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        """Return the batch size."""
        return self.input_fields.shape[0]

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Return the spatial dimensions (H, W) or (H, W, D)."""
        return tuple(self.input_fields.shape[2:-1])

    @property
    def num_channels(self) -> int:
        """Return the number of field channels."""
        return self.input_fields.shape[-1]

    @property
    def input_steps(self) -> int:
        """Return number of input timesteps."""
        return self.input_fields.shape[1]

    @property
    def output_steps(self) -> int:
        """Return number of output timesteps."""
        return self.output_fields.shape[1]

    def to(self, device: torch.device | str) -> "TrajectoryBatch":
        """Move all tensors to the specified device."""
        return TrajectoryBatch(
            input_fields=self.input_fields.to(device),
            output_fields=self.output_fields.to(device),
            scalars=self.scalars.to(device),
            space_grid=self.space_grid.to(device),
            input_times=self.input_times.to(device),
            output_times=self.output_times.to(device),
            metadata=self.metadata,
        )


@dataclass
class WellConfig:
    """
    Configuration for The Well data loader.

    Attributes:
        dataset_name: Name of the Well dataset (e.g., 'shear_flow', 'MHD_64')
        split: Data split ('train', 'valid', 'test')
        base_path: Base path for local data or 'hf://datasets/polymathic-ai/' for streaming
        batch_size: Batch size for DataLoader
        n_steps_input: Number of input timesteps per sample
        n_steps_output: Number of output timesteps per sample
        num_workers: Number of DataLoader workers
        pin_memory: Whether to pin memory for GPU transfer
        restrict_samples: Limit samples (int for count, float for fraction, None for all)
        use_normalization: Whether to normalize field values
        cache_small: Whether to cache small tensors in memory
        shuffle: Whether to shuffle data (only for train split by default)
    """

    dataset_name: str = "shear_flow"
    split: Literal["train", "valid", "test"] = "train"
    base_path: str = "hf://datasets/polymathic-ai/"
    batch_size: int = 4
    n_steps_input: int = 1
    n_steps_output: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    restrict_samples: int | float | None = None
    use_normalization: bool = False
    cache_small: bool = True
    shuffle: bool | None = None  # None = auto (True for train)


class WellLoader:
    """
    Data loader for The Well physics simulation datasets.

    Provides a PyTorch-compatible interface for loading and iterating
    over physics simulation trajectories from The Well dataset collection.

    Example:
        >>> config = WellConfig(dataset_name='shear_flow', split='train')
        >>> loader = WellLoader(config)
        >>> for batch in loader:
        ...     print(batch.input_fields.shape)  # (batch, time, H, W, channels)

    For streaming from Hugging Face (no local download required):
        >>> config = WellConfig(
        ...     dataset_name='shear_flow',
        ...     base_path='hf://datasets/polymathic-ai/',
        ... )
    """

    def __init__(self, config: WellConfig | None = None, **kwargs):
        """
        Initialize the Well data loader.

        Args:
            config: WellConfig instance
            **kwargs: Override config values
        """
        if config is None:
            config = WellConfig(**kwargs)
        else:
            # Apply any kwargs overrides
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        self.config = config
        self._validate_config()

        # Build the underlying WellDataset
        self._dataset = self._create_dataset()
        self._dataloader = self._create_dataloader()

    def _validate_config(self) -> None:
        """Validate the configuration."""
        if self.config.dataset_name not in WELL_DATASETS:
            available = ", ".join(sorted(WELL_DATASETS)[:10])
            raise ValueError(
                f"Unknown dataset: {self.config.dataset_name}. "
                f"Available: {available}... ({len(WELL_DATASETS)} total)"
            )

        if self.config.split not in ("train", "valid", "test"):
            raise ValueError(f"Invalid split: {self.config.split}")

        if self.config.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.config.batch_size}")

        if self.config.n_steps_input < 1:
            raise ValueError(f"n_steps_input must be >= 1")

        if self.config.n_steps_output < 1:
            raise ValueError(f"n_steps_output must be >= 1")

    def _create_dataset(self) -> WellDataset:
        """Create the underlying WellDataset."""
        return WellDataset(
            well_base_path=self.config.base_path,
            well_dataset_name=self.config.dataset_name,
            well_split_name=self.config.split,
            n_steps_input=self.config.n_steps_input,
            n_steps_output=self.config.n_steps_output,
            use_normalization=self.config.use_normalization,
            cache_small=self.config.cache_small,
            restrict_num_samples=self.config.restrict_samples,
            return_grid=True,
        )

    def _create_dataloader(self) -> DataLoader:
        """Create the PyTorch DataLoader."""
        shuffle = self.config.shuffle
        if shuffle is None:
            shuffle = self.config.split == "train"

        return DataLoader(
            self._dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self._collate_batch,
        )

    def _collate_batch(self, samples: list[dict]) -> TrajectoryBatch:
        """
        Collate samples into a TrajectoryBatch.

        Args:
            samples: List of sample dicts from WellDataset

        Returns:
            TrajectoryBatch with stacked tensors
        """
        # Stack all tensors along batch dimension
        input_fields = torch.stack([s["input_fields"] for s in samples])
        output_fields = torch.stack([s["output_fields"] for s in samples])
        scalars = torch.stack([s["constant_scalars"] for s in samples])
        input_times = torch.stack([s["input_time_grid"] for s in samples])
        output_times = torch.stack([s["output_time_grid"] for s in samples])

        # Space grid is typically the same for all samples
        space_grid = samples[0]["space_grid"]

        return TrajectoryBatch(
            input_fields=input_fields,
            output_fields=output_fields,
            scalars=scalars,
            space_grid=space_grid,
            input_times=input_times,
            output_times=output_times,
            metadata={
                "dataset_name": self.config.dataset_name,
                "split": self.config.split,
            },
        )

    def __iter__(self) -> Iterator[TrajectoryBatch]:
        """Iterate over batches."""
        return iter(self._dataloader)

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self._dataloader)

    @property
    def num_samples(self) -> int:
        """Return total number of samples."""
        return len(self._dataset)

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return self.config.dataset_name

    @property
    def split(self) -> str:
        """Return the data split."""
        return self.config.split

    def get_sample(self, idx: int) -> TrajectoryBatch:
        """
        Get a single sample by index.

        Args:
            idx: Sample index

        Returns:
            TrajectoryBatch with batch_size=1
        """
        if idx < 0 or idx >= len(self._dataset):
            raise IndexError(f"Index {idx} out of range [0, {len(self._dataset)})")

        sample = self._dataset[idx]
        return self._collate_batch([sample])

    def get_info(self) -> dict:
        """
        Get information about the loaded dataset.

        Returns:
            Dict with dataset info including shapes, splits, etc.
        """
        sample = self._dataset[0]
        return {
            "dataset_name": self.config.dataset_name,
            "split": self.config.split,
            "num_samples": len(self._dataset),
            "num_batches": len(self),
            "batch_size": self.config.batch_size,
            "input_shape": list(sample["input_fields"].shape),
            "output_shape": list(sample["output_fields"].shape),
            "num_scalars": sample["constant_scalars"].shape[0],
            "space_grid_shape": list(sample["space_grid"].shape),
            "n_steps_input": self.config.n_steps_input,
            "n_steps_output": self.config.n_steps_output,
        }


def list_datasets() -> list[str]:
    """
    List all available Well datasets.

    Returns:
        Sorted list of dataset names
    """
    return sorted(WELL_DATASETS)


def get_dataset_info(dataset_name: str) -> dict:
    """
    Get basic information about a dataset without loading it.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dict with dataset metadata

    Note:
        This provides basic info. For detailed info after loading,
        use WellLoader.get_info().
    """
    if dataset_name not in WELL_DATASETS:
        available = ", ".join(sorted(WELL_DATASETS)[:5])
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {available}..."
        )

    # Known approximate sizes for common datasets (in GB)
    # These are estimates from The Well documentation
    SIZE_ESTIMATES = {
        "shear_flow": 50,
        "rayleigh_benard": 40,
        "MHD_64": 25,
        "MHD_256": 200,
        "active_matter": 20,
        "euler_multi_quadrants_openBC": 500,
        "euler_multi_quadrants_periodicBC": 500,
        "gray_scott_reaction_diffusion": 15,
        "turbulent_radiative_layer_2D": 30,
        "turbulent_radiative_layer_3D": 300,
    }

    # Known physics types
    PHYSICS_TYPES = {
        "shear_flow": "fluid_dynamics",
        "rayleigh_benard": "fluid_dynamics",
        "rayleigh_benard_uniform": "fluid_dynamics",
        "rayleigh_taylor_instability": "fluid_dynamics",
        "MHD_64": "magnetohydrodynamics",
        "MHD_256": "magnetohydrodynamics",
        "active_matter": "biological",
        "euler_multi_quadrants_openBC": "fluid_dynamics",
        "euler_multi_quadrants_periodicBC": "fluid_dynamics",
        "gray_scott_reaction_diffusion": "reaction_diffusion",
        "turbulent_radiative_layer_2D": "astrophysics",
        "turbulent_radiative_layer_3D": "astrophysics",
        "turbulence_gravity_cooling": "astrophysics",
        "supernova_explosion_64": "astrophysics",
        "supernova_explosion_128": "astrophysics",
        "post_neutron_star_merger": "astrophysics",
        "convective_envelope_rsg": "astrophysics",
        "helmholtz_staircase": "fluid_dynamics",
        "planetswe": "planetary",
        "viscoelastic_instability": "fluid_dynamics",
        "acoustic_scattering_discontinuous": "acoustics",
        "acoustic_scattering_inclusions": "acoustics",
        "acoustic_scattering_maze": "acoustics",
    }

    return {
        "name": dataset_name,
        "available": True,
        "estimated_size_gb": SIZE_ESTIMATES.get(dataset_name, None),
        "physics_type": PHYSICS_TYPES.get(dataset_name, "unknown"),
        "huggingface_path": f"polymathic-ai/{dataset_name}",
    }


def create_streaming_loader(
    dataset_name: str,
    split: Literal["train", "valid", "test"] = "train",
    batch_size: int = 4,
    restrict_samples: int | float | None = None,
    **kwargs,
) -> WellLoader:
    """
    Convenience function to create a streaming loader from Hugging Face.

    This loads data directly from Hugging Face without requiring
    local download (but may be slower due to network I/O).

    Args:
        dataset_name: Name of the dataset
        split: Data split
        batch_size: Batch size
        restrict_samples: Limit number of samples
        **kwargs: Additional WellConfig options

    Returns:
        WellLoader configured for streaming
    """
    config = WellConfig(
        dataset_name=dataset_name,
        split=split,
        base_path="hf://datasets/polymathic-ai/",
        batch_size=batch_size,
        restrict_samples=restrict_samples,
        **kwargs,
    )
    return WellLoader(config)


def create_local_loader(
    dataset_name: str,
    base_path: str,
    split: Literal["train", "valid", "test"] = "train",
    batch_size: int = 4,
    **kwargs,
) -> WellLoader:
    """
    Convenience function to create a loader from local data.

    Requires data to be downloaded first with:
        the-well-download --base-path <base_path> --dataset <dataset_name>

    Args:
        dataset_name: Name of the dataset
        base_path: Local base path where data is stored
        split: Data split
        batch_size: Batch size
        **kwargs: Additional WellConfig options

    Returns:
        WellLoader configured for local data
    """
    config = WellConfig(
        dataset_name=dataset_name,
        split=split,
        base_path=base_path,
        batch_size=batch_size,
        **kwargs,
    )
    return WellLoader(config)
