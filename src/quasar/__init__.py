"""
QUASAR Extension: Physics-augmented quantum circuit discovery.

This package provides The Well integration, surrogate models,
and physics-informed training for accelerated circuit discovery.
"""

from src.quasar.well_loader import (
    WellConfig,
    WellLoader,
    TrajectoryBatch,
    list_datasets,
    get_dataset_info,
)

from src.quasar.physics_encoder import (
    PhysicsEncoder,
    PhysicsEncoderConfig,
    PhysicsFeatures,
    DynamicsType,
    SymmetryType,
    encode_trajectory,
    get_physics_embedding,
)

from src.quasar.augmented_data import (
    AugmentedDataConfig,
    AugmentedDataGenerator,
    ExampleType,
    TrainingExample,
    load_augmented_dataset,
    get_dataset_stats,
)

from src.quasar.circuit_encoder import (
    CircuitEncoder,
    CircuitEncoderConfig,
    CircuitFeatures,
    LearnedCircuitEncoder,
    encode_circuit,
    get_circuit_embedding,
    compute_circuit_similarity,
)

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

__all__ = [
    # Well loader
    "WellConfig",
    "WellLoader",
    "TrajectoryBatch",
    "list_datasets",
    "get_dataset_info",
    # Physics encoder
    "PhysicsEncoder",
    "PhysicsEncoderConfig",
    "PhysicsFeatures",
    "DynamicsType",
    "SymmetryType",
    "encode_trajectory",
    "get_physics_embedding",
    # Augmented data
    "AugmentedDataConfig",
    "AugmentedDataGenerator",
    "ExampleType",
    "TrainingExample",
    "load_augmented_dataset",
    "get_dataset_stats",
    # Circuit encoder
    "CircuitEncoder",
    "CircuitEncoderConfig",
    "CircuitFeatures",
    "LearnedCircuitEncoder",
    "encode_circuit",
    "get_circuit_embedding",
    "compute_circuit_similarity",
    # Surrogate
    "HamiltonianEncoder",
    "SurrogateConfig",
    "SurrogateEvaluator",
    "SurrogateModel",
    "SurrogatePrediction",
    "SurrogateTrainer",
    "SurrogateTrainingExample",
    "Trainability",
    "create_surrogate_evaluator",
]
