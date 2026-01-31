"""
Physics feature encoder for simulation trajectories.

Extracts physics-relevant features from The Well simulation data:
- Symmetries (translational, rotational, reflection)
- Conservation laws (mass, energy, momentum)
- Dynamics classification (steady, periodic, chaotic, transient)
- Characteristic scales (length, time, velocity)
- Dense embedding for ML models
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
import torch
from scipy import signal
from scipy.fft import fft, fftfreq


class DynamicsType(str, Enum):
    """Classification of dynamics behavior."""

    STEADY_STATE = "steady_state"
    PERIODIC = "periodic"
    QUASI_PERIODIC = "quasi_periodic"
    CHAOTIC = "chaotic"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


class SymmetryType(str, Enum):
    """Types of spatial/temporal symmetries."""

    TRANSLATION_X = "translation_x"
    TRANSLATION_Y = "translation_y"
    ROTATION = "rotation"
    REFLECTION_X = "reflection_x"
    REFLECTION_Y = "reflection_y"
    TIME_TRANSLATION = "time_translation"


@dataclass
class SymmetryResult:
    """Result of symmetry detection."""

    symmetry_type: SymmetryType
    detected: bool
    confidence: float  # 0-1 confidence score
    period: float | None = None  # For translational symmetries


@dataclass
class ConservationResult:
    """Result of conservation law analysis."""

    quantity_name: str
    initial_value: float
    final_value: float
    mean_value: float
    variation: float  # Relative variation (std/mean)
    is_conserved: bool  # True if variation < threshold


@dataclass
class CharacteristicScales:
    """Characteristic scales of the system."""

    length_scale: float  # Dominant spatial scale
    time_scale: float  # Dominant temporal scale
    velocity_scale: float  # Characteristic velocity
    energy_scale: float | None = None


@dataclass
class PhysicsFeatures:
    """
    Complete physics feature extraction result.

    Contains symmetries, conservation laws, dynamics type,
    characteristic scales, and a dense embedding vector.
    """

    symmetries: list[SymmetryResult]
    conservation: list[ConservationResult]
    dynamics_type: DynamicsType
    dynamics_confidence: float
    characteristic_scales: CharacteristicScales
    embedding: np.ndarray  # Dense feature vector for ML
    metadata: dict = field(default_factory=dict)

    @property
    def detected_symmetries(self) -> list[SymmetryType]:
        """Return list of detected symmetry types."""
        return [s.symmetry_type for s in self.symmetries if s.detected]

    @property
    def conserved_quantities(self) -> list[str]:
        """Return list of conserved quantity names."""
        return [c.quantity_name for c in self.conservation if c.is_conserved]

    @property
    def num_symmetries(self) -> int:
        """Return number of detected symmetries."""
        return len(self.detected_symmetries)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return len(self.embedding)


@dataclass
class PhysicsEncoderConfig:
    """Configuration for physics encoder."""

    # Symmetry detection
    symmetry_threshold: float = 0.8  # Correlation threshold for detection
    max_translation_period: int = 64  # Max period to check for translation

    # Conservation thresholds
    conservation_threshold: float = 0.05  # Relative variation threshold

    # Dynamics classification
    fft_threshold: float = 0.1  # Relative peak threshold for FFT
    chaos_entropy_threshold: float = 0.7  # Entropy threshold for chaos

    # Embedding
    embedding_dim: int = 64  # Output embedding dimension

    # Computation
    downsample_spatial: int = 1  # Spatial downsampling factor
    downsample_temporal: int = 1  # Temporal downsampling factor


class PhysicsEncoder:
    """
    Encodes physics simulation trajectories into feature vectors.

    Extracts symmetries, conservation laws, dynamics type, and
    characteristic scales from raw simulation data.

    Example:
        >>> encoder = PhysicsEncoder()
        >>> features = encoder.encode(trajectory)
        >>> print(features.dynamics_type)
        >>> print(features.embedding.shape)
    """

    def __init__(self, config: PhysicsEncoderConfig | None = None):
        """
        Initialize the physics encoder.

        Args:
            config: Encoder configuration
        """
        self.config = config or PhysicsEncoderConfig()

    def encode(
        self,
        trajectory: torch.Tensor | np.ndarray,
        dt: float = 1.0,
        dx: float = 1.0,
    ) -> PhysicsFeatures:
        """
        Extract physics features from a trajectory.

        Args:
            trajectory: Simulation data with shape (time, height, width, channels)
                       or (time, height, width) for single channel
            dt: Time step between frames
            dx: Spatial grid spacing

        Returns:
            PhysicsFeatures with all extracted information
        """
        # Convert to numpy if needed
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.numpy()

        # Ensure 4D: (time, height, width, channels)
        if trajectory.ndim == 3:
            trajectory = trajectory[..., np.newaxis]

        # Downsample if configured
        traj = self._downsample(trajectory)

        # Extract features
        symmetries = self._detect_symmetries(traj)
        conservation = self._analyze_conservation(traj, dt)
        dynamics_type, dynamics_conf = self._classify_dynamics(traj, dt)
        scales = self._compute_scales(traj, dt, dx)

        # Build embedding
        embedding = self._build_embedding(
            symmetries, conservation, dynamics_type, dynamics_conf, scales
        )

        return PhysicsFeatures(
            symmetries=symmetries,
            conservation=conservation,
            dynamics_type=dynamics_type,
            dynamics_confidence=dynamics_conf,
            characteristic_scales=scales,
            embedding=embedding,
            metadata={
                "shape": trajectory.shape,
                "dt": dt,
                "dx": dx,
            },
        )

    def _downsample(self, trajectory: np.ndarray) -> np.ndarray:
        """Downsample trajectory for efficiency."""
        t_step = self.config.downsample_temporal
        s_step = self.config.downsample_spatial

        if t_step > 1 or s_step > 1:
            return trajectory[::t_step, ::s_step, ::s_step, :]
        return trajectory

    def _detect_symmetries(self, traj: np.ndarray) -> list[SymmetryResult]:
        """
        Detect spatial and temporal symmetries.

        Uses cross-correlation and mirror comparisons.
        """
        results = []

        # Use first frame for spatial symmetries, mean for robustness
        frame = np.mean(traj, axis=0)  # (H, W, C)
        frame_mean = np.mean(frame, axis=-1)  # Average over channels

        # Translation X
        trans_x = self._detect_translation(frame_mean, axis=1)
        results.append(trans_x)

        # Translation Y
        trans_y = self._detect_translation(frame_mean, axis=0)
        results.append(
            SymmetryResult(
                symmetry_type=SymmetryType.TRANSLATION_Y,
                detected=trans_y.detected,
                confidence=trans_y.confidence,
                period=trans_y.period,
            )
        )

        # Reflection X (mirror across vertical axis)
        ref_x = self._detect_reflection(frame_mean, axis=1)
        results.append(ref_x)

        # Reflection Y (mirror across horizontal axis)
        ref_y = self._detect_reflection(frame_mean, axis=0)
        results.append(ref_y)

        # Rotation (90 degree)
        rot = self._detect_rotation(frame_mean)
        results.append(rot)

        # Time translation (steady state check)
        time_trans = self._detect_time_translation(traj)
        results.append(time_trans)

        return results

    def _detect_translation(
        self, field: np.ndarray, axis: int
    ) -> SymmetryResult:
        """Detect translational symmetry along an axis."""
        symmetry_type = (
            SymmetryType.TRANSLATION_X if axis == 1 else SymmetryType.TRANSLATION_Y
        )

        # Compute autocorrelation along axis
        n = field.shape[axis]
        max_lag = min(n // 2, self.config.max_translation_period)

        # Average over the other dimension
        profile = np.mean(field, axis=1 - axis)

        # Compute autocorrelation
        acf = np.correlate(profile - np.mean(profile), profile - np.mean(profile), mode="full")
        acf = acf[len(acf) // 2 :]  # Keep positive lags
        acf = acf / (acf[0] + 1e-10)  # Normalize

        # Find peaks (excluding zero lag)
        peaks, properties = signal.find_peaks(acf[1:max_lag], height=self.config.symmetry_threshold)

        if len(peaks) > 0:
            best_peak = peaks[0] + 1  # Add 1 because we excluded lag 0
            confidence = acf[best_peak]
            return SymmetryResult(
                symmetry_type=symmetry_type,
                detected=True,
                confidence=float(confidence),
                period=float(best_peak),
            )

        return SymmetryResult(
            symmetry_type=symmetry_type,
            detected=False,
            confidence=float(np.max(acf[1:max_lag]) if max_lag > 1 else 0),
            period=None,
        )

    def _detect_reflection(self, field: np.ndarray, axis: int) -> SymmetryResult:
        """Detect reflection symmetry."""
        symmetry_type = (
            SymmetryType.REFLECTION_X if axis == 1 else SymmetryType.REFLECTION_Y
        )

        # Flip and compare
        flipped = np.flip(field, axis=axis)
        diff = np.abs(field - flipped)
        normalized_diff = np.mean(diff) / (np.std(field) + 1e-10)

        # Low normalized difference means high symmetry
        confidence = max(0, 1 - normalized_diff)
        detected = confidence >= self.config.symmetry_threshold

        return SymmetryResult(
            symmetry_type=symmetry_type,
            detected=detected,
            confidence=float(confidence),
            period=None,
        )

    def _detect_rotation(self, field: np.ndarray) -> SymmetryResult:
        """Detect 90-degree rotational symmetry."""
        # Only works for square fields
        h, w = field.shape
        if h != w:
            return SymmetryResult(
                symmetry_type=SymmetryType.ROTATION,
                detected=False,
                confidence=0.0,
                period=None,
            )

        rotated = np.rot90(field)
        diff = np.abs(field - rotated)
        normalized_diff = np.mean(diff) / (np.std(field) + 1e-10)

        confidence = max(0, 1 - normalized_diff)
        detected = confidence >= self.config.symmetry_threshold

        return SymmetryResult(
            symmetry_type=SymmetryType.ROTATION,
            detected=detected,
            confidence=float(confidence),
            period=4.0 if detected else None,  # 4-fold symmetry
        )

    def _detect_time_translation(self, traj: np.ndarray) -> SymmetryResult:
        """Detect time translation symmetry (steady state)."""
        # Compare first and last few frames
        n_compare = max(1, traj.shape[0] // 10)
        early = np.mean(traj[:n_compare], axis=0)
        late = np.mean(traj[-n_compare:], axis=0)

        diff = np.abs(early - late)
        normalized_diff = np.mean(diff) / (np.std(traj) + 1e-10)

        confidence = max(0, 1 - normalized_diff)
        detected = confidence >= self.config.symmetry_threshold

        return SymmetryResult(
            symmetry_type=SymmetryType.TIME_TRANSLATION,
            detected=detected,
            confidence=float(confidence),
            period=None,
        )

    def _analyze_conservation(
        self, traj: np.ndarray, dt: float
    ) -> list[ConservationResult]:
        """
        Analyze conservation laws.

        Tracks total mass, energy proxy, and momentum over time.
        """
        results = []

        # Total "mass" (sum of first channel, typically density or a conserved field)
        mass = self._track_total_mass(traj)
        results.append(mass)

        # "Energy" proxy (sum of squared values)
        energy = self._track_energy_proxy(traj)
        results.append(energy)

        # Momentum X (if we have velocity-like channels)
        if traj.shape[-1] >= 2:
            mom_x = self._track_momentum(traj, channel=0)
            results.append(mom_x)
            mom_y = self._track_momentum(traj, channel=1)
            results.append(mom_y)

        return results

    def _track_total_mass(self, traj: np.ndarray) -> ConservationResult:
        """Track total mass (sum of field) over time."""
        # Sum over spatial dimensions for each timestep
        total = np.sum(traj[..., 0], axis=(1, 2))  # First channel

        initial = total[0]
        final = total[-1]
        mean = np.mean(total)
        std = np.std(total)
        variation = std / (np.abs(mean) + 1e-10)

        return ConservationResult(
            quantity_name="total_mass",
            initial_value=float(initial),
            final_value=float(final),
            mean_value=float(mean),
            variation=float(variation),
            is_conserved=variation < self.config.conservation_threshold,
        )

    def _track_energy_proxy(self, traj: np.ndarray) -> ConservationResult:
        """Track energy proxy (L2 norm squared) over time."""
        # Sum of squared values
        energy = np.sum(traj**2, axis=(1, 2, 3))

        initial = energy[0]
        final = energy[-1]
        mean = np.mean(energy)
        std = np.std(energy)
        variation = std / (np.abs(mean) + 1e-10)

        return ConservationResult(
            quantity_name="energy_proxy",
            initial_value=float(initial),
            final_value=float(final),
            mean_value=float(mean),
            variation=float(variation),
            is_conserved=variation < self.config.conservation_threshold,
        )

    def _track_momentum(self, traj: np.ndarray, channel: int) -> ConservationResult:
        """Track momentum (sum of velocity component) over time."""
        name = "momentum_x" if channel == 0 else "momentum_y"

        # Sum of velocity channel weighted by first channel (density proxy)
        if traj.shape[-1] > channel + 1:
            momentum = np.sum(traj[..., 0] * traj[..., channel + 1], axis=(1, 2))
        else:
            momentum = np.sum(traj[..., channel], axis=(1, 2))

        initial = momentum[0]
        final = momentum[-1]
        mean = np.mean(momentum)
        std = np.std(momentum)
        variation = std / (np.abs(mean) + 1e-10)

        return ConservationResult(
            quantity_name=name,
            initial_value=float(initial),
            final_value=float(final),
            mean_value=float(mean),
            variation=float(variation),
            is_conserved=variation < self.config.conservation_threshold,
        )

    def _classify_dynamics(
        self, traj: np.ndarray, dt: float
    ) -> tuple[DynamicsType, float]:
        """
        Classify dynamics type using FFT analysis.

        Returns:
            Tuple of (DynamicsType, confidence)
        """
        n_time = traj.shape[0]
        if n_time < 4:
            return DynamicsType.UNKNOWN, 0.0

        # Compute global energy over time
        energy = np.sum(traj**2, axis=(1, 2, 3))
        energy = energy - np.mean(energy)  # Remove DC

        # FFT of energy signal
        spectrum = np.abs(fft(energy))
        freqs = fftfreq(n_time, dt)

        # Only positive frequencies
        pos_mask = freqs > 0
        spectrum = spectrum[pos_mask]
        freqs = freqs[pos_mask]

        if len(spectrum) == 0:
            return DynamicsType.UNKNOWN, 0.0

        # Normalize spectrum
        spectrum = spectrum / (np.max(spectrum) + 1e-10)

        # Find peaks
        peaks, properties = signal.find_peaks(spectrum, height=self.config.fft_threshold)

        # Compute spectral entropy
        prob = spectrum / (np.sum(spectrum) + 1e-10)
        entropy = -np.sum(prob * np.log(prob + 1e-10)) / np.log(len(prob) + 1e-10)

        # Check for steady state (low variance over time)
        temporal_std = np.std(traj) / (np.mean(np.abs(traj)) + 1e-10)

        # Classify
        if temporal_std < 0.05:
            return DynamicsType.STEADY_STATE, 1.0 - temporal_std

        if len(peaks) == 0:
            # No clear peaks - either steady, transient, or chaotic
            if entropy > self.config.chaos_entropy_threshold:
                return DynamicsType.CHAOTIC, float(entropy)
            return DynamicsType.TRANSIENT, float(1 - entropy)

        if len(peaks) == 1:
            return DynamicsType.PERIODIC, float(properties["peak_heights"][0])

        if len(peaks) <= 3:
            return DynamicsType.QUASI_PERIODIC, float(np.mean(properties["peak_heights"]))

        # Many peaks with high entropy suggests chaos
        if entropy > self.config.chaos_entropy_threshold:
            return DynamicsType.CHAOTIC, float(entropy)

        return DynamicsType.QUASI_PERIODIC, float(1 - entropy)

    def _compute_scales(
        self, traj: np.ndarray, dt: float, dx: float
    ) -> CharacteristicScales:
        """Compute characteristic scales of the system."""
        # Spatial scale from autocorrelation
        frame = np.mean(traj, axis=(0, -1))  # Average over time and channels
        length_scale = self._compute_correlation_length(frame, dx)

        # Time scale from temporal autocorrelation
        point_series = traj[:, traj.shape[1] // 2, traj.shape[2] // 2, 0]
        time_scale = self._compute_correlation_time(point_series, dt)

        # Velocity scale
        velocity_scale = length_scale / (time_scale + 1e-10)

        # Energy scale (RMS of field)
        energy_scale = float(np.sqrt(np.mean(traj**2)))

        return CharacteristicScales(
            length_scale=length_scale,
            time_scale=time_scale,
            velocity_scale=velocity_scale,
            energy_scale=energy_scale,
        )

    def _compute_correlation_length(self, field: np.ndarray, dx: float) -> float:
        """Compute correlation length from 2D field."""
        h, w = field.shape
        center_h, center_w = h // 2, w // 2

        # Radial profile of autocorrelation
        f_fft = np.fft.fft2(field - np.mean(field))
        acf = np.real(np.fft.ifft2(f_fft * np.conj(f_fft)))
        acf = np.fft.fftshift(acf)
        acf = acf / (acf[center_h, center_w] + 1e-10)

        # Compute radial average
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)
        r_int = r.astype(int)

        max_r = min(center_h, center_w)
        radial_profile = np.array(
            [np.mean(acf[r_int == i]) for i in range(max_r) if np.sum(r_int == i) > 0]
        )

        # Find e-folding length (where correlation drops to 1/e)
        threshold = 1 / np.e
        below_threshold = np.where(radial_profile < threshold)[0]

        if len(below_threshold) > 0:
            return float(below_threshold[0] * dx)
        return float(max_r * dx)

    def _compute_correlation_time(self, series: np.ndarray, dt: float) -> float:
        """Compute correlation time from 1D time series."""
        n = len(series)
        if n < 4:
            return dt

        series = series - np.mean(series)
        acf = np.correlate(series, series, mode="full")
        acf = acf[len(acf) // 2 :]
        acf = acf / (acf[0] + 1e-10)

        # Find e-folding time
        threshold = 1 / np.e
        below_threshold = np.where(acf < threshold)[0]

        if len(below_threshold) > 0:
            return float(below_threshold[0] * dt)
        return float(n * dt)

    def _build_embedding(
        self,
        symmetries: list[SymmetryResult],
        conservation: list[ConservationResult],
        dynamics_type: DynamicsType,
        dynamics_conf: float,
        scales: CharacteristicScales,
    ) -> np.ndarray:
        """
        Build a dense embedding vector from extracted features.

        The embedding has fixed size regardless of input.
        """
        features = []

        # Symmetry features (6 symmetries x 2 values each = 12)
        for sym in symmetries:
            features.append(float(sym.detected))
            features.append(sym.confidence)

        # Pad if fewer symmetries
        while len(features) < 12:
            features.extend([0.0, 0.0])

        # Conservation features (4 quantities x 2 values = 8)
        for cons in conservation[:4]:
            features.append(float(cons.is_conserved))
            features.append(min(1.0, cons.variation * 10))  # Scaled variation

        # Pad if fewer conservation laws
        while len(features) < 20:
            features.extend([0.0, 0.5])

        # Dynamics type (one-hot, 5 types + unknown = 6)
        dynamics_onehot = [0.0] * 6
        type_idx = {
            DynamicsType.STEADY_STATE: 0,
            DynamicsType.PERIODIC: 1,
            DynamicsType.QUASI_PERIODIC: 2,
            DynamicsType.CHAOTIC: 3,
            DynamicsType.TRANSIENT: 4,
            DynamicsType.UNKNOWN: 5,
        }
        dynamics_onehot[type_idx[dynamics_type]] = 1.0
        features.extend(dynamics_onehot)
        features.append(dynamics_conf)

        # Scales (4 values, log-scaled for stability)
        features.append(np.log1p(scales.length_scale))
        features.append(np.log1p(scales.time_scale))
        features.append(np.log1p(scales.velocity_scale))
        features.append(np.log1p(scales.energy_scale or 0))

        # Current: 12 + 8 + 6 + 1 + 4 = 31 features
        # Pad to embedding_dim
        features = np.array(features, dtype=np.float32)
        if len(features) < self.config.embedding_dim:
            features = np.pad(
                features, (0, self.config.embedding_dim - len(features))
            )
        elif len(features) > self.config.embedding_dim:
            features = features[: self.config.embedding_dim]

        return features

    def batch_encode(
        self,
        trajectories: torch.Tensor | np.ndarray,
        dt: float = 1.0,
        dx: float = 1.0,
    ) -> list[PhysicsFeatures]:
        """
        Encode a batch of trajectories.

        Args:
            trajectories: Batch of trajectories (batch, time, height, width, channels)
            dt: Time step
            dx: Spatial grid spacing

        Returns:
            List of PhysicsFeatures for each trajectory
        """
        if isinstance(trajectories, torch.Tensor):
            trajectories = trajectories.numpy()

        return [self.encode(traj, dt, dx) for traj in trajectories]

    def get_embedding_batch(
        self,
        trajectories: torch.Tensor | np.ndarray,
        dt: float = 1.0,
        dx: float = 1.0,
    ) -> np.ndarray:
        """
        Get embeddings for a batch of trajectories.

        Args:
            trajectories: Batch of trajectories (batch, time, height, width, channels)
            dt: Time step
            dx: Spatial grid spacing

        Returns:
            Embeddings array of shape (batch, embedding_dim)
        """
        features_list = self.batch_encode(trajectories, dt, dx)
        return np.stack([f.embedding for f in features_list])


def encode_trajectory(
    trajectory: torch.Tensor | np.ndarray,
    dt: float = 1.0,
    dx: float = 1.0,
    config: PhysicsEncoderConfig | None = None,
) -> PhysicsFeatures:
    """
    Convenience function to encode a single trajectory.

    Args:
        trajectory: Simulation data (time, height, width, channels)
        dt: Time step
        dx: Spatial grid spacing
        config: Optional encoder configuration

    Returns:
        PhysicsFeatures with extracted information
    """
    encoder = PhysicsEncoder(config)
    return encoder.encode(trajectory, dt, dx)


def get_physics_embedding(
    trajectory: torch.Tensor | np.ndarray,
    dt: float = 1.0,
    dx: float = 1.0,
    embedding_dim: int = 64,
) -> np.ndarray:
    """
    Convenience function to get physics embedding vector.

    Args:
        trajectory: Simulation data (time, height, width, channels)
        dt: Time step
        dx: Spatial grid spacing
        embedding_dim: Desired embedding dimension

    Returns:
        Dense embedding vector of shape (embedding_dim,)
    """
    config = PhysicsEncoderConfig(embedding_dim=embedding_dim)
    encoder = PhysicsEncoder(config)
    features = encoder.encode(trajectory, dt, dx)
    return features.embedding
