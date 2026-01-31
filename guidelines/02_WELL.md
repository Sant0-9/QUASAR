# Phase 2: The Well Integration

> Build data pipeline for The Well physics simulations.

---

## Step 2.1: Well Data Loader

**Status**: PENDING

**Prerequisites**: Phase 1 complete (The Well installed)

### Purpose

Create a PyTorch-compatible data loader that:
- Streams The Well HDF5 files efficiently
- Extracts physics trajectories
- Provides batched access for training

### Specification

**Input**: Dataset name, split name, base path

**Output**: PyTorch DataLoader yielding:
- `trajectories`: Shape `(batch, time_steps, height, width, channels)`
- `metadata`: Dict with simulation parameters

### Key Requirements

1. Use `the_well.data.WellDataset` as base
2. Support streaming (don't load all data to memory)
3. Handle train/val/test splits
4. Extract both state fields and any available metadata

### Verification Checklist

- [ ] `src/quasar/well_loader.py` created
- [ ] Can load `shear_flow` dataset
- [ ] DataLoader yields correct shapes
- [ ] Memory usage stays bounded during iteration
- [ ] Can iterate through full dataset without error
- [ ] All tests pass: `pytest tests/test_well_loader.py -v`

### Source Files

- `src/quasar/__init__.py` - Create package
- `src/quasar/well_loader.py` - Data loader implementation
- `tests/test_well_loader.py` - Unit tests

---

## Step 2.2: Physics Encoder

**Status**: PENDING

**Prerequisites**: Step 2.1 complete

### Purpose

Extract physics-relevant features from simulation trajectories:
- Symmetries (translational, rotational, etc.)
- Conservation laws (energy, momentum, mass)
- Dynamics type (periodic, chaotic, steady-state)
- Characteristic scales (length, time, velocity)

### Specification

**Input**: Trajectory tensor `(time_steps, height, width, channels)`

**Output**: Physics features dict:
- `symmetries`: List of detected symmetries
- `conservation`: Dict of conserved quantities and their variation
- `dynamics_type`: Classification string
- `characteristic_scales`: Dict of length/time/velocity scales
- `embedding`: Dense vector representation for ML

### Key Requirements

1. Use signal processing for symmetry detection
2. Track quantity changes over time for conservation
3. Use FFT/autocorrelation for dynamics classification
4. Output deterministic results for same input

### Verification Checklist

- [ ] `src/quasar/physics_encoder.py` created
- [ ] Symmetry detection works on simple test cases
- [ ] Conservation law tracking produces sensible values
- [ ] Dynamics classification distinguishes known cases
- [ ] Embedding is fixed-size dense vector
- [ ] All tests pass: `pytest tests/test_physics_encoder.py -v`

### Source Files

- `src/quasar/physics_encoder.py` - Physics feature extraction
- `tests/test_physics_encoder.py` - Unit tests

---

## Physics Feature Specifications

### Symmetries to Detect

| Symmetry | Detection Method | Example |
|----------|------------------|---------|
| Translation X | Cross-correlation | Shear flow |
| Translation Y | Cross-correlation | Shear flow |
| Rotation | Angular correlation | Vortices |
| Reflection | Mirror comparison | Many |
| Time translation | Autocorrelation | Steady state |

### Conservation Laws to Track

| Quantity | Formula | Expected Behavior |
|----------|---------|-------------------|
| Total mass | Sum of density field | Constant |
| Total energy | Kinetic + potential | May decay or constant |
| Momentum X | Sum of velocity*density | Constant or forced |
| Momentum Y | Sum of velocity*density | Constant or forced |
| Enstrophy | Integral of vorticity^2 | 2D turbulence invariant |

### Dynamics Classification

| Type | Signature | Example Dataset |
|------|-----------|-----------------|
| Steady state | Low temporal variance | Late-time equilibrium |
| Periodic | Sharp FFT peaks | Oscillating systems |
| Quasi-periodic | Multiple FFT peaks | Coupled oscillators |
| Chaotic | Broadband FFT | Turbulence |
| Transient | Decaying envelope | Relaxation |

---

## Data Format Reference

### The Well HDF5 Structure

```
dataset.hdf5
├── field_1/           # e.g., velocity_x
│   └── data           # Shape: (n_traj, n_steps, h, w)
├── field_2/           # e.g., velocity_y
│   └── data
├── metadata/
│   ├── time           # Time values for each step
│   ├── parameters     # Simulation parameters
│   └── ...
```

### Dataset Sizes (Reference)

| Dataset | Resolution | Trajectories | Estimated Size |
|---------|------------|--------------|----------------|
| shear_flow | 256x512 | 1,120 | ~50GB |
| rayleigh_benard | 512x128 | 1,750 | ~40GB |
| MHD_64 | 64^3 | 100 | ~25GB |
| euler_multi_quadrants | 512x512 | 10,000 | ~500GB |

---

## After Completion

Update `00_OVERVIEW.md`:
- Phase 2 progress: 2/2
- Mark both steps as DONE

Next: Proceed to `03_FINETUNING.md` (or `04_SURROGATE.md` in parallel)
