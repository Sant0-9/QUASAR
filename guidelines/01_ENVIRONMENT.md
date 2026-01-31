# Phase 1: Environment Setup

> Set up IBM Quantum access and The Well dependencies.

---

## Step 1.1: IBM Quantum Token

**Status**: BLOCKED (requires user action)

### Actions

1. Sign up at https://quantum.cloud.ibm.com
2. Get API token from dashboard
3. Set environment variable: `export IBM_QUANTUM_TOKEN="your_token_here"`
4. Add to shell profile for persistence
5. Run verification script at `scripts/test_ibm_connection.py`

### Verification Checklist

- [ ] IBM Quantum account created
- [ ] API token obtained from dashboard
- [ ] `IBM_QUANTUM_TOKEN` environment variable set
- [ ] Token added to shell profile
- [ ] Verification script runs without error
- [ ] Shows available backends (5+)
- [ ] Can access ibm_sherbrooke (127 qubits)

### Source Files

- `scripts/test_ibm_connection.py` - Connection verification script

---

## Step 1.2: The Well Dependencies

**Status**: PENDING

### Actions

1. Add The Well package to requirements
2. Install dependencies
3. Verify data download works
4. Test data loader

### Dependencies to Add

Add to `requirements.txt`:

```
the-well
h5py
torch
torchvision
```

### Download Commands

```bash
pip install the-well
the-well-download --base-path ./data/the_well --dataset shear_flow --split train
```

### Verification Checklist

- [ ] `pip install the-well` succeeds
- [ ] `shear_flow` dataset downloads (check disk space first)
- [ ] Can load dataset in Python without error
- [ ] Data shape matches expected: `(n_traj, n_steps, height, width)`
- [ ] Memory usage acceptable for your system

### Source Files

- `requirements.txt` - Update with The Well dependencies
- `scripts/test_well_download.py` - Download verification script

---

## Troubleshooting

### IBM Token Issues

**"Token not found"**
- Check: `echo $IBM_QUANTUM_TOKEN`
- Make sure exported in current shell

**"Could not access ibm_sherbrooke"**
- Free tier may have limited access
- Check IBM Quantum dashboard for available backends

### The Well Issues

**"Download too slow"**
- Use `--split train` only first (skip val/test)
- Start with smaller dataset like `shear_flow`

**"Out of disk space"**
- Full Well is 15TB
- `shear_flow` alone is ~50GB
- Check available space: `df -h`

**"Out of memory"**
- Use streaming/batched loading
- Don't load full dataset into memory

---

## After Completion

Update `00_OVERVIEW.md`:
- Phase 1 progress: 2/2
- Mark both steps as DONE

Next: Proceed to `02_WELL.md`
