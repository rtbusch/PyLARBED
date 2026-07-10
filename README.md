# PyLARBED
pip install -e PyLARBED

# PyLARBED

Python tools for extracting diffracted beam intensities from Large-Angle Rocking Beam Electron Diffraction (LARBED) data collected on an EMPAD (Electron Microscope Pixel Array Detector).

LARBED is a scanning transmission electron microscopy technique used to measure precise diffracted intensities for crystal structure refinement. This repository takes raw 4D-STEM datasets (scan position × scan position × detector pixel × detector pixel), aligns and calibrates them, and integrates the intensity of each diffracted beam (reflection) into a set of 2D "rocking curves" suitable for structure factor refinement.

## What it does

- **Reads raw EMPAD detector output** (`.raw` files) into 4D NumPy arrays and cleans up NaNs/negative counts.
- **Aligns (Optional)** the diffraction pattern stack by tracking the direct beam position across scan positions.
- **Indexes reflections** on a reciprocal-space grid from user-supplied basis vectors (2 or 4 basis vectors, including HOLZ), with optional peak-adjustment to snap the grid onto observed peaks.
- **Integrates intensity** per reflection using circular or annular apertures, with optional background subtraction and Poisson/detector-noise variance propagation.
- **Deconvolves (Optional)** diffraction patterns against a measured modulation transfer function (MTF) using Lucy-Richardson deconvolution, to correct for detector point-spread.
- **Visualizes and exports** the resulting rocking-curve stack (`Store_Larbed`) as `.npy` files alongside the corresponding g-vectors, ready for downstream refinement software.

## Installation

```bash
git clone https://github.com/rtbusch/PyLARBED.git
cd PyLARBED
pip install -e .
```

This installs the `pyLARBED` package in editable mode so changes to the source are picked up immediately.

### Dependencies

- `numpy`
- `scipy`
- `matplotlib`
- `scikit-image`

These aren't yet pinned in `pyproject.toml`, so install them manually if they're not already in your environment:

```bash
pip install numpy scipy matplotlib scikit-image
```

## Basic usage

The typical workflow, as shown in the example notebooks, is:

```python
from pyLARBEDinfo import LARBEDAnalysis, LARBEDCalibration

# 1. Load and align a LARBED dataset
larbed = LARBEDAnalysis("path/to/scan_x256_y256.raw")
larbed.load_data(type=0)
larbed.average_data()
# larbed.align_data()  # optional, depending on data quality

# 2. Define reciprocal-space reflections from two (or four) basis vectors
larbed.assign_gvector((46, 48), (64, 36))
larbed.calculate_grid_vectors(crop=2, order=3, Adjust=True)

g1, g2 = (1, 3, -2), (4, 2, 0)
larbed.calculate_g_vectors(g1, g2)

# 3. Integrate each reflection into a rocking curve
larbed.IntegrateLarbed(ri=3, ro=(6, 8), g=580, m=0.47, Variance=True)
larbed.plot_larbed(ratio=0.5)

# 4. Save the result for downstream refinement
larbed.save_larbed("YIG12/Zone-245")
```

```python
# Optional: calibrate the detector MTF from a separate probe/vacuum scan,
# then deconvolve the main dataset against it
calib = LARBEDCalibration("path/to/probe_scan.raw")
calib.load_data(type=0)
calib.find_peaks(start=100, end=115)
calib.calculate_average_step_length()
calib.crop_peaks()
calib.interpolate()
calib.calculate_mtf()

larbed.deconv_Larbedstack(mtf_2d=calib.mtf_2d, varB=787, g=580, niter=10)
```

See the notebooks in the repo root (`PyLarbedConversionSTO_001.ipynb`, `PyLarbedConversionYIG_Tilt1.ipynb`, `PyLarbedConversionYIG_Tilt12.ipynb`) for complete, dataset-specific walkthroughs, including grid-vector setup, deconvolution, and plotting.

## Output

`save_larbed()` writes out:
- `<name>_Store_Larbed.npy` — the integrated intensity for each reflection, at each tilt/scan step
- `<name>_Store_LarbedVariance.npy` — the corresponding propagated variance (if computed)
- `<name>_g_vectors.npy` — the reciprocal-space g-vector for each reflection

These arrays are the inputs typically passed on to structure-refinement software.

## Status

This is an active research code repository accompanying LARBED data analysis for electron diffraction crystallography. The API is not yet stable, dependencies are not pinned, and the notebooks are dataset-specific working examples rather than polished tutorials — expect some manual adaptation (file paths, calibration parameters) for new datasets.
