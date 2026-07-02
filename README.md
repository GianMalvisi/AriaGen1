# aria_pylib

End-to-end Python framework for **gaze-driven Segmentation and 3D Object Reconstruction** from [Project Aria](https://www.projectaria.com/) Gen 1 egocentric recordings.

Developed as part of a research internship at the **University of Milano-Bicocca**, Department of Computer Science, Systematics and Communication.

## Overview

`aria_pylib` automates the complete path from a raw Aria VRS recording to isolated 3D object meshes, using eye-tracking data to drive the segmentation process without manual annotation:

1. **Data Loading** — VRS multi-modal stream access (RGB, SLAM, IMU, ET), frame extraction at configurable FPS.
2. **Sensor Synchronization** — Temporal alignment between visual frames, inertial sensors, and eye-tracking data.
3. **Gaze Processing** — MPS eye-gaze alignment, CPF-to-RGB projection via device calibration, I-DT fixation detection.
4. **2D Segmentation** — SAM2 multi-run mono-prompt mask generation from detected fixations.
5. **Preprocessing** — Distortion border cropping and COLMAP pose estimation via `ns-process-data`.
6. **3D Reconstruction** — Nerfacto training and Poisson mesh export via nerfstudio.
7. **Semantic Projection** — Mask-to-mesh projection with pinhole + OPENCV distortion model and configurable supermajority voting.
8. **Object Isolation** — Sub-mesh extraction with face remapping, preserving original vertex colors and topology.

## Tech Stack

- **Language:** Python 3.10+
- **Core Library:** `projectaria-tools` (Meta)
- **3D Reconstruction:** nerfstudio, COLMAP
- **Segmentation:** SAM2 (Segment Anything Model 2)
- **Mathematical Engine:** NumPy, PyTorch
- **Mesh Processing:** plyfile, Open3D
- **Visualization:** Matplotlib

## Installation

### Prerequisites

- Windows 10/11 or Linux
- Python 3.10+
- CUDA-capable GPU (recommended for SAM2 and nerfstudio)

### Setup

```bash
git clone https://github.com/gianMalvisi/AriaGen1.git
cd AriaGen1
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux

pip install "projectaria-tools[all]"
pip install -e .
```

For nerfstudio and SAM2 integration:

```bash
pip install -e ".[nerfstudio,sam2]"
```

## Package Structure

```
aria_pylib/
├── __init__.py         — Public API exports
├── loaders.py          — AriaDataset, AriaMultistreamDataset, frame extraction
├── sync.py             — Stream synchronization and temporal alignment
├── utils.py            — Recording inspection and file discovery
├── gaze.py             — Gaze projection, alignment, fixation detection
├── segmentation.py     — SAM2 inference and palettized mask I/O
├── preprocessing.py    — Image cropping and COLMAP integration
├── mesh.py             — PLY I/O and sub-mesh extraction
├── projection.py       — Camera projection and majority voting
└── nerfstudio.py       — Training, export, and pipeline loading helpers
```

## Quick Start

```python
from aria_pylib.loaders import extract_frames
from aria_pylib.gaze import align_gaze_to_rgb, detect_fixations_idt, fixations_to_prompts
from aria_pylib.segmentation import run_multi_prompt, merge_masks, export_masks
from aria_pylib.preprocessing import apply_crop, run_colmap
from aria_pylib.mesh import load_ply, extract_submesh
from aria_pylib.projection import project_masks_onto_mesh, majority_vote
from aria_pylib.nerfstudio import load_pipeline, export_mesh
```

## Project Structure

```
AriaGen1/
├── aria_pylib/         — Installable Python package (this library)
├── notebooks/          — Jupyter notebooks for exploration and prototyping
├── data/
│   ├── raw/            — VRS recordings and MPS outputs
│   └── outputs/        — Extracted frames, masks, nerfstudio models
├── checkpoints/        — SAM2 model weights
├── pyproject.toml      — Package build configuration
└── README.md
```

## Development Roadmap

### Phase 0: Environment & Documentation
Configuration of the Python environment, `projectaria-tools` validation, and study of Aria/nerfstudio documentation.

### Phase 1: Stream Analysis & Validation
Inspection of `.vrs` multi-modal files to understand stream structures (RGB, SLAM, Audio, IMU, Eye Tracking).
    
### Phase 2: Data Extraction Pipeline
Single-camera and multi-stream frame extraction with temporal synchronization across sensors.

### Phase 3: 2D Segmentation (SAM2 / SAM3)
Evaluation of SAM2 and SAM3 for video object segmentation from egocentric recordings. Comparison of single-run multi-prompt vs multi-run mono-prompt strategies.

### Phase 4: 3D Reconstruction & Semantic Projection
Integration with nerfstudio for NeRF-based scene reconstruction. Semantic mask projection onto exported meshes via majority voting. Sub-mesh isolation of segmented objects.

### Phase 5: End-to-End Automated Pipeline
Fully automated pipeline from VRS recording to isolated 3D objects using eye-tracking fixation detection as the sole input for segmentation prompts.

## Contacts

**Gianluca Malvisi** — [g.malvisi04@gmail.com](mailto:g.malvisi04@gmail.com)
- University of Milano-Bicocca, Department of Computer Science, Systematics and Communication
- GitHub: [gianMalvisi](https://github.com/gianMalvisi)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
