# AriaToolbox: Python Framework for Project Aria 1

This project focuses on developing Python-based software tools for the extraction, conversion, and organization of multi-modal sensor data acquired through **Meta Project Aria 1** glasses. The goal is to streamline the processing of raw data, making it readily available and easily integrable into **Computer Vision pipelines**.



## 📌 Project Overview
As part of a research internship at the **University of Milano-Bicocca**, this toolkit is designed to bridge the gap between raw hardware recordings and high-level computer vision analysis.

### Key Features
* **Data Extraction:** Efficiently unpack `.vrs` files to access image streams (RGB, SLAM) and sensor logs.
* **Calibration & Geometry:** Full integration with `projectaria_tools` for camera calibration and coordinate system mapping.
* **Sensor Synchronization:** Tools to align IMU, Magnetometer, and Barometer data with visual frames.
* **Format Conversion:** Exporting data into structured formats suitable for modern AI and CV models.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Core Library:** `projectaria-tools` (Meta)
* **Mathematical Engine:** NumPy, SciPy
* **Visualization:** Rerun / Matplotlib

## ⚙️ Installation & Setup

### Prerequisites
* Windows 10/11
* Python 3.10 installed

### Setup Environment
1. Clone the repository:
   ```
   git clone [https://github.com/gianMalvisi/AriaGen1.git](https://github.com/gianMalvisi/AriaGen1.git)
   ```
2. Create a virtual environment:
  ```
  python -m venv venv
  ```

3. Activate the environment
  ```
  .\venv\Scripts\activate
  ```

4. Install projectaria-tools with all optional features
  ```
  pip install "projectaria-tools[all]"
  ```

## 📂 Development Roadmap

### Phase 0: Environment Setup & Core Validation
* Configuration of the Python 3.10 virtual environment and Git repository.
* Installation of `projectaria-tools` with C++ bindings and validation of the mathematical engine.

### Phase 1: Stream Analysis & Visualization 
* Inspection of `.vrs` multi-modal files to understand stream structures (RGB, SLAM, Audio, IMU).
* Implementation of visualizers to verify recording quality and metadata integrity.

### Phase 2: Data Extraction & Unpacking 
* Development of scripts to extract raw frames and sensor logs into accessible formats.
* Management of high-bandwidth data streams (SLAM cameras at 1kHz/800Hz).

### Phase 3: Temporal & Spatial Synchronization 
* Implementation of precise temporal alignment between visual frames and inertial sensors (IMU).
* Application of camera intrinsic and extrinsic parameters for spatial coordinate mapping.

### Phase 4: CV Pipeline & Research Integration 
* Standardization of processed datasets for Computer Vision tasks.
* Development of export tools for external ML/CV framework compatibility.

* ### Phase 5: Testing, Documentation & Thesis Finalization 
* Final validation of the toolkit using diverse real-world recording scenarios.
* Comprehensive technical documentation of the developed Python framework.
* Preparation of experimental results and data analysis for the final thesis.

## 🏗️ Project Structure
* `scripts/`: Python scripts (.py) for automated data extraction and batch processing.
* `notebooks/`: Jupyter Notebooks (.ipynb) for data exploration, visualization, and prototyping.
* `docs/`: Technical documentation and coordinate system diagrams.
* `requirements.txt`: List of dependencies for the environment.
* `.gitignore`: Configured to exclude heavy binaries, VRS files, and `venv/`.

## 🤝 Contributing
This repository is part of an academic research internship. While it is primarily maintained by the author, feedback and suggestions from the research team are welcome.

## 📬 Contacts
**Gianluca Malvisi** - **Email:** g.malvisi04@gmail.com
- **Affiliation:** University of Milano-Bicocca, Department of Computer Science, Sistemistics and Communication
- **GitHub:** [gianMalvisi](https://github.com/gianMalvisi)

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
