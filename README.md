# Structure-from-Motion (SfM) — Minimal Incremental Demo

A small, calibrated, incremental SfM demo that:
- Detects and matches features between two images
- Estimates relative camera pose with an essential matrix
- Triangulates a sparse 3D point cloud
- Visualizes cameras and points in an interactive viewer

## Prerequisites
- Python 3.10+ recommended
- OS: Windows/macOS/Linux (GUI required for the viewer)

## Quickstart
1) Create and activate a virtual environment
   - Windows (PowerShell):
     - `python -m venv .venv`
     - `.\.venv\Scripts\Activate.ps1`
2) Install dependencies
   - `pip install -r requirements.txt`
   - Note: SIFT requires the “contrib” build of OpenCV (opencv-contrib-python).

3) Run the demo
   - `cd src`
   - `python main.py`
   - A window opens showing the recovered camera frusta and sparse point cloud.

## Project structure
- `src/main.py` — Orchestrates the pipeline and viewer
- `src/feature_extractor.py` — SIFT/ORB/AKAZE/BRISK wrapper
- `src/feature_matcher.py` — BF/FLANN matcher wrapper
- `src/utils.py` — Pose recovery, triangulation, intrinsics utility
- `src/image_data.py` — Per‑image data container
- `src/point_data.py` — Triangulated 3D point container
- `src/point_cloud_viewer.py` — VisPy + PyQt5 visualization
- `dataset/` — Example image pairs


## Credits
Dataset references:
- https://github.com/erik-norlin/Structure-From-Motion
- https://www.cs.cornell.edu/projects/bigsfm/
- https://huggingface.co/datasets/whc/fastmap_sfm/blob/main/README.md
- https://opensfm.org/docs/index.html

