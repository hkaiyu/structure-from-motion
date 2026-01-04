# Structure-from-Motion (SfM) — Minimal Incremental Demo
![demo](https://github.com/user-attachments/assets/c899c063-752d-4e28-abc3-6d2214de0df5)

A small, calibrated, incremental SfM demo that:

- Detects and matches features between two images
- Estimates relative camera pose with an essential matrix
- Triangulates a sparse 3D point cloud
- Visualizes cameras and points in an interactive viewer

## Prerequisites
- Python 3.10+ recommended
- OS: Windows/macOS/Linux (GUI required for the viewer)

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

