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
   - go to download a local Download of COLMAP (Go to https://github.com/colmap/colmap/releases, download CUDA or non CUDA if no Nvidia GPU from assets) Place within src/third_party/colmap/
3) Run the demo
   - `cd src`
   - `python main.py`
   - A window opens showing the recovered camera frusta and sparse point cloud.

4) Test against COLMAP output for comparisons as TXT files (you get points3d.txt, cameras.txt, and images.txt as outputs)
   - New-Item -ItemType Directory -Force -Path "report/output/colmap" ; `.\src\third_party\colmap\COLMAP.bat automatic_reconstructor `    --workspace_path "report/output/colmap/" `    --image_path "dataset/erik/erik_3/" `    --use_gpu 0 ; `.\src\third_party\colmap\COLMAP.bat model_converter `    --input_path "report/output/colmap/sparse/0" `    --output_path "report/output/colmap/" `    --output_type TXT
   
5) When running output of data for the report, make sure any data entries within report/tables is good.  Then proceed to run generate_table.py for png figures to output

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

