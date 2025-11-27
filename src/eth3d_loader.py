"""
This file is responsible for loading the data from ETH3D undistorted datasets to feed into the SfM pipeline.

We assume the directory layout is the following:

<scene_root>/
    dslr_calibration_undistorted/
       cameras.txt
       images.txt
       points3D.txt
    dslr_scan_eval/
       *.ply files
       scan_alignment.mlp
    images/
        dslr_image_undistorted/
            <image files>
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from plyfile import PlyData

from third_party.colmap.read_write_model import read_cameras_text, read_images_text

def parse_scan_alignment_mlp(mlp_path):
    transforms = {} # filename -> transform matrix

    tree = ET.parse(str(mlp_path))
    root = tree.getroot()

    for mlmesh in root.iter("MLMesh"):
        fname = mlmesh.get("filename")
        if fname is None:
            continue

        mat_elem = mlmesh.find("MLMatrix44")
        if mat_elem is None or mat_elem.text is None:
            continue

        vals = [float(v) for v in mat_elem.text.split()]
        if len(vals) != 16:
            continue

        M = np.array(vals, dtype=np.float64).reshape((4, 4))
        transforms[fname] = M

    return transforms

def load_aligned_gt_scans(gt_dir: Path):
    if not gt_dir.exists():
        return None

    ply_files = sorted(p for p in gt_dir.glob("*.ply"))
    if not ply_files:
        return None

    mlp_path = gt_dir / "scan_alignment.mlp"
    if mlp_path.exists():
        transforms = parse_scan_alignment_mlp(mlp_path)
    else:
        transforms = {}

    clouds: List[np.ndarray] = []

    for ply_path in ply_files:
        ply = PlyData.read(str(ply_path))
        verts = ply["vertex"]
        x = np.asarray(verts["x"], dtype=np.float64)
        y = np.asarray(verts["y"], dtype=np.float64)
        z = np.asarray(verts["z"], dtype=np.float64)

        pts_local = np.stack([x, y, z, np.ones_like(x)], axis=1)  # (N, 4)

        M = transforms.get(ply_path.name, np.eye(4, dtype=np.float64))

        # p_global_homog = M @ p_local_homog
        pts_global_h = (M @ pts_local.T).T  # (N, 4)
        pts_global = pts_global_h[:, :3] / pts_global_h[:, 3:4]

        clouds.append(pts_global)

    return np.vstack(clouds) if clouds else None

def load_scene_eth3d(scene_root):
    """
    This is the main loading function we would call outside this file. It will return:
        - the paths to each of the images for the scene
        - the camera intrinsics the scene uses
        - the camera poses
        - the ground-truth point cloud
       all image files

    The first two outputs will be the inputs to our pipeline. The last two outputs are
    for us to compare our point cloud to the GT scan.
    """
    root = Path(scene_root)
 
    img_dir = root / "images" / "dslr_images_undistorted"
    gt_dir = root / "dslr_scan_eval"

    calib_dir = root / "dslr_calibration_undistorted"
    cameras_txt = calib_dir / "cameras.txt"
    images_txt = calib_dir / "images.txt"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {img_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing GT directory: {gt_dir}")
    if not calib_dir.is_dir():
        raise FileNotFoundError(
            f"Missing dslr_calibration_undistorted directory: {calib_dir}"
        )
    if not cameras_txt.is_file():
        raise FileNotFoundError(f"Missing cameras.txt: {cameras_txt}")
    if not images_txt.is_file():
        raise FileNotFoundError(f"Missing images.txt: {images_txt}")

    # 1. get image paths
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    images_list = sorted(
        p for p in img_dir.iterdir() if p.suffix in exts and p.is_file()
    )

    # 2. Calibration (camera.txt + images.txt)
    cameras = read_cameras_text(calib_dir / "cameras.txt")
    images = read_images_text(calib_dir / "images.txt")

    # Intrinsics: build K from params for PINHOLE model
    intrinsics = {} # cam_id -> K
    for cam_id, cam in cameras.items():
        if cam.model not in ("PINHOLE", "SIMPLE_PINHOLE"):
            # ETH3D undistorted DSLRs should be PINHOLE; warn otherwise.
            print(
                f"[WARN] Unexpected camera model '{cam.model}' for camera {cam_id}. "
                f"Proceeding, but K construction may be wrong."
            )

        if cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params
        elif cam.model == "SIMPLE_PINHOLE":
            # SIMPLE_PINHOLE: [f, cx, cy]
            f, cx, cy = cam.params
            fx, fy = f, f
        else:
            raise ValueError(
                f"Cannot construct intrinsics for camera model {cam.model} with params {cam.params}"
            )

        K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        intrinsics[cam_id] = K

    # Extrinsics: per image (world->camera pose)
    extrinsics = {} # image_name -> (R, t, cam_id)
    for img in images.values():
        R = img.qvec2rotmat()
        t = img.tvec.reshape(3, 1)  # column vector
        extrinsics[img.name] = (R, t, img.camera_id)

    # 3. Ground truth
    gt = load_aligned_gt_scans(gt_dir)
    return images_list, intrinsics, extrinsics, gt

# simple test code to make sure we can read the directories and parse the data correctly
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load an ETH3D scene and print basic info."
    )
    parser.add_argument("scene_root", type=str, help="Path to ETH3D scene root")
    args = parser.parse_args()

    images, intrinsics, extrinsics, gt = load_scene_eth3d(args.scene_root)
    print(f"# images: {len(images)}")
    print(f"# cameras: {len(intrinsics)}")
    print(f"# extrinsics: {len(extrinsics)}")
    print(f"GT points: {gt.shape[0]} points")
