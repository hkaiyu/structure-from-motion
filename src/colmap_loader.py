"""
This file loads a COLMAP text model to feed into the SfM pipeline, returning data
in the same structure as eth3d_loader.load_scene_eth3d:

Returns:
    images_list: list[Path] - paths to input images (that COLMAP used)
    intrinsics: dict[int, np.ndarray] - camera_id -> 3x3 K matrix
    extrinsics: dict[str, tuple[np.ndarray, np.ndarray, int]] - image_name -> (R, t, camera_id)
    gt: np.ndarray | None - (N, 3) array of reconstructed 3D points from COLMAP (used here as GT)

Assumptions:
- You have run COLMAP and converted the model to TXT, using command from README.md, producing:
    cameras.txt, images.txt, points3D.txt
- By default we will look for these files in:
    - the provided scene_root (directly)
    - scene_root / "sparse/0"
    - project_root / "report" / "output" / "colmap"
- Image names stored by COLMAP (images.txt)

"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from third_party.colmap.read_write_model import (
    read_cameras_text,
    read_images_text,
    read_points3D_text,
)


def _has_colmap_txt(dir_path: Path) -> bool:
    return (
        (dir_path / "cameras.txt").is_file()
        and (dir_path / "images.txt").is_file()
        and (dir_path / "points3D.txt").is_file()
    )


def _find_colmap_model_dir(scene_root: Path) -> Path:
    """
    Try to find a directory containing cameras.txt, images.txt, points3D.txt.
    Priority:
      1) scene_root
      2) scene_root / 'sparse/0'
      3) project_root / 'report/output/colmap'
    """
    candidates: List[Path] = [scene_root, scene_root / "sparse" / "0"]

    project_root = Path(__file__).resolve().parent.parent
    candidates.append(project_root / "report" / "output" / "colmap")

    for c in candidates:
        if _has_colmap_txt(c):
            return c

    raise FileNotFoundError(
        "Could not find COLMAP TXT model. Expected cameras.txt, images.txt, points3D.txt "
        f"in one of: {', '.join(str(c) for c in candidates)}"
    )


def _build_intrinsics_from_cameras(cameras) -> Dict[int, np.ndarray]:
    """
    Build intrinsics K per camera_id. Supports PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL.
    """
    intrinsics: Dict[int, np.ndarray] = {}
    for cam_id, cam in cameras.items():
        if cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params
        elif cam.model == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params
            fx, fy = f, f
        elif cam.model == "SIMPLE_RADIAL":
            # SIMPLE_RADIAL: [f, cx, cy, k]; we ignore distortion k in K construction
            f, cx, cy, _k = cam.params
            fx, fy = f, f
        else:
            # Extend as needed for other models (RADIAL, OPENCV, etc.)
            raise ValueError(
                f"Unsupported camera model '{cam.model}' for camera {cam_id}. "
                f"Add handling here if needed. Params: {cam.params}"
            )

        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        intrinsics[cam_id] = K
    return intrinsics


def _resolve_image_paths(scene_root: Path, image_names: List[str]) -> List[Path]:
    """
    Resolve image names from COLMAP images.txt to actual file paths on disk.
    Tries several strategies:
      - as-is (absolute or relative to CWD)
      - scene_root / name
      - match by basename under scene_root
    Falls back to listing all images in scene_root if nothing resolves.
    """
    resolved: List[Path] = []

    # Pre-index by basename for matching
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    all_imgs = [p for p in scene_root.iterdir() if p.suffix in exts and p.is_file()]
    by_basename = {p.name.lower(): p for p in all_imgs}

    for name in image_names:
        name_stripped = name.strip()
        p = Path(name_stripped)

        if p.is_file():
            resolved.append(p)
            continue

        p2 = scene_root / name_stripped
        if p2.is_file():
            resolved.append(p2)
            continue

        # Try matching by just the basename under scene_root
        base = Path(name_stripped).name.lower()
        if base in by_basename:
            resolved.append(by_basename[base])
            continue

    if len(resolved) < max(1, int(0.7 * len(image_names))):
        if all_imgs:
            return sorted(all_imgs)
        else:
            # As a last resort return what we have (may be empty)
            return resolved

    return resolved


def load_scene_colmap(scene_root) -> Tuple[List[Path], Dict[int, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray, int]], Optional[np.ndarray]]:
    """
    Main loading function: returns (images_list, intrinsics, extrinsics, gt_points)

    - images_list: paths to images used in the reconstruction
    - intrinsics: camera_id -> K
    - extrinsics: image_name -> (R, t, camera_id)
    - gt_points: Nx3 array of points from points3D.txt (can be used as 'GT' reference)
    """
    root = Path(scene_root)
    model_dir = _find_colmap_model_dir(root)

    cameras = read_cameras_text(model_dir / "cameras.txt")
    images = read_images_text(model_dir / "images.txt")
    points3D = read_points3D_text(model_dir / "points3D.txt")

    # Build intrinsics
    intrinsics = _build_intrinsics_from_cameras(cameras)

    # Build extrinsics
    extrinsics: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
    for img in images.values():
        R = img.qvec2rotmat().astype(np.float64)
        t = img.tvec.reshape(3, 1).astype(np.float64)
        extrinsics[img.name] = (R, t, img.camera_id)

    # Ground-truth (we treat COLMAP points as reference here)
    if len(points3D) > 0:
        gt = np.stack([pt.xyz for pt in points3D.values()], axis=0).astype(np.float64)
    else:
        gt = None

    # Build images list from names in images.txt
    # Order by image_id to keep consistent ordering
    image_items = sorted(images.items(), key=lambda kv: kv[0])
    image_names = [img.name for _, img in image_items]
    images_list = _resolve_image_paths(root, image_names)

    return images_list, intrinsics, extrinsics, gt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load a COLMAP model inputs and print basic info.")
    parser.add_argument("scene_root", type=str, help="Path to image root or model dir")
    args = parser.parse_args()

    images_list, intrinsics, extrinsics, gt = load_scene_colmap(args.scene_root)
    print(f"# images: {len(images_list)}")
    print(f"# cameras: {len(intrinsics)}")
    print(f"# extrinsics (images): {len(extrinsics)}")
    print(f"GT points: {gt.shape[0]} points")
