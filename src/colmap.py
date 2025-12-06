import os
import pycolmap
import numpy as np
from utils import profile

def getIntrinsics(model, params):
    """
    We don't account for distortion in our pipeline, so if "RADIAL" or "SIMPLE_RADIAL" is detected,
    we will just try treating it as pinhole.
    """
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])

    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])

    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
    else:
        raise NotImplementedError(f"Camera model {model} not handled.")
    return K

@profile
def runColmap(images_dir, workspace):
    os.makedirs(workspace, exist_ok=True)

    database_path = os.path.join(workspace, "database.db")
    sparse_dir = os.path.join(workspace, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    pycolmap.extract_features(
        database_path=database_path,
        image_path=images_dir,
    )

    pycolmap.match_exhaustive(
        database_path=database_path,
    )

    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=images_dir,
        output_path=sparse_dir,
    )
    if maps:
        maps[0].write(sparse_dir)

    return pycolmap.Reconstruction(sparse_dir)

