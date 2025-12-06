import os
import numpy as np
import cv2 as cv

class DatasetInfo:
    """
    To integrate cleanly with colmap, the file structure can be:

    project_root\
        datasets\
            datasetA\
                sceneA\
                    images\
                    colmap\
                sceneB\
                    ...
    """
    def __init__(self, scene_dir, focal_length=None, from_cli = False):
        self.scene_dir = scene_dir
        self.image_dir = os.path.join(scene_dir, "images")
        self.colmap_dir = os.path.join(scene_dir, "colmap")
        self.from_cli = from_cli
        self.im_height, self.im_width = self.getDimensions()
        if (from_cli):
            self.focal_length = focal_length
            self.K = self.compute_K()
        else:
            self.focal_length = None
            self.K = None

    def getDimensions(self):
        images = os.listdir(self.image_dir)
        print(images)
        # Throw error if dir is empty
        if not images:
            raise FileNotFoundError(f"No files found in {self.scene_dir}")

        # Load first image, and throw an error if it is not an image
        image_path = os.path.join(self.image_dir, images[0])
        img = cv.imread(image_path)
        if img is None:
            raise ValueError(f"Invalid image file: {image_path}")

        # Get the width, height
        height, width, _ = img.shape
        return height, width

    def compute_K(self):
        return np.array([[self.focal_length, 0, self.im_width / 2], [0, self.focal_length, self.im_height / 2], [0, 0, 1]])

# Hardcode intrinsics for our known datasets
# Edit this if you want to add a new dataset and have it auto get focal length, or manually add enter when typing dataset name
KNOWN_DATASETS = {
    "erik/erik_1": {"focal_length": 2489.14},
    "erik/erik_2": {"focal_length": 2378.51},
    "erik/erik_3": {"focal_length": 2378.51},
    "erik/erik_4": {"focal_length": 2378.51},
    "erik/erik_5": {"focal_length": 2489.14},
    "erik/erik_6": {"focal_length": 2466.74},
    "erik/erik_7": {"focal_length": 2466.74},
    "erik/erik_8": {"focal_length": 2466.74},
    "erik/erik_9": {"focal_length": 2466.74},
    #"misc/colosseum": {"focal_length": 4838.4},
    #"misc/colosseum_rotated": {"focal_length": 4838.4},
    #"misc/trevi": {"focal_length": 4838.4},
}

def select_dataset(datasets_path):
    known_datasets = []
    unknown_datasets = []
    rel_map = {} # absolute dir -> relative dir

    for dataset in os.listdir(datasets_path):
        dataset_dir = os.path.join(datasets_path, dataset)
        if not os.path.isdir(dataset_dir):
            continue

        for scene in os.listdir(dataset_dir):
            scene_dir = os.path.join(dataset_dir, scene)
            if not os.path.isdir(os.path.join(scene_dir, "images")):
                continue

            rel = os.path.join(dataset, scene).replace("\\", "/")
            rel_map[scene_dir] = rel

            if any(rel.endswith(k) or rel == k for k in KNOWN_DATASETS.keys()):
                known_datasets.append(scene_dir)
            else:
                unknown_datasets.append(scene_dir)

    all_datasets = known_datasets + unknown_datasets

    print(f"\nDatasets found in: {datasets_path}")
    # Print known datasets
    print("\nKnown Datasets:")
    for idx, dataset_name in enumerate(known_datasets):
        print(f"{idx} : {dataset_name}")

    # Print unknown datasets, starting from the last idx
    print("\nUnknown Datasets:")
    for idx, dataset_name in enumerate(unknown_datasets):
        print(f"{idx + len(known_datasets)}. {dataset_name}")

    # Ask user to select a dataset
    choice = input(f"\nPlease select a dataset by entering it's corresponding number.\n").strip()

    # Loop until user enters a valid dataset
    while True:
        # If known dataset is chosen
        if choice.isdigit() and (0 <= int(choice) <= len(known_datasets)-1):
            dataset = all_datasets[int(choice)]
            rel = rel_map[dataset]
            print(f"You have chosen: {choice} : {rel}\n")
            focal = KNOWN_DATASETS[rel].get("focal_length", None)
            return DatasetInfo(dataset, focal, True)

        # If unknown dataset is chosen
        elif choice.isdigit() and (len(known_datasets)-1 < int(choice) < len(all_datasets)):
            # User selected an unknown dataset
            dataset_name = unknown_datasets[int(choice) - len(known_datasets)]
            print(f"You have chosen: {choice} : {dataset_name}\n")
            dataset_path = os.path.join(dataset_dir, dataset_name)
            return DatasetInfo(dataset_path)
        else:
            # Repeat if invalid input was entered
            print(f"Invalid input. Please enter a valid number from 0 to {len(all_datasets)-1}")
            choice = input().strip()
            continue
