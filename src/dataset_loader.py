import os
import numpy as np
import cv2 as cv

class DatasetInfo:
    def __init__(self, dataset_dir, focal_length):
        self.dataset_dir = dataset_dir
        self.focal_length = focal_length
        self.im_width, self.im_height = self.getDimensions()
        # Compute intrinsic matrix
        self.K = self.compute_K()

    def getDimensions(self):
        images = os.listdir(self.dataset_dir)
        print(images)
        # Throw error if dir is empty
        if not images:
            raise FileNotFoundError(f"No files found in {self.dataset_dir}")
        
        # Load first image, and throw an error if it is not an image
        image_path = os.path.join(self.dataset_dir, images[0])
        img = cv.imread(image_path)
        if img is None:
            raise ValueError(f"Invalid image file: {image_path}")

        # Get the width, height
        width, height, _ = img.shape
        return width, height
            
    def compute_K(self):  
        f = max(self.im_width, self.im_height) * self.focal_length / 35.0
        return np.array([[f, 0, self.im_width / 2], [0, f, self.im_height / 2], [0, 0, 1]])
    
# Hardcode intrinsics for our known datasets
# Edit this if you want to add a new dataset and have it auto get focal length, or manually add enter when typing dataset name
dataset_intrics = {
    "erik_1": {"focal_length": 45.0},
    "erik_2": {"focal_length": 43.0},
    "erik_3": {"focal_length": 43.0},
    "erik_4": {"focal_length": 43.0},
    "erik_5": {"focal_length": 45.0},
    "erik_6": {"focal_length": 38.0},
    "erik_7": {"focal_length": 38.0},
    "erik_8": {"focal_length": 38.0},
    "erik_9": {"focal_length": 38.0},
}

def select_dataset(dataset_dir):    
    all_datasets = [dir for dir in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, dir))]

    known_datasets = []
    unknown_datasets = []

    # list known and unknown datasets separately
    for dir in all_datasets:
        if dir in dataset_intrics:
            known_datasets.append(dir)
        else:
            unknown_datasets.append(dir)
            
    print(f"\nDatasets found in: {dataset_dir}")
    # Print known datasets
    print("\nKnown Datasets:")
    for idx, dataset_name in enumerate(known_datasets):
        print(f"{idx} : {dataset_name}")

    # Print unknown datasets, starting from the last idx
    print("\nUnknown Datasets:")
    for idx, dataset_name in enumerate(unknown_datasets):
        print(f"{idx + len(known_datasets)}. {dataset_name}")

    # Ask user to select a dataset
    choice = input(f"\nPlease select a dataset by entering it's corresponding number. If an unknown dataset is chosen, you will be asked to enter a focal length for it:\n").strip()

    # Loop until user enters a valid dataset
    while True:
        # If known dataset is chosen
        if choice.isdigit() and (0 <= int(choice) <= len(known_datasets)-1):
            dataset_name = known_datasets[int(choice)]
            print(f"You have chosen: {choice} : {dataset_name}\n")
            dataset_path = os.path.join(dataset_dir, dataset_name)
            return DatasetInfo(dataset_path, dataset_intrics[dataset_name]["focal_length"])

        # If unknown dataset is chosen
        elif choice.isdigit() and (len(known_datasets)-1 < int(choice) < len(all_datasets)):
            # User selected an unknown dataset
            dataset_name = unknown_datasets[int(choice) - len(known_datasets)]
            print(f"You have chosen: {choice} : {dataset_name}\n")
            dataset_path = os.path.join(dataset_dir, dataset_name)
            # Loop until user enters a valid focal length
            while True:
                try:
                    focal_length = float(input(f"Enter the focal length for in mm: ").strip())
                    return DatasetInfo(dataset_path, focal_length)
                except ValueError:
                    print("Error: Please enter a valid number for the focal length.")
                    continue
        else:
            # Repeat if invalid input was entered
            print(f"Invalid input. Please enter a valid number from 0 to {len(all_datasets)-1}")
            choice = input().strip()
            continue