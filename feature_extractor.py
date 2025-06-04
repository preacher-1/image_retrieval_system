# image_retrieval_system/feature_extractor.py
import cv2
import numpy as np
from typing import List, Tuple, Optional


def extract_sift_features(
    image_path,
) -> Tuple[
    Optional[List[cv2.KeyPoint]], Optional[np.ndarray], Optional[Tuple[int, int]]
]:
    """
    Extracts SIFT keypoints and descriptors from an image.
    Inspired by the use of SIFT descriptors in the paper[cite: 84].
    Returns:
        keypoints: List of SIFT keypoints.
        descriptors: SIFT descriptors for the keypoints.
        img_shape: Shape of the input image.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return None, None, None

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)

        if descriptors is None:
            print(f"Warning: No SIFT descriptors found for {image_path}")
            return None, None, img.shape

        return keypoints, descriptors, img.shape
    except Exception as e:
        print(f"Error extracting SIFT features for {image_path}: {e}")
        return None, None, None


def extract_descriptors_for_database(image_paths):
    """
    Extracts all SIFT descriptors from a list of image paths.
    (No change needed here if vocabulary is built only on descriptors)
    """
    all_descriptors = []
    valid_image_paths = []
    for image_path in image_paths:
        _, descriptors, _ = extract_sift_features(image_path)  # Adjusted unpacking
        if descriptors is not None:
            all_descriptors.append(descriptors)
            valid_image_paths.append(image_path)
    if not all_descriptors:
        return None, []
    return np.vstack(all_descriptors), valid_image_paths
