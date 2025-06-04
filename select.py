import json
import os
import shutil
import random
from tqdm import tqdm


def select_and_copy_images(json_path, source_dir, target_dir, query_dir):
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)

    # Read JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Process each landmark category
    for landmark, categories in tqdm(
        data.items(), desc="Processing landmarks", unit="landmark"
    ):
        # Combine 'ok' and 'good' images
        positive_images = categories.get("ok", []) + categories.get("good", [])

        # Select up to 10 images from positive examples
        selected_images = (
            positive_images
            if len(positive_images) <= 10
            else random.sample(positive_images, 10)
        )

        # Copy selected positive images with landmark prefix
        for idx, img_path in enumerate(selected_images):
            src_path = os.path.join(source_dir, img_path)
            dst_path = os.path.join(
                target_dir, f"{landmark}_{idx+1}{os.path.splitext(img_path)[1]}"
            )
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)

        # Copy one query image for each landmark
        if categories.get("query"):
            query_image = categories["query"][0]
            src_path = os.path.join(source_dir, query_image)
            dst_path = os.path.join(
                query_dir, f"{landmark}{os.path.splitext(query_image)[1]}"
            )
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)


if __name__ == "__main__":
    # Configure these paths according to your setup
    json_path = "oxford5k/groundtruth.json"
    source_dir = "oxford5k/images"
    target_dir = "database_images/oxford5k"
    query_dir = "query_images/oxford5k"

    select_and_copy_images(json_path, source_dir, target_dir, query_dir)
    print("Image selection and copying completed.")
