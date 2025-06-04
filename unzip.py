import os
import tarfile
import shutil
import random
from tqdm import tqdm


def extract_images(tar_path, output_dir, samples_per_category=10):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the tar.gz file
    with tarfile.open(tar_path, "r:gz") as tar:
        # Get all members (files) in the archive
        members = tar.getmembers()

        # Track current category
        current_category = None
        category_files = []

        for member in tqdm(members, desc="Extracting images", total=len(members)):
            # Skip if not a file
            if not member.isfile():
                continue

            # Get category name from path
            path_parts = member.name.split("/")
            if len(path_parts) < 2:
                continue

            category = path_parts[1]

            # If we encounter a new category, process the previous one
            if current_category and category != current_category:
                # Randomly select images from the category
                selected_files = random.sample(
                    category_files, min(samples_per_category, len(category_files))
                )

                # Extract selected files
                for idx, file_member in enumerate(selected_files):
                    new_filename = f"{current_category}_{idx+1}.jpg"
                    extracted_path = os.path.join(output_dir, new_filename)

                    with tar.extractfile(file_member) as source:
                        with open(extracted_path, "wb") as target:
                            shutil.copyfileobj(source, target)

                # Reset for next category
                category_files = []

            current_category = category
            category_files.append(member)

        # Process the last category
        if category_files:
            selected_files = random.sample(
                category_files, min(samples_per_category, len(category_files))
            )

            for idx, file_member in enumerate(selected_files):
                new_filename = f"{current_category}_{idx+1}.jpg"
                extracted_path = os.path.join(output_dir, new_filename)

                with tar.extractfile(file_member) as source:
                    with open(extracted_path, "wb") as target:
                        shutil.copyfileobj(source, target)


if __name__ == "__main__":
    # Define paths
    tar_path = "101_ObjectCategories.tar.gz"
    output_dir = "database_images"

    # Extract 5 images per category
    extract_images(tar_path, output_dir, samples_per_category=5)
    print("Extraction completed!")
