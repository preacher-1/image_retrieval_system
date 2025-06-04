# oxford_5k_evaluation/data_preparation_oxford5k.py
from email.mime import image
import os


def parse_gt_file(filepath):
    """Parses a single ground truth file (good, ok, junk, query)."""
    image_names = []
    with open(filepath, "r") as f:
        for line in f:
            image_names.append(
                line.strip().split(".")[0]
            )  # Get imagename without extension
    return set(image_names)


def parse_query_file(filepath):
    """Parses a query file to get query image name and bbox."""
    with open(filepath, "r") as f:
        line = f.readline().strip()
    # Expected format: "oxc1_all_souls_000013 143.0 20.0 600.0 700.0" (for Oxford Paris)
    # Or "all_souls_000013 143.0 20.0 600.0 700.0" (for original Oxford 5k)
    parts = line.split(" ")
    if parts[0].startswith("oxc1_"):  # Handle Paris dataset naming if mixed
        image_name = parts[0][5:]  # Remove "oxc1_" prefix
    else:
        image_name = parts[0]

    # Convert bbox from string to float, then to int if desired, or keep as float
    # bbox is x1, y1, x2, y2
    bbox = tuple(map(float, parts[1:]))
    return image_name, bbox


def load_oxford5k_ground_truth(dataset_base_path, gt_files_path):
    """
    Loads and parses the Oxford 5K ground truth.
    Returns a list of query dicts, each containing:
    'query_name', 'query_image_path', 'query_bbox',
    'positive_paths', 'junk_paths', 'negative_paths' (implicitly all others)
    """
    queries_data = []
    # image_dir = os.path.join(dataset_base_path, "oxbuild_images")
    image_dir = "../oxford5k/images"

    all_image_files_map = {
        os.path.splitext(f)[0]: os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(".jpg")
    }

    gt_query_files = sorted(
        [f for f in os.listdir(gt_files_path) if f.endswith("_query.txt")]
    )

    for query_file_name in gt_query_files:
        query_name_base = query_file_name.replace("_query.txt", "")

        query_image_name, query_bbox = parse_query_file(
            os.path.join(gt_files_path, query_file_name)
        )

        query_image_path = all_image_files_map.get(query_image_name)
        if not query_image_path:
            print(
                f"Warning: Query image {query_image_name}.jpg not found for {query_file_name}"
            )
            continue

        good_file = os.path.join(gt_files_path, f"{query_name_base}_good.txt")
        ok_file = os.path.join(gt_files_path, f"{query_name_base}_ok.txt")
        junk_file = os.path.join(gt_files_path, f"{query_name_base}_junk.txt")

        good_set = parse_gt_file(good_file) if os.path.exists(good_file) else set()
        ok_set = parse_gt_file(ok_file) if os.path.exists(ok_file) else set()
        junk_set = parse_gt_file(junk_file) if os.path.exists(junk_file) else set()

        positive_image_names = good_set.union(ok_set)

        positive_paths = {
            all_image_files_map.get(name)
            for name in positive_image_names
            if all_image_files_map.get(name)
        }
        junk_paths = {
            all_image_files_map.get(name)
            for name in junk_set
            if all_image_files_map.get(name)
        }

        # Ensure the query image itself isn't accidentally in its own positive/junk list for scoring
        # (though it shouldn't be based on typical ground truth).
        # For AP calculation, the query image itself is not retrieved.
        positive_paths.discard(query_image_path)
        junk_paths.discard(query_image_path)

        queries_data.append(
            {
                "query_name": query_name_base,
                "query_image_path": query_image_path,
                "query_bbox": query_bbox,  # (x1, y1, x2, y2)
                "positive_paths": positive_paths,
                "junk_paths": junk_paths,
            }
        )

    print(f"Loaded {len(queries_data)} queries from Oxford 5K ground truth.")
    return queries_data


if __name__ == "__main__":
    # Example usage:
    # Assume this script is in oxford_5k_evaluation/
    # and oxford_5k_evaluation/oxbuild_images/ and oxford_5k_evaluation/gt_files/ exist
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_base = (
        current_script_dir  # Main directory containing 'oxbuild_images' and 'gt_files'
    )
    gt_files = os.path.join(dataset_base, "gt_files")

    # if not os.path.exists(
    #     os.path.join(dataset_base, "oxbuild_images")
    # ) or not os.path.exists(gt_files):
    #     print(
    #         "Please ensure 'oxbuild_images' and 'gt_files' directories exist in the same directory as this script."
    #     )
    #     print("Download from http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/")
    # else:
    #     queries = load_oxford5k_ground_truth(dataset_base, gt_files)
    #     if queries:
    #         print(f"\nExample Query 0: {queries[0]['query_name']}")
    #         print(f"  Image: {queries[0]['query_image_path']}")
    #         print(f"  BBox: {queries[0]['query_bbox']}")
    #         print(f"  Num Positives: {len(queries[0]['positive_paths'])}")
    #         print(f"  Num Junk: {len(queries[0]['junk_paths'])}")

    queries = load_oxford5k_ground_truth(dataset_base, gt_files)
    if queries:
        print(f"Loaded {len(queries)} queries.")
        print(f"Example Query 0: {queries[0]['query_name']}")
        print(f"  Image: {queries[0]['query_image_path']}")
        print(f"  BBox: {queries[0]['query_bbox']}")
        print(f"  Num Positives: {len(queries[0]['positive_paths'])}")
        print(f"  Num Junk: {len(queries[0]['junk_paths'])}")
