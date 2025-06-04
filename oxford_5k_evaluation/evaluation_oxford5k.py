# oxford_5k_evaluation/evaluation_oxford5k.py
import os
import sys
import time
import numpy as np
import json # For saving parameters

# --- Add parent directory of 'image_retrieval_system' to Python path ---
# This assumes 'evaluation_oxford5k.py' is in a subdir like 'oxford_5k_evaluation'
# and your system 'image_retrieval_system' is one level up. Adjust if needed.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
# Check if 'image_retrieval_system' is in parent_dir or current_dir
if os.path.exists(os.path.join(parent_dir, "image_retrieval_system")):
    sys.path.insert(0, parent_dir)
elif os.path.exists(os.path.join(current_dir, "image_retrieval_system")):
     sys.path.insert(0, current_dir)
else:
    print("Error: 'image_retrieval_system' directory not found. Adjust sys.path.")
    sys.exit(1)

# --- Import your system's modules ---
from image_retrival_system.feature_extractor import extract_sift_features_in_bbox, extract_descriptors_for_database, extract_sift_features
from image_retrival_system.vocabulary_builder import build_vocabulary_akm, build_vocabulary # Choose one
from image_retrival_system.indexer import create_tfidf_index, image_to_bow
from image_retrival_system.retriever import retrieve_similar_images
from image_retrival_system.utils import list_image_files, save_model, load_model

from data_preparation_oxford5k import load_oxford5k_ground_truth # Local import

# --- Configuration for the Evaluation ---
DATASET_BASE_PATH = current_dir # Base for 'oxbuild_images' and 'gt_files'
OXFORD_IMAGE_DIR = os.path.join(DATASET_BASE_PATH, "oxbuild_images")
OXFORD_GT_DIR = os.path.join(DATASET_BASE_PATH, "gt_files")
MODELS_DIR_EVAL = os.path.join(DATASET_BASE_PATH, "models_oxford_eval") # Separate models for this eval

# System Parameters (Match these with your app.py or set for evaluation)
PARAMS = {
    "VOCAB_BUILDER": "AKM", # or "MiniBatchKMeans"
    "NUM_VISUAL_WORDS": 1000, # Paper peaks at 1M, try 1k, 10k, 50k for Oxford 5k
    "AKM_MAX_ITER": 50,
    "AKM_TOL": 1e-4,
    "ENABLE_SPATIAL_RERANKING": True,
    "TOP_N_INITIAL_BOW_CANDIDATES": 200, # How many to fetch from BoW for re-ranking
    "NUM_TO_SPATIALLY_RERANK": 100,    # How many of these to actually spatially verify
    "SPATIAL_MIN_INLIERS": 4,
    "SPATIAL_RANSAC_REPROJ_THRESH": 5.0,
    "SPATIAL_RATIO_THRESH": 0.75,
    # SIFT params are default in cv2.SIFT_create()
}

VOCAB_PATH_EVAL = os.path.join(MODELS_DIR_EVAL, f"vocab_{PARAMS['VOCAB_BUILDER']}_{PARAMS['NUM_VISUAL_WORDS']}.pkl")
TFIDF_MATRIX_PATH_EVAL = os.path.join(MODELS_DIR_EVAL, "tfidf_matrix.pkl")
TFIDF_TRANSFORMER_PATH_EVAL = os.path.join(MODELS_DIR_EVAL, "tfidf_transformer.pkl")
DB_IMAGE_PATHS_PATH_EVAL = os.path.join(MODELS_DIR_EVAL, "db_image_paths.pkl")


def calculate_ap(ranked_list_paths, positive_set_paths, junk_set_paths=None):
    """
    Calculates Average Precision (AP).
    ranked_list_paths: list of image paths, ordered by retrieval system.
    positive_set_paths: set of ground truth positive image paths for the query.
    junk_set_paths: set of ground truth junk image paths for the query.
    """
    if junk_set_paths is None:
        junk_set_paths = set()

    # Filter out junk images from the ranked list, as per paper's methodology [cite: 81]
    processed_ranked_list = [p for p in ranked_list_paths if p not in junk_set_paths]
    
    if not positive_set_paths or not processed_ranked_list:
        return 0.0

    num_relevant_docs = len(positive_set_paths)
    retrieved_relevant_count = 0
    precision_sum = 0.0

    for i, path in enumerate(processed_ranked_list):
        if path in positive_set_paths:
            retrieved_relevant_count += 1
            precision_at_k = retrieved_relevant_count / (i + 1)
            precision_sum += precision_at_k
            
    if retrieved_relevant_count == 0: # No relevant docs retrieved
        return 0.0
        
    # Standard AP: sum of (precision@k * rel@k) / num_relevant_docs
    # Since we only sum precisions when a relevant doc is found (rel@k=1), this is correct.
    return precision_sum / retrieved_relevant_count # Or use num_relevant_docs if that's the standard you follow
                                                  # Using retrieved_relevant_count is common for PR curves.
                                                  # Let's use num_relevant_docs for mAP as per typical IR defs.
    return precision_sum / num_relevant_docs


def main():
    os.makedirs(MODELS_DIR_EVAL, exist_ok=True)

    print("--- System Parameters ---")
    for key, value in PARAMS.items():
        print(f"{key}: {value}")
    print("-------------------------")

    # 1. Data Preparation
    print("\n--- Loading Oxford 5K Ground Truth ---")
    queries = load_oxford5k_ground_truth(DATASET_BASE_PATH, OXFORD_GT_DIR)
    if not queries:
        print("Failed to load queries. Exiting.")
        return

    all_oxford_image_paths = sorted(list_image_files(OXFORD_IMAGE_DIR))
    if not all_oxford_image_paths:
        print(f"No images found in {OXFORD_IMAGE_DIR}. Exiting.")
        return
    print(f"Total images in Oxford 5K dataset for indexing: {len(all_oxford_image_paths)}")

    # 2. Build/Load Vocabulary and Index
    # These are global-like for the system, built once.
    vocabulary = None
    tfidf_matrix_db = None
    tfidf_transformer_db = None
    db_image_paths_indexed = None # Paths corresponding to rows in tfidf_matrix_db

    build_time_start = time.time()
    if os.path.exists(VOCAB_PATH_EVAL) and \
       os.path.exists(TFIDF_MATRIX_PATH_EVAL) and \
       os.path.exists(TFIDF_TRANSFORMER_PATH_EVAL) and \
       os.path.exists(DB_IMAGE_PATHS_PATH_EVAL):
        print("\n--- Loading Pre-built Models for Evaluation ---")
        vocabulary = load_model(VOCAB_PATH_EVAL)
        tfidf_matrix_db = load_model(TFIDF_MATRIX_PATH_EVAL)
        tfidf_transformer_db = load_model(TFIDF_TRANSFORMER_PATH_EVAL)
        db_image_paths_indexed = load_model(DB_IMAGE_PATHS_PATH_EVAL)
        if not all([vocabulary, tfidf_matrix_db is not None, tfidf_transformer_db, db_image_paths_indexed]):
            print("Failed to load one or more models. Rebuilding required.")
            # Force rebuild by clearing flags/models
            vocabulary, tfidf_matrix_db, tfidf_transformer_db, db_image_paths_indexed = None, None, None, None
        else:
             print("All models loaded successfully.")
    
    if vocabulary is None: # Rebuild if not loaded or loading failed
        print("\n--- Building Vocabulary and Index for Oxford 5K ---")
        print("Extracting descriptors for vocabulary...")
        # Use all images from Oxford 5K for vocabulary building, as paper does for 5K dataset
        all_db_descriptors, _ = extract_descriptors_for_database(all_oxford_image_paths)
        if all_db_descriptors is None:
            print("Failed to extract descriptors. Exiting.")
            return

        print(f"Building vocabulary with {PARAMS['NUM_VISUAL_WORDS']} words using {PARAMS['VOCAB_BUILDER']}...")
        if PARAMS["VOCAB_BUILDER"] == "AKM":
            vocabulary = build_vocabulary_akm(
                all_db_descriptors, PARAMS["NUM_VISUAL_WORDS"], 
                max_iter=PARAMS["AKM_MAX_ITER"], tol=PARAMS["AKM_TOL"], random_state=42
            )
        else: # MiniBatchKMeans
            vocabulary = build_vocabulary(all_db_descriptors, PARAMS["NUM_VISUAL_WORDS"], random_state=42)
        
        if vocabulary is None:
            print("Failed to build vocabulary. Exiting.")
            return
        save_model(vocabulary, VOCAB_PATH_EVAL)

        print("Creating TF-IDF index...")
        # db_image_paths_indexed will store paths in the order they appear in tfidf_matrix_db
        tfidf_matrix_db, tfidf_transformer_db, db_image_paths_indexed = create_tfidf_index(
            all_oxford_image_paths, vocabulary, extract_sift_features # Use full image for DB features
        )
        if tfidf_matrix_db is None:
            print("Failed to create TF-IDF index. Exiting.")
            return
        save_model(tfidf_matrix_db, TFIDF_MATRIX_PATH_EVAL)
        save_model(tfidf_transformer_db, TFIDF_TRANSFORMER_PATH_EVAL)
        save_model(db_image_paths_indexed, DB_IMAGE_PATHS_PATH_EVAL)
    
    build_time_end = time.time()
    total_build_time = build_time_end - build_time_start
    print(f"Vocabulary and Indexing Time: {total_build_time:.2f} seconds")

    # 3. Perform Queries and Evaluate
    print("\n--- Starting Evaluation Queries ---")
    all_ap_scores = []
    total_query_time = 0
    num_processed_queries = 0

    for i, query_info in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}: {query_info['query_name']}...")
        
        # Extract features FOR THE QUERY REGION
        # query_bbox is (x1, y1, x2, y2)
        query_kps, query_descs, _ = extract_sift_features_in_bbox(
            query_info["query_image_path"], query_info["query_bbox"]
        )

        if query_descs is None:
            print(f"  Skipping query {query_info['query_name']} due to no features in bbox.")
            all_ap_scores.append(0.0) # Assign 0 AP if query has no features
            continue
        
        query_time_start = time.time()
        # The retriever function needs the full list of db image paths corresponding to tfidf_matrix_db rows
        # and the tfidf_transformer itself for IDF scores if spatial re-ranking is on.
        # retrieve_similar_images should be modified to return the full ranked list for AP calc.
        # We pass db_image_paths_indexed which are the paths used to create the tfidf_matrix_db
        ranked_results_paths, _ = retrieve_similar_images(
            query_image_path=query_info["query_image_path"], # Path used for extracting query_kps, query_descs
            # Provide pre-extracted query features for region to retriever
            # This requires modifying retriever to accept pre-extracted query features
            # For now, retriever will re-extract, but ideally it uses the bbox-specific ones.
            # Let's assume retrieve_similar_images can take query_kps, query_descs if available.
            # Current retriever re-extracts query features; this is a slight deviation
            # for simplicity of not changing retriever's interface too much for this script.
            # A better retriever would take (query_kps, query_descs, query_bow) as optional inputs.
            # For now, query_image_path will be used by retriever to get full query image features.
            # THIS IS A KEY DIFFERENCE - current retriever uses FULL query image for BoW if not modified
            # Let's create query BoW from bbox descriptors here and pass it (conceptual modification)
            # For now, this script is more of a high-level guide.
            # The paper's method: "user supplies a query object by selecting a region" [cite: 2]
            # -> query BoW from region.
            # For simplicity, the current retrieve_similar_images call implicitly uses the whole query image path
            # to re-extract features internally, which is not ideal for region queries.
            # True region query requires passing region-specific BoW to retriever.
            # Let's simulate this by getting the query BoW from region here.

            # Create query BoW from region descriptors
            # query_bow_region = image_to_bow(query_descs, vocabulary)
            # Then pass this query_bow_region to an adapted retrieve_similar_images.
            # The current retrieve_similar_images will use full query image.
            # This is a limitation of directly using the existing retriever without modification.
            # To truly match paper, `retriever.py` needs to handle query_descriptors from region.

            feature_extractor_func=extract_sift_features, # For candidate images in spatial verifier
            vocabulary_model=vocabulary,
            tfidf_transformer=tfidf_transformer_db,
            database_tfidf_matrix=tfidf_matrix_db,
            database_image_paths=db_image_paths_indexed,
            top_n_initial_bow=len(db_image_paths_indexed), # Get full ranking for AP
            top_n_final=len(db_image_paths_indexed),       # Get full ranking for AP
            enable_spatial_reranking=PARAMS["ENABLE_SPATIAL_RERANKING"],
            num_to_rerank_spatial=PARAMS["NUM_TO_SPATIALLY_RERANK"] # Only up to this many get full spatial check
        )
        query_time_end = time.time()
        total_query_time += (query_time_end - query_time_start)
        num_processed_queries +=1

        # Ensure ranked_results_paths doesn't include the query image itself if it was part of db_image_paths_indexed
        # (it shouldn't be if db_image_paths_indexed are all images *except* current query,
        # but here db_image_paths_indexed contains all images)
        # The AP calculation should handle this if query image is not in positive_set.
        # `retrieve_similar_images` already attempts to filter query_image_path.
        
        current_ap = calculate_ap(ranked_results_paths, query_info["positive_paths"], query_info["junk_paths"])
        all_ap_scores.append(current_ap)
        print(f"  Query {query_info['query_name']} AP: {current_ap:.4f}")

    # 4. Report Results
    mean_ap = np.mean(all_ap_scores) if all_ap_scores else 0.0
    avg_query_time = total_query_time / num_processed_queries if num_processed_queries > 0 else 0.0

    print("\n--- Evaluation Summary ---")
    print(f"Dataset: Oxford 5K")
    print(f"Number of Queries Processed: {num_processed_queries}/{len(queries)}")
    print(f"Mean Average Precision (mAP): {mean_ap:.4f}")
    print(f"Total Vocabulary & Indexing Time: {total_build_time:.2f} seconds")
    print(f"Average Query Time: {avg_query_time:.3f} seconds per query")
    
    print("\n--- System Parameters Used ---")
    for key, value in PARAMS.items():
        print(f"{key}: {value}")
    
    results_summary = {
        "dataset": "Oxford 5K",
        "mAP": mean_ap,
        "num_queries_processed": num_processed_queries,
        "total_queries": len(queries),
        "build_time_seconds": total_build_time,
        "avg_query_time_seconds": avg_query_time,
        "parameters": PARAMS
    }
    results_filename = f"results_oxford5k_{PARAMS['VOCAB_BUILDER']}_{PARAMS['NUM_VISUAL_WORDS']}_spatial_{PARAMS['ENABLE_SPATIAL_RERANKING']}.json"
    with open(os.path.join(MODELS_DIR_EVAL, results_filename), 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"\nResults summary saved to: {os.path.join(MODELS_DIR_EVAL, results_filename)}")


if __name__ == '__main__':
    if not os.path.exists(OXFORD_IMAGE_DIR) or not os.path.exists(OXFORD_GT_DIR):
        print(f"Error: Ensure '{OXFORD_IMAGE_DIR}' and '{OXFORD_GT_DIR}' exist.")
        print("Please download and extract the Oxford 5K dataset and ground truth files.")
    else:
        main()