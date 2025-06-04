# image_retrieval_system/retriever.py
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from indexer import image_to_bow
from spatial_verifier import spatial_re_rank  # Import the new function
import os


def retrieve_similar_images(
    query_image_path,
    feature_extractor_func,
    vocabulary_model,
    tfidf_transformer,  # Provides IDF scores
    database_tfidf_matrix,
    database_image_paths,
    top_n_initial_bow=100,  # How many candidates for BoW
    top_n_final=10,  # How many to return after re-ranking
    enable_spatial_reranking=True,
    num_to_rerank_spatial=50,  # How many of BoW results to spatially check
):
    """
    Retrieves the top_n most similar images from the database for a query image.
    Includes an optional spatial re-ranking step.
    """
    if (
        database_tfidf_matrix is None
        or vocabulary_model is None
        or tfidf_transformer is None
    ):
        print("Error: Index or models not loaded for retrieval.")
        return [], "Error: Index or models not loaded."

    query_keypoints, query_descriptors, _ = feature_extractor_func(
        query_image_path
    )  # Get keypoints too
    if query_descriptors is None:
        msg = f"Could not extract features from query image: {query_image_path}"
        print(msg)
        return [], msg

    query_bow = image_to_bow(query_descriptors, vocabulary_model)
    if query_bow is None:
        msg = "Could not create BoW for query image."
        print(msg)
        return [], msg

    query_tfidf = tfidf_transformer.transform(query_bow.reshape(1, -1))

    distances = euclidean_distances(query_tfidf, database_tfidf_matrix).flatten()

    # Get more initial candidates than top_n_final if spatial re-ranking is enabled
    num_candidates_for_bow = (
        top_n_initial_bow if enable_spatial_reranking else top_n_final
    )

    sorted_indices = np.argsort(distances)

    initial_results_with_scores = []
    # Collect initial BoW results (path, L2_distance_score)
    # L2_distance_score is lower is better
    for i in range(min(num_candidates_for_bow, len(sorted_indices))):
        idx = sorted_indices[i]
        # Basic check to not include query image itself if it happens to be in DB with same path
        if database_image_paths[idx] != query_image_path:
            initial_results_with_scores.append(
                (database_image_paths[idx], distances[idx])
            )

    if not initial_results_with_scores:
        return [], "No initial matches found via BoW."

    final_results_paths = []
    scores_info_str = ""

    if enable_spatial_reranking:
        print("Performing spatial re-ranking...")
        if tfidf_transformer.idf_ is None:
            return (
                [],
                "IDF scores not available from tfidf_transformer for spatial re-ranking.",
            )

        idf_scores = (
            tfidf_transformer.idf_
        )  # These are the IDF scores for each visual word

        # The spatial_re_rank function will return a list of (path, original_L2_score) tuples,
        # but their order is now based on spatial verification.
        re_ranked_results_tuples = spatial_re_rank(
            query_image_path,
            query_keypoints,
            query_descriptors,
            initial_results_with_scores,
            feature_extractor_func,
            vocabulary_model,
            idf_scores,
            num_to_rerank=num_to_rerank_spatial,
        )

        # Take top_n_final from the re-ranked list
        # The scores in re_ranked_results_tuples are still the original L2 distances.
        # The order reflects the new ranking.
        for i, (path, l2_dist) in enumerate(re_ranked_results_tuples[:top_n_final]):
            final_results_paths.append(path)
            # For scores_info_str, we might want to indicate if it was spatially verified later,
            # but for now, we just use the L2 distance from BoW.
            # The spatial_verifier module could be modified to return more info like spatial score/inliers.
            item_spatial_info = next(
                (item for item in re_ranked_results_tuples if item[0] == path), None
            )  # Find full info if needed
            # This is a bit clunky to get spatial scores back here; ideally, spatial_re_rank would return richer objects
            scores_info_str += (
                f"{i+1}. {os.path.basename(path)} (L2 BoW: {l2_dist:.4f})\n"
            )

    else:  # No spatial re-ranking
        for i, (path, l2_dist) in enumerate(initial_results_with_scores[:top_n_final]):
            final_results_paths.append(path)
            scores_info_str += (
                f"{i+1}. {os.path.basename(path)} (L2 BoW: {l2_dist:.4f})\n"
            )

    if not final_results_paths:
        return [], "No similar images found after processing."

    return final_results_paths, scores_info_str.strip()
