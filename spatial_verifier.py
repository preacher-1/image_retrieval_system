# image_retrieval_system/spatial_verifier.py
import cv2
import numpy as np


def match_features(
    query_kp, query_desc, candidate_kp, candidate_desc, ratio_thresh=0.75
):
    """Matches SIFT features between query and candidate using Lowe's ratio test."""
    if (
        query_desc is None
        or candidate_desc is None
        or len(query_kp) == 0
        or len(candidate_kp) == 0
    ):
        return [], [], []

    bf = cv2.BFMatcher(cv2.NORM_L2)  # SIFT uses L2 norm
    matches = bf.knnMatch(query_desc, candidate_desc, k=2)

    good_matches = []
    # Apply Lowe's ratio test [cite: 256] (though not explicitly detailed as Lowe's in this paper, it's standard for SIFT)
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # Extract matched keypoints
    query_matched_kp = np.float32(
        [query_kp[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    candidate_matched_kp = np.float32(
        [candidate_kp[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # Store the indices of the descriptors for inliers
    query_inlier_desc_indices = [m.queryIdx for m in good_matches]
    candidate_inlier_desc_indices = [m.trainIdx for m in good_matches]

    return (
        query_matched_kp,
        candidate_matched_kp,
        good_matches,
        query_inlier_desc_indices,
        candidate_inlier_desc_indices,
    )


def verify_transform(
    query_matched_kp,
    candidate_matched_kp,
    min_inliers_for_transform=4,
    ransac_reproj_thresh=5.0,
):
    """
    Estimates an affine transformation using RANSAC and returns inlier count and mask.
    The paper uses RANSAC [cite: 185] and tests affine transformations[cite: 194, 195].
    """
    if len(query_matched_kp) < min_inliers_for_transform:
        return None, 0, []

    # Using estimateAffine2D for a 6 DoF affine transformation, similar to the paper's refinement step [cite: 195]
    # For simpler transformations (3,4,5 DoF) mentioned in paper's hypothesis stage[cite: 194],
    # cv2.estimateAffinePartial2D could be used, or custom hypothesis generation.
    # Here, we use a more general affine transform for simplicity in a single RANSAC step.
    affine_matrix, inlier_mask = cv2.estimateAffine2D(
        query_matched_kp,
        candidate_matched_kp,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_thresh,
    )

    if affine_matrix is None or inlier_mask is None:
        return None, 0, []

    num_inliers = np.sum(inlier_mask)
    return affine_matrix, num_inliers, inlier_mask.flatten()


def calculate_spatial_score(
    inlier_indices_query, query_descriptors, vocabulary_model, idf_scores
):
    """
    Calculates spatial score based on the sum of IDF values of inlier visual words.
    As described in the paper[cite: 218].
    """
    score = 0
    if (
        vocabulary_model is None
        or idf_scores is None
        or inlier_indices_query is None
        or query_descriptors is None
    ):
        return 0

    # Get descriptors of inliers from the query image
    query_inlier_descriptors = query_descriptors[inlier_indices_query]

    if query_inlier_descriptors.shape[0] > 0:
        try:
            visual_words = vocabulary_model.predict(
                query_inlier_descriptors.astype(np.float32)
            )
            for vw_idx in visual_words:
                if 0 <= vw_idx < len(idf_scores):
                    score += idf_scores[vw_idx]
        except Exception as e:
            print(f"Error predicting visual words or calculating IDF score: {e}")
            return 0
    return score


def spatial_re_rank(
    query_image_path,
    query_keypoints_initial,
    query_descriptors_initial,
    initial_ranked_results_with_scores,  # List of (path, initial_score)
    feature_extractor_func,
    vocabulary_model,
    idf_scores,
    num_to_rerank=50,  # Paper re-ranks up to 1000[cite: 216], smaller for demo
    min_inliers_spatial=4,  # Paper considers verification successful with >= 4 inliers [cite: 217]
    ransac_reproj_thresh=5.0,
    ratio_thresh=0.75,
):
    """
    Performs spatial re-ranking on an initial list of ranked images.
    """
    if query_keypoints_initial is None or query_descriptors_initial is None:
        print("Query features not available for spatial re-ranking.")
        return (
            initial_ranked_results_with_scores  # Return original if no query features
        )

    spatially_verified_results = []
    remaining_results_tuples = []  # Store original tuples (path, initial_score)

    # Only re-rank the top 'num_to_rerank' images
    images_to_process = initial_ranked_results_with_scores[:num_to_rerank]
    # Keep the rest to append later
    results_not_reranked_tuples = initial_ranked_results_with_scores[num_to_rerank:]

    for i, (candidate_path, initial_score) in enumerate(images_to_process):
        # print(f"Spatially verifying {i+1}/{len(images_to_process)}: {candidate_path}")
        candidate_kp, candidate_desc, _ = feature_extractor_func(candidate_path)

        if candidate_kp is None or candidate_desc is None:
            remaining_results_tuples.append(
                {
                    "path": candidate_path,
                    "original_score": initial_score,
                    "spatial_score": -1,
                    "inliers": 0,
                    "verified": False,
                }
            )
            continue

        # 1. Match features
        # query_matched_kp_pts: points from query image that have a match
        # candidate_matched_kp_pts: points from candidate image that have a match
        # good_matches: list of DMatch objects for good matches
        # query_desc_indices_for_good_matches: indices of query descriptors involved in good_matches
        # candidate_desc_indices_for_good_matches: indices of candidate descriptors involved in good_matches
        (
            query_matched_kp_pts,
            candidate_matched_kp_pts,
            good_matches,
            query_desc_indices_for_good_matches,
            _,
        ) = match_features(
            query_keypoints_initial,
            query_descriptors_initial,
            candidate_kp,
            candidate_desc,
            ratio_thresh=ratio_thresh,
        )

        if len(good_matches) < min_inliers_spatial:
            remaining_results_tuples.append(
                {
                    "path": candidate_path,
                    "original_score": initial_score,
                    "spatial_score": -2,
                    "inliers": len(good_matches),
                    "verified": False,
                }
            )
            continue

        # 2. Verify transform using RANSAC
        affine_matrix, num_estimated_inliers, inlier_mask = verify_transform(
            query_matched_kp_pts,
            candidate_matched_kp_pts,
            min_inliers_for_transform=min_inliers_spatial,  # RANSAC needs at least this many points
            ransac_reproj_thresh=ransac_reproj_thresh,
        )

        if affine_matrix is not None and num_estimated_inliers >= min_inliers_spatial:
            # Get the indices of the query descriptors that were part of the *RANSAC inlier* set
            # The inlier_mask corresponds to the 'good_matches' list

            actual_inlier_query_desc_indices = []
            for i_mask, mask_val in enumerate(inlier_mask):
                if mask_val:  # If this good_match is an inlier according to RANSAC
                    actual_inlier_query_desc_indices.append(
                        query_desc_indices_for_good_matches[i_mask]
                    )

            # 3. Calculate spatial score using IDF of inlier query words
            spatial_score = calculate_spatial_score(
                actual_inlier_query_desc_indices,
                query_descriptors_initial,
                vocabulary_model,
                idf_scores,
            )
            spatially_verified_results.append(
                {
                    "path": candidate_path,
                    "original_score": initial_score,
                    "spatial_score": spatial_score,
                    "inliers": num_estimated_inliers,
                    "verified": True,
                }
            )
        else:
            remaining_results_tuples.append(
                {
                    "path": candidate_path,
                    "original_score": initial_score,
                    "spatial_score": -3,
                    "inliers": (
                        num_estimated_inliers
                        if affine_matrix is not None
                        else len(good_matches)
                    ),
                    "verified": False,
                }
            )

    # Sort verified results: higher spatial_score is better, then more inliers
    # Paper: "place spatially verified images above unverified ones in the ranking" [cite: 218]
    spatially_verified_results.sort(
        key=lambda x: (x["spatial_score"], x["inliers"]), reverse=True
    )

    # Combine sorted verified results with remaining (unverified or failed verification) results
    # Unverified results can keep their original relative BoW ranking.
    remaining_results_tuples.sort(
        key=lambda x: x["original_score"]
    )  # original_score is L2 distance, so sort ascending

    final_ranked_list_tuples = spatially_verified_results + remaining_results_tuples

    # Append results that were not re-ranked at all
    for path, score in results_not_reranked_tuples:
        final_ranked_list_tuples.append(
            {
                "path": path,
                "original_score": score,
                "spatial_score": -99,
                "inliers": -1,
                "verified": False,
            }
        )

    # Convert back to list of (path, score_for_display)
    # For display, we can use a combined score or just show path based on new ranking
    # Here, we just return paths based on the new ranking. The original score is L2 distance.
    # Spatial score is sum of IDFs. Higher is better.

    # For simplicity, we'll return the same (path, original_L2_score_from_BoW) format,
    # but the ORDER is now determined by spatial verification.
    final_ranked_list_for_display = []
    for item in final_ranked_list_tuples:
        final_ranked_list_for_display.append(
            (item["path"], item["original_score"])
        )  # Or item['spatial_score'] if you want to display that

    # print(f"Spatial Re-ranking: Verified {len(spatially_verified_results)} images.")
    # for res in spatially_verified_results[:5]:
    #     print(f"  Path: {res['path']}, Spatial Score: {res['spatial_score']:.2f}, Inliers: {res['inliers']}")
    # print(f"Remaining (unverified/failed): {len(remaining_results_tuples)} images.")

    return final_ranked_list_for_display
