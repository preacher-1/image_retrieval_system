# image_retrieval_system/vocabulary_builder.py
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from sklearn.utils import check_random_state, shuffle
from sklearn.neighbors import NearestNeighbors  # For approximate assignment step in AKM
from tqdm import tqdm


class AKMeans:
    """
    Approximate K-Means (AKM) implementation inspired by the paper.
    "Object retrieval with large vocabularies and fast spatial matching" by Philbin et al.
    This implementation uses scikit-learn's NearestNeighbors with a k-d tree
    to approximate the fast assignment step. The paper's original AKM uses a
    forest of randomized k-d trees[cite: 110].
    """

    def __init__(
        self,
        n_clusters,
        max_iter=100,
        random_state=None,
        tol=1e-4,
        ann_algorithm="kd_tree",
        n_jobs_ann=-1,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state_ = check_random_state(
            random_state
        )  # Ensure random_state is a RandomState instance
        self.tol = tol
        self.ann_algorithm = ann_algorithm  # Algorithm for NearestNeighbors
        self.n_jobs_ann = n_jobs_ann  # Parallel jobs for NearestNeighbors

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = 0.0
        self.n_iter_ = 0

    def _initialize_centers(self, X):
        """
        Initializes cluster centers by randomly picking points from X.
        """
        n_samples = X.shape[0]
        # Ensure we don't request more clusters than samples
        if self.n_clusters > n_samples:
            raise ValueError(
                f"Number of clusters ({self.n_clusters}) cannot be greater than "
                f"the number of samples ({n_samples})."
            )

        # Use sklearn's shuffle for robust random sampling
        shuffled_indices = shuffle(
            np.arange(n_samples), random_state=self.random_state_
        )
        indices = shuffled_indices[: self.n_clusters]
        self.cluster_centers_ = X[
            indices
        ].copy()  # Use .copy() to avoid issues with views

    def _assign_points_ann(self, X):
        """
        Assigns points to the nearest cluster center using Approximate Nearest Neighbors.
        The paper uses a forest of randomized k-d trees[cite: 110].
        We use scikit-learn's NearestNeighbors as an approximation.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Cluster centers not initialized.")

        # Ensure cluster_centers_ are float32, if X is (common for SIFT)
        # This helps prevent potential type mismatches in NearestNeighbors
        if X.dtype == np.float32 and self.cluster_centers_.dtype != np.float32:
            self.cluster_centers_ = self.cluster_centers_.astype(np.float32)

        ann_index = NearestNeighbors(
            n_neighbors=1, algorithm=self.ann_algorithm, n_jobs=self.n_jobs_ann
        )
        ann_index.fit(self.cluster_centers_)

        distances, labels = ann_index.kneighbors(X)
        self.labels_ = labels.ravel()

        # Calculate inertia (sum of squared distances to closest center)
        self.inertia_ = np.sum(distances**2)

    def _update_centers(self, X):
        """
        Updates cluster centers based on the mean of assigned points.
        Handles empty clusters by reassigning them from the largest cluster's points.
        """
        new_centers = np.zeros_like(self.cluster_centers_)
        counts = np.zeros(self.n_clusters, dtype=np.int32)  # Use np.int32 for counts

        # Sum up points for each cluster
        # This can be done more efficiently with np.add.at or similar for large X
        for i in range(X.shape[0]):
            label = self.labels_[i]
            new_centers[label] += X[i]
            counts[label] += 1

        empty_clusters = []
        for k in range(self.n_clusters):
            if counts[k] > 0:
                new_centers[k] /= counts[k]
            else:
                empty_clusters.append(k)

        # Handle empty clusters (if any)
        if len(empty_clusters) > 0:
            # Find points from the largest cluster to re-assign to empty clusters
            # This is a common strategy to prevent cluster collapse.
            # Other strategies: pick random points, pick points furthest from their centers.
            non_empty_counts = counts[counts > 0]
            if len(non_empty_counts) > 0:  # Check if there are any non-empty clusters
                largest_cluster_idx = np.argmax(counts)
                points_from_largest_cluster = X[self.labels_ == largest_cluster_idx]

                if points_from_largest_cluster.shape[0] >= len(empty_clusters):
                    # Shuffle points from largest cluster
                    points_from_largest_cluster = shuffle(
                        points_from_largest_cluster, random_state=self.random_state_
                    )
                    for i, k_empty in enumerate(empty_clusters):
                        new_centers[k_empty] = points_from_largest_cluster[i]
                else:  # Fallback: re-initialize empty clusters randomly from X
                    shuffled_X_indices = shuffle(
                        np.arange(X.shape[0]), random_state=self.random_state_
                    )
                    points_for_reinit = X[shuffled_X_indices[: len(empty_clusters)]]
                    for i, k_empty in enumerate(empty_clusters):
                        new_centers[k_empty] = points_for_reinit[i]

            else:  # All clusters became empty (highly unlikely with proper init) - reinitialize all
                # This indicates a serious issue if reached.
                print(
                    "Warning: All clusters became empty. Re-initializing all centers."
                )
                self._initialize_centers(X)  # Re-initialize all centers
                return self.cluster_centers_  # Return newly initialized centers

        return new_centers

    def fit(self, X):
        """
        Computes AK-Means clustering.
        X should be a 2D array like (n_samples, n_features)
        """
        if X is None or X.shape[0] == 0:
            raise ValueError("Input data X cannot be empty.")
        if X.ndim != 2:
            raise ValueError("Input data X must be 2-dimensional.")

        # Ensure X is float32 for consistency, esp. if SIFT descriptors are used.
        X = X.astype(np.float32)

        self._initialize_centers(X)

        for i_iter in tqdm(
            range(self.max_iter), desc="AKM Iterations", unit="iter", disable=False
        ):
            self.n_iter_ = i_iter + 1
            prev_centers = self.cluster_centers_.copy()

            # 1. Assignment Step (Approximate)
            self._assign_points_ann(X)

            # 2. Update Step
            self.cluster_centers_ = self._update_centers(X)

            # Check for convergence
            center_shift_sq = np.sum((self.cluster_centers_ - prev_centers) ** 2)

            # print(f"AKM Iteration {self.n_iter_}, Center shift sq: {center_shift_sq:.4f}, Inertia: {self.inertia_:.2f}")
            if center_shift_sq < self.tol:
                print(
                    f"AKM Converged at iteration {self.n_iter_} with center shift {center_shift_sq:.2e}."
                )
                break

        if self.n_iter_ == self.max_iter and center_shift_sq >= self.tol:
            print(
                f"AKM did not converge within {self.max_iter} iterations. Final center shift: {center_shift_sq:.2e}"
            )

        return self

    def predict(self, X):
        """
        Predicts the closest cluster each sample in X belongs to.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        X = X.astype(np.float32)  # Ensure type consistency

        # For prediction, use the same ANN approach
        # This is consistent with how assignments are made during fitting.
        ann_index = NearestNeighbors(
            n_neighbors=1, algorithm=self.ann_algorithm, n_jobs=self.n_jobs_ann
        )

        # Ensure cluster_centers_ are float32 for fitting ANN index
        if self.cluster_centers_.dtype != np.float32:
            self.cluster_centers_ = self.cluster_centers_.astype(np.float32)

        ann_index.fit(self.cluster_centers_)
        _, labels = ann_index.kneighbors(X)
        return labels.ravel()


def build_vocabulary_akm(
    all_descriptors, num_clusters=100, random_state=42, max_iter=100, tol=1e-4
):
    """
    Builds a visual vocabulary using Approximate K-Means (AKM).
    """
    if all_descriptors is None or len(all_descriptors) == 0:
        print("Error: No descriptors provided to build AKM vocabulary.")
        return None

    # Ensure descriptors are float32, as SIFT descriptors often are.
    all_descriptors = all_descriptors.astype(np.float32)

    akm_model = AKMeans(
        n_clusters=num_clusters, random_state=random_state, max_iter=max_iter, tol=tol
    )
    print(f"Starting AKM clustering with {num_clusters} clusters...")
    akm_model.fit(all_descriptors)

    if akm_model.cluster_centers_ is not None:
        print(
            f"AKM Vocabulary built with {akm_model.n_clusters} visual words after {akm_model.n_iter_} iterations."
        )
    else:
        print("AKM failed to build vocabulary.")
        return None

    return akm_model


# --- Original MiniBatchKMeans function for reference/fallback ---
def build_vocabulary(all_descriptors, num_clusters=100, random_state=42):
    """
    Builds a visual vocabulary using MiniBatchKMeans clustering.
    (This is the original function using MiniBatchKMeans)
    """
    if all_descriptors is None or len(all_descriptors) == 0:
        print("Error: No descriptors provided to build vocabulary.")
        return None

    all_descriptors = all_descriptors.astype(np.float32)

    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=random_state,
        batch_size=max(2048, 3 * num_clusters),  # Ensure batch_size is adequate
        n_init="auto",
        max_iter=100,
    )
    kmeans.fit(all_descriptors)
    print(f"MiniBatchKMeans Vocabulary built with {num_clusters} visual words.")
    return kmeans
