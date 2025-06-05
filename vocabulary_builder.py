from sklearn.cluster import MiniBatchKMeans
import numpy as np

def build_vocabulary(all_descriptors, num_clusters=100, random_state=42):
    """
    Builds a visual vocabulary using MiniBatchKMeans clustering.
    """
    if all_descriptors is None or len(all_descriptors) == 0:
        print("Error: No descriptors provided to build vocabulary.")
        return None

    all_descriptors = all_descriptors.astype(np.float32)

    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=random_state,
        batch_size=max(2048, 3 * num_clusters),
        n_init="auto",
        max_iter=100,
    )
    kmeans.fit(all_descriptors)
    print(f"MiniBatchKMeans Vocabulary built with {num_clusters} visual words.")
    return kmeans