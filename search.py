import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class ImageRetriever:
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
        self.image_features = {}
        self.image_paths = []

    def add_image(self, image_path, quantized_features):
        self.image_features[image_path] = np.bincount(quantized_features, minlength=self.vocabulary_size)
        self.image_paths.append(image_path)

    def retrieve_images(self, query_features, top_k=5):
        query_vector = np.bincount(query_features, minlength=self.vocabulary_size)
        scores = {}
        for image_path, feature_vector in self.image_features.items():
            score = euclidean_distances([query_vector], [feature_vector])[0][0]
            scores[image_path] = -score  # Negative distance for ascending sorting
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [image_path for image_path, _ in sorted_scores[:top_k]]

    def spatial_re_ranking(self, query_keypoints, query_descriptors, candidate_image_paths, top_k=5):
        # Placeholder for spatial re-ranking logic
        # Implement RANSAC-based spatial verification here
        return candidate_image_paths[:top_k]