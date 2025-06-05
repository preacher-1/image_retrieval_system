# image_retrieval_system/indexer.py
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from feature_extractor import extract_sift_features
from tqdm import tqdm


def image_to_bow(descriptors, vocabulary_model):
    """
    Converts image descriptors to a Bag-of-Words vector based on the vocabulary.
    Each image is represented as a bag of visual words[cite: 18].
    """
    if descriptors is None or vocabulary_model is None:
        return None

    visual_words = vocabulary_model.predict(descriptors.astype(np.float32))
    bow_vector = np.zeros(vocabulary_model.n_clusters)
    for word in visual_words:
        bow_vector[word] += 1
    return bow_vector


def create_tfidf_index(
    image_paths, vocabulary_model, feature_extractor_func=extract_sift_features
):
    """
    Creates a TF-IDF weighted Bag-of-Words index for all images.
    The system uses tf-idf weighting[cite: 91].
    """
    if not image_paths or vocabulary_model is None:
        print("Error: Image paths or vocabulary model not provided.")
        return None, None, None

    doc_term_matrix_list = []
    valid_image_paths_for_index = []

    for image_path in tqdm(image_paths, desc="Creating TF-IDF index"):
        _, descriptors, _ = feature_extractor_func(image_path)
        if descriptors is not None:
            bow_vector = image_to_bow(descriptors, vocabulary_model)
            if bow_vector is not None:
                doc_term_matrix_list.append(bow_vector)
                valid_image_paths_for_index.append(image_path)
        else:
            print(f"Skipping {image_path} due to no descriptors.")

    if not doc_term_matrix_list:
        print("Error: No BoW vectors could be created for the index.")
        return None, None, None

    doc_term_matrix = np.array(doc_term_matrix_list)

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_matrix = tfidf_transformer.fit_transform(doc_term_matrix)

    print("TF-IDF index created.")
    return tfidf_matrix, tfidf_transformer, valid_image_paths_for_index
