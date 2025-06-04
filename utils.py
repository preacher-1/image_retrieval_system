# image_retrieval_system/utils.py
import os
import pickle
from PIL import Image # For checking image integrity

def list_image_files(directory):
    """Lists all valid image files in a directory."""
    image_files = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_extensions):
            image_path = os.path.join(directory, filename)
            try:
                # Try to open the image to catch corrupted files
                img = Image.open(image_path)
                img.verify() # Verify CHUNKS -- this will raise an exception on bad files
                image_files.append(image_path)
            except (IOError, SyntaxError) as e:
                print(f"Skipping corrupted or invalid image {filename}: {e}")
    return image_files

def save_model(model, filepath):
    """Saves a model to a file using pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Loads a model from a file using pickle."""
    if not os.path.exists(filepath):
        print(f"Model file not found: {filepath}")
        return None
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model from {filepath}: {e}")
        return None