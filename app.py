import os
import cv2
import numpy as np
import mahotas.features as features
from skimage.feature import graycomatrix, graycoprops
from BiT import bio_taxo  

def read_image(image_path):
    """Read and preprocess an image in grayscale."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return img
    except Exception as e:
        print(f"Error reading image: {e}")
        return None


def glcm(image_data):
    """Extract GLCM features from an image."""
    try:
        glcm_matrix = graycomatrix(image_data, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
        return [graycoprops(glcm_matrix, prop).mean() for prop in ['dissimilarity', 'contrast', 'correlation', 'energy', 'homogeneity']]
    except Exception as e:
        print(f"Error extracting GLCM features: {e}")
        return None

def haralick(image_data):
    """Extract Haralick features from an image."""
    try:
        return features.haralick(image_data).mean(0).tolist()
    except Exception as e:
        print(f"Error extracting Haralick features: {e}")
        return None

def bitdesc(image_data):
    """Extract BiT features from an image."""
    try:
        return bio_taxo(image_data)
    except Exception as e:
        print(f"Error extracting BiT features: {e}")
        return None

# Combined descriptors
def haralick_glcm(image_data):
    haralick_features = haralick(image_data)
    glcm_features = glcm(image_data)
    return haralick_features + glcm_features if haralick_features and glcm_features else None

def bit_glcm(image_data):
    bit_features = bitdesc(image_data)
    glcm_features = glcm(image_data)
    return bit_features + glcm_features if bit_features and glcm_features else None

def bit_haralick(image_data):
    bit_features = bitdesc(image_data)
    haralick_features = haralick(image_data)
    return bit_features + haralick_features if bit_features and haralick_features else None

def process_images(main_folder_path, feature_function, descriptor_name):
    """Process all images in all subfolders and extract features."""
    all_features = []
    for root, dirs, files in os.walk(main_folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image_data = read_image(image_path)
                features = feature_function(image_data)
                if features is not None:
                    features.append(root.split(os.sep)[-1]) 
                    features.append(image_path)  
                    all_features.append(features)
                print(f"Processed {image_path}")
    
  
    output_dir = "extracted"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{descriptor_name}.npy")
    np.save(output_file, np.array(all_features))
    print(f"Data for descriptor '{descriptor_name}' stored successfully in {output_file}")


def process_datasets(main_folder_path):
    
    descriptors = [
        (glcm, 'glcm'),
        (haralick, 'haralick'),
        (bitdesc, 'bitdesc'),
        (haralick_glcm, 'haralick_glcm'),
        (bit_glcm, 'bit_glcm'),
        (bit_haralick, 'bit_haralick')
    ]
    for feature_function, descriptor_name in descriptors:
        process_images(main_folder_path, feature_function, descriptor_name)

main_folder_path = './datasets'
process_datasets(main_folder_path)
