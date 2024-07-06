import streamlit as st
import cv2
import numpy as np
import os
from skimage import io
from skimage.color import rgb2gray
import mahotas.features as features
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial import distance
from BiT import bio_taxo 
import matplotlib.pyplot as plt


def manhattan_distance(v1, v2):
    return np.sum(np.abs(np.array(v1).astype('float') - np.array(v2).astype('float')))

def ecludien_distance(v1, v2):
    return np.sqrt(np.sum((np.array(v1).astype('float') - np.array(v2).astype('float'))**2))

def chebyshev_distance(v1, v2):
    return np.max(np.abs(np.array(v1).astype('float') - np.array(v2).astype('float')))

def canberra_distance(v1, v2):
    return distance.canberra(v1, v2)


def glcm(image_data):
    glcm_matrix = graycomatrix(image_data, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    return [graycoprops(glcm_matrix, prop).mean() for prop in ['dissimilarity', 'contrast', 'correlation', 'energy', 'homogeneity']]

def haralick(image_data):
    return features.haralick(image_data).mean(0).tolist()

def bitdesc(image_data):
    return bio_taxo(image_data)


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

def retrieve_similare_image(feature_db, query_features, distance_type, num_results):
    similar_images = []
    for instance in feature_db:
        features, label, img_path = instance[:-2], instance[-2], instance[-1]
        if distance_type == 'ecludien':
            dist = ecludien_distance(query_features, features)
        elif distance_type == 'manhattan':
            dist = manhattan_distance(query_features, features)
        elif distance_type == 'chebyshev':
            dist = chebyshev_distance(query_features, features)
        elif distance_type == 'canberra':
            dist = canberra_distance(query_features, features)
        similar_images.append((img_path, dist, label))
    similar_images.sort(key=lambda x: x[1])
    return similar_images[:num_results]


def load_features(descriptor):
    features_path = f'extracted/{descriptor}.npy'
    if os.path.exists(features_path):
        return np.load(features_path, allow_pickle=True)
    else:
        st.error(f"Feature file not found: {features_path}")
        return None

st.title('Projet1 Images Similaires')

# Sidebar
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg","bmp"])
descriptor_type = st.sidebar.selectbox("Select Descriptor Type", ['glcm', 'haralick', 'bitdesc','haralick_glcm', 'bit_glcm', 'bit_haralick'])
distance_type = st.sidebar.selectbox("Select Distance Metric", ['ecludien', 'manhattan', 'chebyshev', 'canberra'])
num_results = st.sidebar.slider("Number of Similar Images", min_value=1, max_value=100, value=1)

if uploaded_file:
    uploaded_image = io.imread(uploaded_file)
    uploaded_image_gray = rgb2gray(uploaded_image) * 255
    uploaded_image_gray = uploaded_image_gray.astype('uint8')
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    st.session_state['uploaded_image'] = uploaded_image_gray

if 'uploaded_image' in st.session_state:
    if st.button("Find Similar Images"):
        if descriptor_type == 'glcm':
            image_features = glcm(st.session_state['uploaded_image'])
        elif descriptor_type == 'haralick':
            image_features = haralick(st.session_state['uploaded_image'])
        elif descriptor_type == 'bitdesc':
            image_features = bitdesc(st.session_state['uploaded_image'])
        elif descriptor_type == 'haralick_glcm':
            image_features = haralick_glcm(st.session_state['uploaded_image'])
        elif descriptor_type == 'bit_glcm':
            image_features = bit_glcm(st.session_state['uploaded_image'])
        elif descriptor_type == 'bit_haralick':
            image_features = bit_haralick(st.session_state['uploaded_image'])

        dataset_features = load_features(descriptor_type)
        if dataset_features is not None:
            similar_images = retrieve_similare_image(dataset_features, image_features, distance_type, num_results)
            st.write("Top Similar Images:")
            col1, col2, col3, col4, col5 = st.columns(5)
            for i, (img_path, dist, label) in enumerate(similar_images):
                col = i % 5
                if col == 0:
                    container = col1
                elif col == 1:
                    container = col2
                elif col == 2:
                    container = col3
                elif col == 3:
                    container = col4
                else:
                    container = col5
                with container:
                    st.image(img_path, caption=f"{label}, Distance: {dist}", use_column_width=True)

            # Histogramme
            labels = [label for _, _, label in similar_images]
            unique_labels, counts = np.unique(labels, return_counts=True)
            plt.figure(figsize=(10, 5))
            plt.bar(unique_labels, counts, color='skyblue')
            plt.title('Distribution des Images similaires')
            plt.xlabel('Class')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            st.pyplot(plt)
        else:
            st.write("Error while loading dataset.")
