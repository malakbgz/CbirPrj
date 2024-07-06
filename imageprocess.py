import os
import cv2
import numpy as np
from descriptors import glcm, Bitdesc, haralick, Bitdesc_glcm, haralick_bitdesc

def extract_features_Bitdesc(image):
   
    img_array = np.array(image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    if img_gray is not None:
        features = Bitdesc(img_gray)
        return features
    else:
        pass

def extract_features_GLCM(image):
    
    img_array = np.array(image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    if img_gray is not None:
        features = glcm(img_gray)
        return features
    else:
        pass

def extract_features_Haralick(image):
   
    img_array = np.array(image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    if img_gray is not None:
        features = haralick(img_gray)
        return features
    else:
        pass

def extract_features_haralick_Bitdesc(image):
   
    img_array = np.array(image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    if img_gray is not None:
        features = haralick_bitdesc(img_gray)
        return features
    else:
        pass

def extract_features_Bitdesc_GLCM(image):
    
    img_array = np.array(image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    if img_gray is not None:
        features = Bitdesc_glcm(img_gray)
        return features
    else:
        pass