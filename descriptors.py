from skimage.feature import graycomatrix, graycoprops
from BiT import bio_taxo
import mahotas.features as features

def glcm(data):
    glcm = graycomatrix(data, [2], [0],None, symmetric=True,normed=True)
    diss = graycoprops(glcm, 'dissimilarity')[0,0]
    cont = graycoprops(glcm, 'contrast')[0,0]
    corr = graycoprops(glcm, 'correlation')[0,0]
    ener = graycoprops(glcm, 'energy')[0,0]
    homo = graycoprops(glcm, 'homogeneity')[0,0]
    return [diss, cont, corr, ener, homo]

def Bitdesc(data):
    return bio_taxo(data)

def haralick(data):
    return features.haralick(data).mean(0).tolist() 
 
def Bitdesc_glcm(image):
    return Bitdesc(image) + glcm(image)
 
def haralick_bitdesc(image):
    return haralick(image) + Bitdesc(image)

