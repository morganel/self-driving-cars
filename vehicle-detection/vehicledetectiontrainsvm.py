import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
import scipy as sc

import pickle

def color_hist(img, nbins=32, bins_range=(0, 256)):
    '''
    Computes color histogram features  
    '''
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    '''
    Returns HOG features and visualization
    '''
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def extract_features(imgs, cspace='RGB', cspacehog='HLS', 
                        hist_bins=32, hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    
    '''
    Extract features from a list of images.
    It converts the image to color space "cspace" for the color histogram and "cspacehog" for the HOG.
    It then performs color histogram (calling color_hist() function) and HOG (calling get_hog_features() function) on all 3 channels.
    It concatenates all the features together before returning them.
    '''
    
    # Create a list to append feature vectors to
    features = []
    convert =  isinstance(imgs[0], str)

    # Iterate through the list of images
    for img in imgs:
        # Read in each one by one
        if convert:
            image = sc.misc.imread(img) 
        else:
            image = img
        
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      
            
        if cspacehog != 'RGB':
            if cspacehog == 'HSV':
                feature_image_hog = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspacehog == 'LUV':
                feature_image_hog = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspacehog == 'HLS':
                feature_image_hog = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspacehog == 'YUV':
                feature_image_hog = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspacehog == 'YCrCb':
                feature_image_hog = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image_hog = np.copy(image)      
            
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        
        # Call get_hog_features() with vis=False, feature_vec=True
        hog_features_0 = get_hog_features(feature_image_hog[:,:,0], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        hog_features_1 = get_hog_features(feature_image_hog[:,:,1], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        hog_features_2 = get_hog_features(feature_image_hog[:,:,2], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(np.concatenate((hog_features_0, hog_features_1, hog_features_2, hist_features)))
    # Return list of feature vectors
    return features

def extract_feature_1_img(img, cspace='RGB', cspacehog='HLS',
                        hist_bins=32, hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    '''
    FOR VISUALIZATION ONLY
    Similar to extract_features() function
    '''
    
    # Create a list to append feature vectors to
    features = []
    convert =  isinstance(img, str)

    # Read in each one by one
    if convert:
        image = sc.misc.imread(img) 
    else:
        image = img

    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)      

    if cspacehog != 'RGB':
        if cspacehog == 'HSV':
            feature_image_hog = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspacehog == 'LUV':
            feature_image_hog = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspacehog == 'HLS':
            feature_image_hog = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspacehog == 'YUV':
            feature_image_hog = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspacehog == 'YCrCb':
            feature_image_hog = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image_hog = np.copy(image)      

    fig = plt.figure(figsize=(12,12))

    plt.subplot(141)
    plt.imshow(image)
    plt.title('Original Car Image')
    plt.subplot(142)
    plt.imshow(feature_image[:,:,0], cmap = 'gray')
    plt.title('Y')
    plt.subplot(143)
    plt.imshow(feature_image[:,:,1], cmap = 'gray')
    plt.title('Cr')
    plt.subplot(144)
    plt.imshow(feature_image[:,:,2], cmap = 'gray')
    plt.title('Cb')
    
    plt.show()
        
    # Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    # Call get_hog_features() with vis=False, feature_vec=True
    hog_features_0, hog_image_0 = get_hog_features(feature_image_hog[:,:,0], orient, 
                    pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    hog_features_1, hog_image_1 = get_hog_features(feature_image_hog[:,:,1], orient, 
                    pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    hog_features_2, hog_image_2 = get_hog_features(feature_image_hog[:,:,2], orient, 
                    pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    
    print(np.min(hog_features_0), np.max(hog_features_0),np.min(hog_features_1), np.max(hog_features_1),np.min(hog_features_2), np.max(hog_features_2))
    print(np.max(hist_features))
    print(hog_features_0.shape, hog_features_1.shape, hog_features_2.shape, hist_features.shape)
    
    # Return list of feature vectors
    return hog_image_0, hog_image_1, hog_image_2, hist_features

def get_features_parameters():
    '''
    Returns the final set of parameters to compute the features.
    It is called from vehicle-detection-train-svm.ipynb and vehicle-detection.ipynb 
    '''
    d = dict()
    d['_CSPACE_'] = 'YCrCb'
    d['_CSPACE_HOG_'] = 'YCrCb'
    d['_HIST_BINS_'] = 32
    d['_HIST_RANGE_'] = (0, 256)
    d['_ORIENT_'] = 8
    d['_PIX_PER_CELL_'] = 8
    d['_CELL_PER_BLOCK_'] = 2
    d['_HOG_CHANNEL_'] = 1
    return d

