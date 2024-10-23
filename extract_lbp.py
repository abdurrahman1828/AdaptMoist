# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May 13 09:50:26 2021
@reference: Ojala, A Comparative Study of Texture Measures with Classification on Feature Distributions
            Ojala, Gray Scale and Roation Invariaant Texture Classification with Local Binary Patterns
==============================================================================
"""


import os
import cv2
import pandas as pd
import numpy as np
from skimage import feature

def _energy(x):
    return np.multiply(x, x).sum()


def _entropy(x):
    return -np.multiply(x, np.log(x + 1e-16)).sum()


def lbp_features(f, mask, P=[8, 16, 24], R=[1, 2, 3]):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    P : list, optional
        Number of points in neighborhood. The default is [8,16,24].
    R : list, optional
        Radius/Radii. The default is [1,2,3].

    Returns
    -------
    features : numpy ndarray
        Energy and entropy of LBP image (2 x 1).
    labels : list
        Labels of features.
    '''

    if mask is None:
        mask = np.ones(f.shape)

    P = np.array(P)
    R = np.array(R)
    n = P.shape[0]
    mask_ravel = mask.ravel()
    features = []
    labels = []

    for i in range(n):
        lbp = feature.local_binary_pattern(f, P[i], R[i], 'uniform')
        lbp_ravel = lbp.ravel()
        roi = lbp_ravel[mask_ravel.astype(bool)]
        feats = np.zeros(2, np.double)
        feats[0] = _energy(roi) / roi.sum()
        feats[1] = _entropy(roi) / roi.sum()
        features.append(feats)
        labels.append('LBP_R_' + str(R[i]) + '_P_' + str(P[i]) + '_energy')
        labels.append('LBP_R_' + str(R[i]) + '_P_' + str(P[i]) + '_entropy')

    features = np.array(features, np.double).ravel()

    return features, labels


# Define paths
base_directory = 'D:\\Abdur\\Woodchip\\enviva_class\\batch_1\\data' #'D:\\Abdur\\Woodchip\\enviva_whole' #'D:/Abdur/Woodchip/batch_2_classes/data'  # Replace with your directory path containing class folders
output_csv = 'data/lbp_features_enviva_b1.csv'  # Path to save the CSV file

# Initialize a list to store the feature data
data = []
labels = None

# Traverse the class folders and images
for class_folder in os.listdir(base_directory):
    class_path = os.path.join(base_directory, class_folder)
    if not os.path.isdir(class_path):
        continue

    # Process each image in the class folder
    for image_file in os.listdir(class_path):
        print(image_file)
        image_path = os.path.join(class_path, image_file)
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        # Read the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (640, 480))
        if image is None:
            print(f"Error reading image {image_path}")
            continue

        # Define mask as None to use the entire image
        mask = None

        # Extract LBP features using the provided function
        features, labels = lbp_features(image, mask)

        # Combine features into a single row and add the label (folder name)
        features = list(features)
        data.append([image_file] + features + [class_folder])  # Add class label

# Save the features to CSV
if data and labels:
    columns = ["Image"] + labels + ["Label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"LBP features successfully saved to {output_csv}")
else:
    print("No data to save or labels are missing.")
