# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: May  8 15:18:53 2021
@reference: Weszka, A Comparative Study of Texture Measures for Terrain Classification
            Wu, Texture Features for Classification
==============================================================================
"""

import os
import cv2
import pandas as pd
import numpy as np



def _fft2(x, msk):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    msk : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else.

    Returns
    -------
    F : numpy ndarray
        2D Fourier Transform inside msk.
    '''
    N1, N2 = x.shape
    F = np.zeros((N1, N2), np.complex64)

    for i in range(1, N2 + 1):
        sp = 0
        ep = 0
        j = 1
        while (j < N1):
            while (msk[j - 1, i - 1] == 1) & (j < N1):
                if (sp == 0):
                    sp = j
                j += 1
            if (sp > 0) & (ep == 0):
                if (j < N1):
                    ep = j - 1
                else:
                    ep = j
                F[(sp - 1):(ep), i - 1] = np.fft.fft(x[(sp - 1):(ep), i - 1])
                sp = 0
                ep = 0
            while (msk[j - 1, i - 1] == 0) & (j < N1):
                j += 1

    F = F.T
    msk = msk.T

    for i in range(1, N1 + 1):
        sp = 0
        ep = 0
        j = 1
        while (j < N2):
            while (msk[j - 1, i - 1] == 1) & (j < N2):
                if (sp == 0):
                    sp = j
                j += 1
            if (sp > 0) & (ep == 0):
                if (j < N1):
                    ep = j - 1
                else:
                    ep = j
                F[(sp - 1):(ep), i - 1] = np.fft.fft(F[(sp - 1):(ep), i - 1])
                sp = 0
                ep = 0
            while (msk[j - 1, i - 1] == 0) & (j < N2):
                j += 1

    F = F.T
    msk = msk.T
    return F


def fps(f, mask):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.

    Returns
    -------
    features : numpy ndarray
        1)Radial Sum, 2)Angular Sum.
    labels : list
        Labels of features.
    '''

    if mask is None:
        mask = np.ones(f.shape)

    # 1) Labels
    labels = ["FPS_RadialSum", "FPS_AngularSum"]

    # 2) Parameters
    f = np.array(f, np.double)
    mask = np.array(mask, np.double)

    # 3) Calculate Features
    area = mask.sum()
    F = _fft2(f, mask)
    F_real = np.real(F)
    F_imag = np.imag(F)
    # F_real = np.multiply(F_real, mask)
    # F_imag = np.multiply(F_imag, mask)
    features = np.zeros(2, np.double)
    features[0] = np.sqrt(sum(sum(np.multiply(F_real, F_real))) / area)
    features[1] = np.sqrt(sum(sum(np.multiply(F_imag, F_imag))) / area)
    return features, labels


# Define paths
base_directory =  'D:\\Abdur\\Woodchip\\enviva_class\\batch_1\\data' #'D:\\Abdur\\Woodchip\\enviva_whole' #'D:/Abdur/Woodchip/batch_2_classes/data'  # Replace with your directory path containing class folders
output_csv = 'data/fps_features_enviva_b1.csv'  # Path to save the CSV file

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
        if image is None:
            print(f"Error reading image {image_path}")
            continue

        # Define mask as None to use the entire image
        mask = None

        # Extract FPS features using the provided function
        features, labels = fps(image, mask)

        # Combine features into a single row and add the label (folder name)
        features = list(features)
        data.append([image_file] + features + [class_folder])  # Add class label

# Save the features to CSV
if data and labels:
    columns = ["Image"] + labels + ["Label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"FPS features successfully saved to {output_csv}")
else:
    print("No data to save or labels are missing.")
