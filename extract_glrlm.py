# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@author: https://github.com/eiproject/lib-GLRLM-Python3/blob/master/lib-GLRLM/libpreprocessing.py
@author: https://github.com/szhHr/Gray-Level-Run-Length-Matrix-to-get-image-feature/blob/master/GrayRumatrix.py
@date: Sat May  8 17:00:35 2021
@reference: Gallowway, Texture Analysis using Gray Level Run Lengths
==============================================================================
"""

import os
import cv2
import pandas as pd
import numpy as np


def glrlm_0(f, mask, grayLevel=5, runLength=5, skipFirstRow=True):
    degree0Matrix = np.zeros([grayLevel, runLength])
    counter = 0
    for y in range(f.shape[0]):
        for x in range(f.shape[1]):
            nowVal = f[y][x]
            if x + 1 >= f.shape[1]:
                nextVal = None
            else:
                nextVal = f[y][x + 1]
            if nextVal != nowVal and counter == 0:
                degree0Matrix[int(nowVal)][counter] += 1
            elif nextVal == nowVal:
                counter += 1
            elif nextVal != nowVal and counter != 0:
                degree0Matrix[int(nowVal)][counter] += 1
                counter = 0
    return degree0Matrix[1:, :] if skipFirstRow else degree0Matrix


def glrlm_90(f, mask, grayLevel=5, runLength=5, skipFirstRow=True):
    degree90Matrix = np.zeros([grayLevel, runLength])
    counter = 0
    for x in range(f.shape[1]):
        for y in range(f.shape[0]):
            nowVal = f[y][x]
            if y + 1 >= f.shape[0]:
                nextVal = None
            else:
                nextVal = f[y + 1][x]
            if nextVal != nowVal and counter == 0:
                degree90Matrix[int(nowVal)][counter] += 1
            elif nextVal == nowVal:
                counter += 1
            elif nextVal != nowVal and counter != 0:
                degree90Matrix[int(nowVal)][counter] += 1
                counter = 0
    return degree90Matrix[1:, :] if skipFirstRow else degree90Matrix


def glrlm_45(f, mask, grayLevel=5, runLength=5, skipFirstRow=True):
    degree45Matrix = np.zeros([grayLevel, runLength])
    for y in range(f.shape[0]):
        counter = 0
        i_range = max(f.shape)
        for i in range(i_range):
            y1 = y - i
            if i >= f.shape[1] or y1 < 0:
                break
            else:
                nowVal = f[y1][i]
            if y1 - 1 < 0 or i + 1 >= f.shape[1]:
                nextVal = None
            else:
                nextVal = f[y1 - 1][i + 1]
            if nextVal != nowVal and counter == 0:
                degree45Matrix[int(nowVal)][counter] += 1
            elif nextVal == nowVal:
                counter += 1
            elif nextVal != nowVal and counter != 0:
                degree45Matrix[int(nowVal)][counter] += 1
                counter = 0
    for x in range(f.shape[1]):
        if x == f.shape[1] - 1:
            break
        counter = 0
        i_range = max(f.shape)
        for i in range(i_range):
            y_i = -1 - i
            x_i = -1 + i - x
            if x_i >= 0 or y_i <= -1 - f.shape[0]:
                break
            else:
                nowVal = f[y_i][x_i]
            if y_i - 1 <= -(f.shape[0] + 1) or x_i + 1 >= 0:
                nextVal = None
            else:
                nextVal = f[y_i - 1][x_i + 1]
            if nextVal != nowVal and counter == 0:
                degree45Matrix[int(nowVal)][counter] += 1
            elif nextVal == nowVal:
                counter += 1
            elif nextVal != nowVal and counter != 0:
                degree45Matrix[int(nowVal)][counter] += 1
                counter = 0
    degree45Matrix[0, 1:] = 0
    return degree45Matrix[1:, :] if skipFirstRow else degree45Matrix


def glrlm_135(f, mask, grayLevel=5, runLength=5, skipFirstRow=True):
    degree135Matrix = np.zeros([grayLevel, runLength])
    for y in range(f.shape[0]):
        counter = 0
        i_range = max(f.shape)
        for i in range(i_range):
            y1 = y + i
            if y1 >= f.shape[0] or i >= f.shape[1]:
                break
            else:
                nowVal = f[y1][i]
                if y1 >= f.shape[0] - 1 or i >= f.shape[1] - 1:
                    nextVal = None
                else:
                    nextVal = f[y1 + 1][i + 1]
                if nextVal != nowVal and counter == 0:
                    degree135Matrix[int(nowVal)][counter] += 1
                elif nextVal == nowVal:
                    counter += 1
                elif nextVal != nowVal and counter != 0:
                    degree135Matrix[int(nowVal)][counter] += 1
                    counter = 0
    for x in range(f.shape[1]):
        if x == 0:
            continue
        i_range = max(f.shape)
        counter = 0
        for i in range(i_range):
            x1 = x + i
            if i >= f.shape[0] or x1 >= f.shape[1]:
                break
            else:
                nowVal = f[i][x1]
            if i >= f.shape[0] - 1 or x1 >= f.shape[1] - 1:
                nextVal = None
            else:
                nextVal = f[i + 1][x1 + 1]
            if nextVal != nowVal and counter == 0:
                degree135Matrix[int(nowVal)][counter] += 1
            elif nextVal == nowVal:
                counter += 1
            elif nextVal != nowVal and counter != 0:
                degree135Matrix[int(nowVal)][counter] += 1
                counter = 0
    degree135Matrix[0, 1:] = 0
    return degree135Matrix[1:, :] if skipFirstRow else degree135Matrix


def _apply_over_degree(function, x1, x2):
    if function == np.divide:
        x2 = x2 + 1e-16
    rows, cols, nums = x1.shape
    result = np.ndarray((rows, cols, nums))
    for i in range(nums):
        result[:, :, i] = function(x1[:, :, i], x2)
        result[result == np.inf] = 0
        result[np.isnan(result)] = 0
    return result


def _calculate_ij(rlmatrix):
    gray_level, run_length, _ = rlmatrix.shape
    I, J = np.ogrid[0:gray_level, 0:run_length]
    return I, J + 1


def _calculate_s(rlmatrix):
    return np.apply_over_axes(np.sum, rlmatrix, axes=(0, 1))[0, 0]


def glrlm(f, mask, Ng=256):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else.
    Ng : int, optional
        Image number of gray values. The default is 256.

    Returns
    -------
    mat : numpy ndarray
        GLRL Matrices for 0, 45, 90 and 135 degrees.
    '''
    runLength = max(f.shape)
    mat0 = glrlm_0(f, mask, grayLevel=Ng, runLength=runLength)
    mat45 = glrlm_45(f, mask, grayLevel=Ng, runLength=runLength)
    mat90 = glrlm_90(f, mask, grayLevel=Ng, runLength=runLength)
    mat135 = glrlm_135(f, mask, grayLevel=Ng, runLength=runLength)
    mat = np.dstack((mat0, mat45, mat90, mat135))
    return mat


def glrlm_features(f, mask, Ng=256):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    Ng : int, optional
        Image number of gray values. The default is 256.

    Returns
    -------
    features : numpy ndarray
        1)Short Run Emphasis, 2)Long Run Emphasis, 3)Gray Level
        Non-Uniformity/Gray Level Distribution, 4)Run Length
        Non-Uniformity/Run Length Distribution, 5)Run Percentage,
        6)Low Gray Level Run Emphasis, 7)High Gray Level Run Emphasis,
        8)Short Low Gray Level Emphasis, 9)Short Run High Gray Level
        Emphasis, 10)Long Run Low Gray Level Emphasis, 11)Long Run
        High Gray Level Emphasis.
    labels : list
        Labels of features.
    '''

    if mask is None:
        mask = np.ones(f.shape)

    labels = ["GLRLM_ShortRunEmphasis",
              "GLRLM_LongRunEmphasis",
              "GLRLM_GrayLevelNo-Uniformity",
              "GLRLM_RunLengthNonUniformity",
              "GLRLM_RunPercentage",
              "GLRLM_LowGrayLevelRunEmphasis",
              "GLRLM_HighGrayLevelRunEmphasis",
              "GLRLM_Short owGrayLevelEmphasis",
              "GLRLM_ShortRunHighGrayLevelEmphasis",
              "GLRLM_LongRunLowGrayLevelEmphasis",
              "GLRLM_LongRunHighGrayLevelEmphasis"]

    rlmatrix = glrlm(f, mask, Ng)

    I, J = _calculate_ij(rlmatrix)
    S = _calculate_s(rlmatrix)
    G = np.apply_over_axes(np.sum, rlmatrix, axes=1)
    R = np.apply_over_axes(np.sum, rlmatrix, axes=0)

    features = np.zeros(11, np.double)
    features[0] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, rlmatrix, (J * J)), axes=(0, 1))[
        0, 0]) / S).mean()
    features[1] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.multiply, rlmatrix, (J * J)), axes=(0, 1))[
        0, 0]) / S).mean()
    features[2] = ((np.apply_over_axes(np.sum, (G * G), axes=(0, 1))[0, 0]) / S).mean()
    features[3] = ((np.apply_over_axes(np.sum, (R * R), axes=(0, 1))[0, 0]) / S).mean()

    gray_level, run_length, _ = rlmatrix.shape
    num_voxels = gray_level * run_length
    features[4] = (S / num_voxels).mean()

    features[5] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, rlmatrix, (I * I)), axes=(0, 1))[
        0, 0]) / S).mean()
    features[6] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.multiply, rlmatrix, (I * I)), axes=(0, 1))[
        0, 0]) / S).mean()
    features[7] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, rlmatrix, (I * I * J * J)), axes=(0, 1))[
        0, 0]) / S).mean()

    temp = _apply_over_degree(np.multiply, rlmatrix, (I * I))
    features[8] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, temp, (J * J)), axes=(0, 1))[
        0, 0]) / S).mean()

    temp = _apply_over_degree(np.multiply, rlmatrix, (J * J))
    features[9] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, temp, (J * J)), axes=(0, 1))[
        0, 0]) / S).mean()
    features[10] = ((
                    np.apply_over_axes(np.sum, _apply_over_degree(np.multiply, rlmatrix, (I * I * J * J)), axes=(0, 1))[
                        0, 0]) / S).mean()

    return features, labels

# Paste the GLRLM feature extraction functions provided above here
# (glrlm_0, glrlm_90, glrlm_45, glrlm_135, glrlm, _apply_over_degree, _calculate_ij, _calculate_s, glrlm_features)

# Define paths
base_directory =  'D:\\Abdur\\Woodchip\\enviva_class\\batch_2\\data' #'D:\\Abdur\\Woodchip\\enviva_whole'#'D:/Abdur/Woodchip/batch_2_classes/data'  # Replace with your directory path containing class folders
output_csv = 'data/glrlm_features_enviva_b2.csv'  # Path to save the CSV file

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

        # Extract GLRLM features using the provided function
        features, labels = glrlm_features(image, mask)

        # Combine features into a single row and add the label (folder name)
        features = list(features)
        data.append([image_file] + features + [class_folder])  # Add class label

# Save the features to CSV
if data and labels:
    columns = ["Image"] + labels + ["Label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"GLRLM features successfully saved to {output_csv}")
else:
    print("No data to save or labels are missing.")
