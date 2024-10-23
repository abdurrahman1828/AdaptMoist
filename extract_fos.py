import os
import cv2
import pandas as pd
import numpy as np



def fos(f, mask):
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
        1)Mean, 2)Variance, 3)Median (50-Percentile), 4)Mode,
        5)Skewness, 6)Kurtosis, 7)Energy, 8)Entropy,
        9)Minimal Gray Level, 10)Maximal Gray Level,
        11)Coefficient of Variation, 12,13,14,15)10,25,75,90-
        Percentile, 16)Histogram width
    labels : list
        Labels of features.
    '''
    if mask is None:
        mask = np.ones(f.shape)

    # 1) Labels
    labels = ["FOS_Mean", "FOS_Variance", "FOS_Median", "FOS_Mode", "FOS_Skewness",
              "FOS_Kurtosis", "FOS_Energy", "FOS_Entropy", "FOS_MinimalGrayLevel",
              "FOS_MaximalGrayLevel", "FOS_CoefficientOfVariation",
              "FOS_10Percentile", "FOS_25Percentile", "FOS_75Percentile",
              "FOS_90Percentile", "FOS_HistogramWidth"]

    # 2) Parameters
    f = f.astype(np.uint8)
    mask = mask.astype(np.uint8)
    level_min = 0
    level_max = 255
    Ng = (level_max - level_min) + 1
    bins = Ng

    # 3) Calculate Histogram H inside ROI
    f_ravel = f.ravel()
    mask_ravel = mask.ravel()
    roi = f_ravel[mask_ravel.astype(bool)]
    H = np.histogram(roi, bins=bins, range=[level_min, level_max], density=True)[0]

    # 4) Calculate Features
    features = np.zeros(16, np.double)
    i = np.arange(0, bins)
    features[0] = np.dot(i, H)
    features[1] = sum(np.multiply(((i - features[0]) ** 2), H))
    features[2] = np.percentile(roi, 50)
    features[3] = np.argmax(H)
    features[4] = sum(np.multiply(((i - features[0]) ** 3), H)) / (np.sqrt(features[1]) ** 3)
    features[5] = sum(np.multiply(((i - features[0]) ** 4), H)) / (np.sqrt(features[1]) ** 4)
    features[6] = sum(np.multiply(H, H))
    features[7] = -sum(np.multiply(H, np.log(H + 1e-16)))
    features[8] = min(roi)
    features[9] = max(roi)
    features[10] = np.sqrt(features[2]) / features[0]
    features[11] = np.percentile(roi, 10)
    features[12] = np.percentile(roi, 25)
    features[13] = np.percentile(roi, 75)
    features[14] = np.percentile(roi, 90)
    features[15] = features[14] - features[11]

    return features, labels

# Define paths
base_directory =  'D:\\Abdur\\Woodchip\\enviva_class\\batch_2\\data' #'D:\\Abdur\\Woodchip\\enviva_whole' #'D:/Abdur/Woodchip/batch_2_classes/data'  # Replace with your directory path containing class folders
output_csv = 'data/fos_features_enviva_b2.csv'  # Path to save the CSV file

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

        # Extract FOS features using the provided function
        features, labels = fos(image, mask)

        # Combine features into a single row and add the label (folder name)
        features = list(features)
        data.append([image_file] + features + [class_folder])  # Add class label

# Save the features to CSV
if data and labels:
    columns = ["Image"] + labels + ["Label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"FOS features successfully saved to {output_csv}")
else:
    print("No data to save or labels are missing.")
