import os
import cv2
import pandas as pd
import numpy as np
import mahotas

# Define paths
base_directory = 'D:\\Abdur\\Woodchip\\enviva_class\\batch_2\\data' #'D:\\Abdur\\Woodchip\\enviva_whole'  # 'D:/Abdur/Woodchip/batch_2_classes/data'
output_csv = 'data/haralick_features_enviva_b2.csv'  # Path to save the CSV file

# Function to extract Haralick features using mahotas
def glcm_features(f, ignore_zeros=False):
    labels = [
        "GLCM_ASM", "GLCM_Contrast", "GLCM_Correlation",
        "GLCM_SumOfSquaresVariance", "GLCM_InverseDifferenceMoment",
        "GLCM_SumAverage", "GLCM_SumVariance", "GLCM_SumEntropy",
        "GLCM_Entropy", "GLCM_DifferenceVariance",
        "GLCM_DifferenceEntropy", "GLCM_Information1",
        "GLCM_Information2", "GLCM_MaximalCorrelationCoefficient"
    ]
    labels_mean = [label + "_Mean" for label in labels]
    labels_range = [label + "_Range" for label in labels]

    f = f.astype(np.uint8)
    features = mahotas.features.haralick(
        f, ignore_zeros=ignore_zeros, compute_14th_feature=True, return_mean_ptp=True
    )
    features_mean = features[0:14]
    features_range = features[14:]

    return features_mean, features_range, labels_mean, labels_range

# Initialize a list to store the feature data
data = []
labels_mean = None
labels_range = None

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

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(img, (640, 480))
        if image is None:
            print(f"Error reading image {image_path}")
            continue

        # Extract Haralick features
        features_mean, features_range, labels_mean, labels_range = glcm_features(image)

        # Combine mean and range features into a single row and add the label (folder name)
        features = list(features_mean) + list(features_range)
        data.append([image_file] + features + [class_folder])  # Add class label

# Save the features to CSV
if data and labels_mean and labels_range:
    columns = ["Image"] + labels_mean + labels_range + ["Label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Haralick features successfully saved to {output_csv}")
else:
    print("No data to save or labels are missing.")
