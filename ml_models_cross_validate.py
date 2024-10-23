# General purpose imports
import os
import random
import warnings
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

# Scikit-learn utilities and metrics
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Scikit-learn classifiers
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)


def load_image(data_dir):
    print("Loading all images")
    # Get the list of subdirectories (i.e., class names)
    class_names = os.listdir(data_dir)
    # Initialize the arrays to store the images and class labels
    X = []
    y = []
    # Loop over each class and load the images
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))  # Resize the image
            X.append(img)
            y.append(class_name)

    return shuffle(np.array(X), np.array(y))


# Assuming X and y are numpy arrays or lists
def shuffle_data(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]


def evaluate_ml(X_train, y_train, X_test, y_test, data_name, f_name, fold):
    print("Fold: ", fold)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # List of classifiers
    classifiers = [
        ('Random Forest', RandomForestClassifier()),
        ('Gradient Boosting', GradientBoostingClassifier()),
        ('SVM', SVC(kernel='rbf', probability=True)),
        ('Logistic Regression', LogisticRegression()),
        ('K-Nearest Neighbors', KNeighborsClassifier()),
        ('Naive Bayes', GaussianNB()),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Neural Network', MLPClassifier()),
        ('AdaBoost', AdaBoostClassifier()),
        ('Bagging', BaggingClassifier()),
        ('Extra Trees', ExtraTreesClassifier()),
        ('Voting Classifier', VotingClassifier(estimators=[
            ('Logistic Regression', LogisticRegression()),
            ('Neural Network', MLPClassifier()),
            ('SVM', SVC(kernel='rbf', probability=True)),
        ], voting='soft'))
    ]

    results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

    # Iterate over classifiers
    for model_name, model in classifiers:
        # Fit the model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Generate the classification report as a string
        report = classification_report(y_test, y_pred, target_names=["dry", "medium", "wet"])

        # Save the classification report to a text file
        with open(f'results/{f_name}/cls_report/{model_name}_{data_name}_{f_name}_Fold_{fold}.txt', 'w') as f:
            f.write(f"Classification Report on {data_name} {f_name} features at Fold: {fold} using {model_name}:\n")
            f.write(report)
        # Evaluate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        conf_matrix = confusion_matrix(y_test, y_pred)
        # Normalize the confusion matrix to percentages
        conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        # Plot the confusion matrix
        plt.figure(figsize=(1.8, 1.3))
        sns.heatmap(conf_matrix_percentage, annot=True, fmt='.1f', cmap='Blues', xticklabels=['D', 'M', 'W'],
                    yticklabels=['D', 'M', 'W'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        # plt.title('Confusion Matrix')
        plt.savefig(f'plots/conf_matrix/{data_name}/{model_name}_{f_name}_Fold_{fold}.jpg', dpi=600,
                    bbox_inches='tight')
        plt.close()

        # Append results to DataFrame
        results_df = results_df.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }, ignore_index=True)

    ## Save results to a CSV file
    results_file_path = f'results/{f_name}/{data_name}_{f_name}_Fold_{fold}.csv'  # Replace with your desired file path
    results_df.to_csv(results_file_path, index=False)

    # Display results
    print(results_df)


data_name = 'enviva_b2'
f_name = 'lbp'
file_path = f'D:\\Abdur\\Woodchip\\AdaptMoist\\data\\{f_name}_features_{data_name}.csv'  # Replace with your actual file path

data = pd.read_csv(file_path, header=0)

# Assume the last column is the label and the rest are features
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# Define the K-fold cross-validation splitter
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
fold = 1
for train_idx, val_idx in kfold.split(X, y):
    print("Started Fold: ", fold)
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    evaluate_ml(X_train, y_train, X_val, y_val, data_name, f_name, fold)
    fold = fold + 1
