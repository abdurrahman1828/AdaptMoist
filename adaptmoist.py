# General-purpose imports
import warnings
import random
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback

# Adapt library for domain adaptation models
from adapt.feature_based import DANN, MDD, CDAN

# Scikit-learn utilities and metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, adjusted_mutual_info_score
)

# Suppress warnings
warnings.filterwarnings('ignore')
# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)


s_data_name = 'enviva_b1'
t_data_name = 'enviva_b2'


train = pd.read_csv(f'D:\Abdur\Woodchip\AdaptMoist\data\haralick_features_{s_data_name}.csv')
test = pd.read_csv(f'D:\Abdur\Woodchip\AdaptMoist\data\haralick_features_{t_data_name}.csv')

X_train = train.iloc[:, 1:-1]
X_test = test.iloc[:, 1:-1]
y_train = train.iloc[:, -1]
y_test = test.iloc[:, -1]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values


# Define the encoder (feature extractor)
encoder = models.Sequential([
    layers.Input(shape=(28,)),
    layers.Dense(32, activation='relu')
])

# Define the task network (classifier)
task = models.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Define the discriminator (for domain classification)
discriminator = models.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])




class AMICallback(Callback):
    def __init__(self, validation_data, encoder, save_path):
        super(AMICallback, self).__init__()
        self.validation_data = validation_data  # Validation data to compute AMI
        self.encoder = encoder  # The encoder model to extract features
        self.save_path = save_path  # Path to save the best model
        self.best_ami = -1  # Track the best AMI

    def on_epoch_end(self, epoch, logs=None):
        # Extract validation data
        X_val = self.validation_data

        # Predict on the validation data
        y_pred = np.argmax(self.model.predict(X_val), axis=1)


        # Extract features using the encoder
        features = self.encoder.predict(X_val)

        # Generate pseudo labels using KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        pseudo_labels = kmeans.fit_predict(features)

        # Compute AMI between the model's predictions and pseudo labels
        ami = adjusted_mutual_info_score(pseudo_labels, y_pred)


        print(f"Epoch {epoch + 1} - AMI: {ami:.4f}")

        # Save the model if the AMI is better than the previous best
        if epoch < 15:
            self.best_ami = -1.0
        else:
            if ami > self.best_ami:
                self.best_ami = ami
                self.model.save_weights(self.save_path)
                print(f"Best model saved with AMI: {self.best_ami:.4f}")


# Path to save the best model
save_path = 'best_model_adaptmoist.h5'

# Initialize the AMI callback
ami_callback = AMICallback(validation_data=(X_test), encoder=encoder, save_path=save_path)

# Initialize the DANN model
model = DANN(
    encoder=encoder,
    task=task,
    discriminator=discriminator,
    Xt=X_test,
    yt=y_test,
    lambda_= 0.5,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  #0.0001
    random_state=42,
)

# Train the DANN model on the source data
model.fit(X_train, y_train, epochs=30, batch_size=2, verbose=1, callbacks = [ami_callback])

# Load the best model weights after training
model.load_weights(save_path)

# Evaluate the model on the target test set
y_pred = np.argmax(model.predict(X_test), axis=1)
y_test = np.argmax(y_test, axis=1)
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
plt.savefig(f'plots/da/{s_data_name}2{t_data_name}_haralick.jpg', dpi=600, bbox_inches='tight')
plt.show()

print(f"Target domain accuracy: {accuracy:.2f}")
print(f"Target domain precision: {precision:.2f}")
print(f"Target domain recall: {recall:.2f}")
print(f"Target domain f1-score: {f1:.2f}")



