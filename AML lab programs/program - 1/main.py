import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Load the Iris dataset
iris_data = load_iris()

# Split features and target
X = iris_data.data
y = iris_data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''In summary, fit_transform is used on the training data to compute the mean and standard deviation and standardize the features. Then, transform is used on the testing data to apply the same standardization using the parameters learned from the training data, maintaining consistency in the scaling process between the two datasets. This is important to ensure that the model is evaluated on data that has been processed in the same way as the data it was trained on.'''

# Create and train a K-Nearest Neighbors model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)

# Predict using the model
knn_predictions = knn_model.predict(X_test_scaled)

# Print correct and wrong predictions for K-Nearest Neighbors
print("K-Nearest Neighbors Predictions:")
for i in range(len(y_test)):
    true_label = iris_data.target_names[y_test[i]]
    predicted_label = iris_data.target_names[knn_predictions[i]]
    
    if y_test[i] == knn_predictions[i]:
        print(f"Correct prediction for instance {i + 1}: True label = {true_label}, Predicted label = {predicted_label}")
    else:
        print(f"Wrong prediction for instance {i + 1}: True label = {true_label}, Predicted label = {predicted_label}")

# Calculate accuracy
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("K-Nearest Neighbors Accuracy:", knn_accuracy)

# Print classification report
print("K-Nearest Neighbors Classification Report:\n", classification_report(y_test, knn_predictions, target_names=iris_data.target_names))
