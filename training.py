# Train an SVM model using the dataset class
import torch
from torch.utils.data import DataLoader
from sklearn import svm
from dataset import MDDataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Parameters
file_path = './dataset/mental_disorders_dataset.csv'  # Path to your CSV file
target_column = 'Expert Diagnose'  # Name of the target column in your CSV
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Create dataset
dataset = MDDataset(file_path, target_column, normalize=True, pca_components=5, select_k_best=10)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize SVM model
model = svm.SVC(kernel='linear', C=1.0, random_state=42)

# Training loop
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.numpy()
        labels = labels.numpy()
        model.fit(inputs, labels)
    print(f'Epoch [{epoch+1}/{num_epochs}] completed.')

# Evaluate the model
X_test = dataset.X_test.numpy()
y_test = dataset.y_test.numpy()
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(classification_report(y_test, y_pred))

