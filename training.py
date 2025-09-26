# Train an SVM model using the dataset class
import torch
from torch.utils.data import DataLoader
from sklearn import svm
from dataset import MDDataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

# Parameters
file_path = './dataset/mental_disorders_dataset.csv'  # Path to your CSV file
target_column = 'Expert Diagnose'  # Name of the target column in your CSV
batch_size = 32
num_epochs = 100
learning_rate = 0.001

# Create dataset
dataset = MDDataset(file_path, target_column, normalize=True, pca_components=5, select_k_best=10)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize SVM model
model = svm.SVC(kernel='linear', C=1.0, random_state=42)

# Cargar modelo si existe
try:
    model = joblib.load('./models/svm_model.pkl')
    print("Modelo cargado exitosamente.")
except FileNotFoundError:
    print("No se encontró el modelo. Se creará uno nuevo.")

# Training loop
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.numpy()
        labels = labels.numpy()
        model.fit(inputs, labels)
      
    print(f'Epoch [{epoch+1}/{num_epochs}] completed.')
    # save model after each epoch
    joblib.dump(model, './models/svm_model.pkl')


# Evaluate the model
X_test, y_test = dataset.get_test_data()
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(classification_report(y_test, y_pred))

print("Labels:", dataset.get_labels(), len(dataset.get_labels()))
