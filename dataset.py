# Load dataset from CSV file and create a class in pytorch
import joblib
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

class MDDataset(Dataset):
    def __init__(self, file_path: str, target_column: str, test_size: float = 0.2, random_state: int = 42, 
                 normalize: bool = False, pca_components: int = None):
        """
        Args:
            file_path (string): Path to the csv file.
            target_column (string): Name of the target column.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
            normalize (bool): Whether to normalize the features.
            pca_components (int or None): Number of PCA components to keep. If None, PCA is not applied.
        """
        self.normalize = normalize
        self.pca_components = pca_components

        # Load data
        self.data = pd.read_csv(file_path)

        # Get predict values
        self.labels = self.data[target_column].unique().tolist()

        # One hot encode categorical features if any expect target column
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols.remove(target_column)
        self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True)
        
        # Remove na values
        self.data = self.data.dropna()

        # Shuffle data
        self.data = shuffle(self.data, random_state=random_state).reset_index(drop=True)
        
        # Split features and target
        self.X = self.data.drop(columns=[target_column]).values
        self.y = self.data[target_column].values
        self.y = LabelEncoder().fit_transform(self.y)
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)
        
        # Normalize features if required
        if normalize:
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)

        # Apply PCA if required
        if pca_components is not None:
            self.pca = PCA(n_components=pca_components)
            self.X_train = self.pca.fit_transform(self.X_train)
            self.X_test = self.pca.transform(self.X_test)
            
        
        # Convert to torch tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.long)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test, dtype=torch.long)

    def __len__(self):
        return len(self.X_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
    
    def get_test_data(self):
        return self.X_test, self.y_test
    
    def get_labels(self):
        return self.labels
    
    def process_point(self, point) -> torch.Tensor:
        """
        Process a single data point (1D numpy array) in the same way as the training data.
        Args:
            point: single data point.
        Returns:
            torch.Tensor: Processed data point as a tensor.
        """
        point = pd.DataFrame([point])
        categorical_cols = point.select_dtypes(include=['object', 'category']).columns.tolist()
        point = pd.get_dummies(point, columns=categorical_cols, drop_first=True)
        point = point.reindex(columns=self.data.drop(columns=[self.data.columns[-1]]).columns, fill_value=0)
        
        if self.normalize:
            point = self.scaler.transform(point)

        if self.pca_components is not None:
            point = self.pca.transform(point)

        return torch.tensor(point, dtype=torch.float32)