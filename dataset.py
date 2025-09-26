# Load dataset from CSV file and create a class in pytorch
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

class MDDataset(Dataset):
    def __init__(self, file_path: str, target_column: str, test_size: float = 0.2, random_state: int = 42, 
                 normalize: bool = False, pca_components: int = None, select_k_best: int = None):
        """
        Args:
            file_path (string): Path to the csv file.
            target_column (string): Name of the target column.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
            normalize (bool): Whether to normalize the features.
            pca_components (int or None): Number of PCA components to keep. If None, PCA is not applied.
            select_k_best (int or None): Number of top features to select using SelectKBest. If None, feature selection is not applied.
        """
        # Load data
        self.data = pd.read_csv(file_path)
        
        # Shuffle data
        self.data = shuffle(self.data, random_state=random_state).reset_index(drop=True)
        
        # Split features and target
        self.X = self.data.drop(columns=[target_column]).values
        self.y = self.data[target_column].values
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)
        
        # Normalize features if required
        if normalize:
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
        
        # Apply PCA if required
        if pca_components is not None:
            pca = PCA(n_components=pca_components)
            self.X_train = pca.fit_transform(self.X_train)
            self.X_test = pca.transform(self.X_test)
        
        # Apply SelectKBest if required
        if select_k_best is not None:
            selector = SelectKBest(score_func=f_classif, k=select_k_best)
            self.X_train = selector.fit_transform(self.X_train, self.y_train)
            self.X_test = selector.transform(self.X_test)
        
        # Convert to torch tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.long)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test, dtype=torch.long)

    def __len__(self):
        return len(self.X_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]