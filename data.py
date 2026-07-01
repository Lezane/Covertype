import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_covtype

class TabularDataset(Dataset):
    def __init__(self, x_num, x_cat, y, s0):
        self.x_num = x_num
        self.x_cat = x_cat
        self.y = y
        self.s0 = s0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx], self.s0[idx]

def get_dataloaders(batch_size=2048, random_seed=42):
    print("Downloading and preparing Covertype dataset...")
    covtype = fetch_covtype(as_frame=True)
    df = covtype.frame
    
    # 1. Covertype Label: 1 if tree is Class 2 (Lodgepole Pine), 0 if not
    df['target_bin'] = (df['Cover_Type'] == 2).astype(int)
    
    # Extract original categorical variables from one-hot encoded format
    wilderness_cols = [col for col in df.columns if col.startswith('Wilderness_Area')]
    soil_cols = [col for col in df.columns if col.startswith('Soil_Type')]
    
    df['Wilderness_Area'] = np.argmax(df[wilderness_cols].values, axis=1)
    df['Soil_Type'] = np.argmax(df[soil_cols].values, axis=1)
    
    # Drop the one-hot encoded columns
    df = df.drop(columns=wilderness_cols + soil_cols)
    
    numerical_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                      'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                      'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                      'Horizontal_Distance_To_Fire_Points']
    categorical_cols = ['Wilderness_Area', 'Soil_Type']
    
    # Split first to avoid data leakage
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_seed, stratify=df['target_bin'])
    
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    # 2. Minority S0 group is the subclass of covertype with the 2nd least samples in the training set
    class_counts = df_train['Cover_Type'].value_counts()
    minority_class = class_counts.index[-2]
    print(f"Minority Subclass S0 identified as Cover_Type: {minority_class} (with {class_counts[minority_class]} samples in train)")
    
    df_train['S0'] = (df_train['Cover_Type'] == minority_class).astype(int)
    df_test['S0'] = (df_test['Cover_Type'] == minority_class).astype(int)
    
    cat_cardinalities = []
    for col in categorical_cols:
        unique_vals = df_train[col].astype(str).unique()
        val_to_idx = {val: i for i, val in enumerate(unique_vals)}
        oov_idx = len(unique_vals)
        
        df_train[col] = df_train[col].astype(str).map(val_to_idx)
        df_test[col] = df_test[col].astype(str).map(val_to_idx).fillna(oov_idx).astype(int)
        cat_cardinalities.append(oov_idx + 1)
        
    scaler = StandardScaler()
    df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
    df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])
    
    X_num_train = torch.tensor(df_train[numerical_cols].values.astype(np.float32), dtype=torch.float32)
    X_num_test = torch.tensor(df_test[numerical_cols].values.astype(np.float32), dtype=torch.float32)
    
    X_cat_train = torch.tensor(df_train[categorical_cols].values.astype(np.int64), dtype=torch.long)
    X_cat_test = torch.tensor(df_test[categorical_cols].values.astype(np.int64), dtype=torch.long)
    
    y_train = torch.tensor(df_train['target_bin'].values, dtype=torch.long)
    y_test = torch.tensor(df_test['target_bin'].values, dtype=torch.long)
    
    S0_train = torch.tensor(df_train['S0'].values, dtype=torch.bool)
    S0_test = torch.tensor(df_test['S0'].values, dtype=torch.bool)
    
    print(f"Total training samples: {len(y_train)}\nTotal test samples: {len(y_test)}\n")
    
    train_dataset = TabularDataset(X_num_train, X_cat_train, y_train, S0_train)
    test_dataset = TabularDataset(X_num_test, X_cat_test, y_test, S0_test)
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader, len(numerical_cols), cat_cardinalities