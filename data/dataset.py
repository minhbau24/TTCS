import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch

def create_mappings_and_scaler(full_interactions, product_df):
    print("Columns in product_df:", product_df.columns.tolist())
    customer_ids = full_interactions['customer_id'].unique()
    product_ids = product_df['id'].unique()
    seller_ids = product_df['seller_id'].unique()
    
    customer_id_map = {id: idx for idx, id in enumerate(customer_ids)}
    product_id_map = {id: idx for idx, id in enumerate(product_ids)}
    seller_id_map = {id: idx for idx, id in enumerate(seller_ids)}
    
    product_df = product_df.copy()
    product_df['product_idx'] = product_df['id'].map(product_id_map)
    product_df['seller_idx'] = product_df['seller_id'].map(seller_id_map)
    
    scaler = StandardScaler()
    numeric_cols = ['price']
    if all(col in product_df.columns for col in numeric_cols):
        product_df[numeric_cols] = product_df[numeric_cols].fillna(product_df[numeric_cols].mean())
        product_features = scaler.fit_transform(product_df[numeric_cols])
    else:
        print("Warning: Missing numeric columns. Using zeros for features.")
        product_features = np.zeros((len(product_df), len(numeric_cols)))
    
    product_features_dict = {row['product_idx']: feat for row, feat in zip(product_df.to_dict('records'), product_features)}
    seller_idx_dict = product_df.set_index('product_idx')['seller_idx'].to_dict()
    
    cat_columns = ['cat_level_1', 'cat_level_2', 'cat_level_3', 'cat_level_4', 'cat_level_5']
    cat_maps = {}
    cat_indices = {}
    
    valid_cat_columns = [col for col in cat_columns if col in product_df.columns]
    if valid_cat_columns:
        for col in valid_cat_columns:
            product_df[col] = product_df[col].replace('<PAD>', 'unknown').fillna('unknown')
            unique_vals = product_df[col].unique()
            cat_maps[col] = {val: idx for idx, val in enumerate(unique_vals)}
            product_df[f'{col}_idx'] = product_df[col].map(cat_maps[col])
            cat_indices[col] = product_df.set_index('product_idx')[f'{col}_idx'].to_dict()
    else:
        print("No valid category columns found. Proceeding without category features.")
    
    return customer_id_map, product_id_map, product_features_dict, seller_idx_dict, cat_maps, cat_indices

def prepare_inputs(df, customer_id_map, product_id_map, product_features_dict, seller_idx_dict, cat_indices):
    df = df.copy()
    df['customer_idx'] = df['customer_id'].map(customer_id_map)
    df['product_idx'] = df['product_id'].map(product_id_map)
    
    required_cols = ['customer_idx', 'product_idx', 'is_positive']
    df = df.dropna(subset=[col for col in required_cols if col in df.columns])
    df['customer_idx'] = df['customer_idx'].astype(int)
    df['product_idx'] = df['product_idx'].astype(int)
    
    customer_indices = df['customer_idx'].values
    product_indices = df['product_idx'].values
    labels = df['is_positive'].values
    features = np.array([product_features_dict.get(p, [0, 0]) for p in product_indices])
    sellers = np.array([seller_idx_dict.get(p, 0) for p in product_indices])
    
    cat_levels = []
    if cat_indices:
        for col in sorted(cat_indices.keys()):
            cat_levels.append(np.array([cat_indices[col].get(p, 0) for p in product_indices]))
        cat_levels = np.stack(cat_levels, axis=1)
    else:
        cat_levels = np.zeros((len(product_indices), 1), dtype=np.int64)
    
    return {
        'customer_idx': customer_indices,
        'product_idx': product_indices,
        'seller_idx': sellers,
        'features': features,
        'cat_levels': cat_levels,
        'labels': labels
    }

class BookRecDataset(Dataset):
    def __init__(self, customer_idx, product_idx, seller_idx, features, cat_levels, labels):
        self.customer_idx = torch.tensor(customer_idx, dtype=torch.long)
        self.product_idx = torch.tensor(product_idx, dtype=torch.long)
        self.seller_idx = torch.tensor(seller_idx, dtype=torch.long)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.cat_levels = torch.tensor(cat_levels, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

        self.has_categories = self.cat_levels.shape[1] > 1 or (
            self.cat_levels.shape[1] == 1 and torch.any(self.cat_levels)
        )
        self.default_cat = torch.tensor([0], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'customer': self.customer_idx[idx],
            'product': self.product_idx[idx],
            'seller': self.seller_idx[idx],
            'features': self.features[idx],
            'label': self.labels[idx]
        }
        item['cat_levels'] = self.cat_levels[idx] if self.has_categories else self.default_cat
        return item