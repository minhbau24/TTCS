import pandas as pd
import torch
from torch.utils.data import DataLoader
from data.dataset import create_mappings_and_scaler, prepare_inputs, BookRecDataset
from models.neumf import NeuMF
from train.trainer import train_model, evaluate_model

if __name__ == "__main__":
    train_df = pd.read_csv("./data/train_interactions.csv")
    val_df = pd.read_csv("./data/val_interactions.csv")
    test_df = pd.read_csv("./data/test_interactions.csv")
    product = pd.read_csv("./data/product.csv", usecols=['id', 'price', 'cat_level_1', 'cat_level_2', 'cat_level_3', 'cat_level_4', 'cat_level_5', 'seller_id'])

    full_interactions = pd.concat([train_df, val_df, test_df])

    cust_map, prod_map, feat_dict, seller_dict, cat_maps, cat_indices = create_mappings_and_scaler(full_interactions, product)

    num_customers = len(cust_map)
    num_products = len(prod_map)
    num_sellers = len(seller_dict)
    cat_vocab_sizes = [len(cat_maps[col]) for col in sorted(cat_maps.keys())] if cat_maps else []
    num_features = 1  # price

    train_inputs = prepare_inputs(train_df, cust_map, prod_map, feat_dict, seller_dict, cat_indices)
    val_inputs = prepare_inputs(val_df, cust_map, prod_map, feat_dict, seller_dict, cat_indices)
    test_inputs = prepare_inputs(test_df, cust_map, prod_map, feat_dict, seller_dict, cat_indices)

    train_dataset = BookRecDataset(**train_inputs)
    val_dataset = BookRecDataset(**val_inputs)
    test_dataset = BookRecDataset(**test_inputs)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuMF(
        num_customers=num_customers,
        num_products=num_products,
        num_sellers=num_sellers,
        cat_vocab_sizes=cat_vocab_sizes,
        num_features=num_features,
        embed_dim=32,
        mlp_layers=[64, 32, 16]
    ).to(device)

    train_model(model, train_loader, val_loader, device, epochs=10, patience=3)

    evaluate_model(model, test_loader, device, prefix="Test")