import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import time
import os

def evaluate_model(model, data_loader, device, prefix="Val", threshold=0.5, k=10):
    model.eval()
    all_preds, all_labels, all_customer_ids = [], [], []
    
    start_time = time.perf_counter()
    eval_bar = tqdm(data_loader, desc=f"{prefix} Evaluation", leave=False)
    with torch.no_grad():
        for batch in eval_bar:
            customer = batch['customer']
            product = batch['product']
            seller = batch['seller']
            features = batch['features']
            cat_levels = batch['cat_levels']
            label = batch['label']

            output = model(customer, product, seller, features, cat_levels)
            all_preds.extend(output.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_customer_ids.extend(customer.cpu().numpy())

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    auc = roc_auc_score(all_labels, all_preds)
    print(f"{prefix} AUC: {auc:.4f}")
    print(f"{prefix} Evaluation Time: {elapsed_time:.2f} seconds\n")
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_customer_ids = np.array(all_customer_ids)
    if prefix == "Test":
        unique_customers = np.unique(all_customer_ids)
        hits = []
        ndcgs = []
        customer_bar = tqdm(unique_customers, desc=f"{prefix} Ranking Metrics", leave=False)
        for cust_id in customer_bar:
            cust_mask = all_customer_ids == cust_id
            cust_preds = all_preds[cust_mask]
            cust_labels = all_labels[cust_mask]
    
            sorted_indices = np.argsort(cust_preds)[::-1]
            sorted_labels = cust_labels[sorted_indices]
    
            hit = 1 if sorted_labels[:k].sum() > 0 else 0
            hits.append(hit)
    
            dcg = 0
            for i, label in enumerate(sorted_labels[:k], 1):
                if label == 1:
                    dcg += 1 / np.log2(i + 1)
                    break
            idcg = 1 / np.log2(2)
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)
    
        hr_at_k = np.mean(hits)
        ndcg_at_k = np.mean(ndcgs)
        print(f"{prefix} HR@{k}: {hr_at_k:.4f}")
        print(f"{prefix} NDCG@{k}: {ndcg_at_k:.4f}")
    
        return auc, hr_at_k, ndcg_at_k
    return auc

def train_model(model, train_loader, val_loader, device, epochs=10, patience=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    best_auc = 0
    patience_counter = 0

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in train_bar:
            customer = batch['customer']
            product = batch['product']
            seller = batch['seller']
            features = batch['features']
            cat_levels = batch['cat_levels']
            label = batch['label']
            optimizer.zero_grad()
            output = model(customer, product, seller, features, cat_levels)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

        val_auc = evaluate_model(model, val_loader, device, prefix="Val")
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break