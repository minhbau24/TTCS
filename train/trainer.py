import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import pandas as pd

def evaluate_model(model, data_loader, device, prefix="Val", threshold=0.5, k=10):
    model.eval()
    all_preds, all_labels, all_customer_ids = [], [], []
    
    eval_bar = tqdm(data_loader, desc=f"{prefix} Evaluation", leave=False)
    with torch.no_grad():
        for batch in eval_bar:
            customer = batch['customer'].to(device)
            product = batch['product'].to(device)
            seller = batch['seller'].to(device)
            features = batch['features'].to(device)
            cat_levels = batch['cat_levels'].to(device)
            label = batch['label'].to(device)

            output = model(customer, product, seller, features, cat_levels).squeeze()
            all_preds.extend(output.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_customer_ids.extend(customer.cpu().numpy())
    # === AUC ===
    auc = roc_auc_score(all_labels, all_preds)
    print(f"{prefix} AUC: {auc:.4f}")
    # === HR@k và NDCG@k ===
    df = pd.DataFrame({
        'customer_id': all_customer_ids,
        'prediction': all_preds,
        'label': all_labels
    })

    df_sorted = df.sort_values(['customer_id', 'prediction'], ascending=[True, False])
    df_sorted['rank'] = df_sorted.groupby('customer_id').cumcount() + 1

    hr_df = df_sorted[df_sorted['label'] == 1].copy()
    hr_df['in_top_k'] = hr_df['rank'] <= k
    hr_at_k = hr_df['in_top_k'].mean()
    print(f"{prefix} HR@{k}: {hr_at_k:.4f}")

    ndcg_df = hr_df.copy()
    ndcg_df['dcg'] = 1 / np.log2(ndcg_df['rank'] + 1)
    ndcg_df['idcg'] = 1 / np.log2(2)  # IDCG = 1 nếu đúng item ở top-1
    ndcg_at_k = (ndcg_df['dcg'] / ndcg_df['idcg']).mean()
    print(f"{prefix} NDCG@{k}: {ndcg_at_k:.4f}")

    return auc, hr_at_k, ndcg_at_k

# Hàm huấn luyện
def train_model(model, train_loader, val_loader, device, epochs=10, patience=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    best_auc = 0
    best_hr_at_k = 0
    best_ndcg_at_k = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in train_bar:
            customer = batch['customer'].to(device)
            product = batch['product'].to(device)
            seller = batch['seller'].to(device)
            features = batch['features'].to(device)
            cat_levels = batch['cat_levels'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            output = model(customer, product, seller, features, cat_levels)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

        val_auc, val_hr_at_k, val_ndcg_at_k = evaluate_model(model, val_loader, device, prefix="Val")

        if val_auc > best_auc or val_hr_at_k > best_hr_at_k or val_ndcg_at_k > best_ndcg_at_k:
            best_auc = max(val_auc, best_auc)
            best_hr_at_k = max(val_hr_at_k, best_hr_at_k)
            best_ndcg_at_k = max(val_ndcg_at_k, best_ndcg_at_k)
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break