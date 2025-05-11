from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import json
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from models.neumf import NeuMF  
from data.dataset import create_mappings_and_scaler

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SelectedProduct(BaseModel):
    id: int
    name: str
    price: float
    image_url: str

# Read CSV and create list
def load_products(df, category_name="Sach", category_lv=0):
    products_list = []
    df_filtered = df.copy()
    if category_lv != 0:
        df_filtered = df[df[f"cat_level_{category_lv}"] == category_name]
    for _, row in df_filtered.iterrows():
        products_list.append({
            "id": row["id"],
            "name": row["name"],
            "price": row["price"],
            "image_url": row["image_base_url"].split(",")[0].strip() if isinstance(row["image_base_url"], str) else ""
        })
    return products_list

# Read JSON and create dictionary
def load_categories():
    with open("./data/categories.json", "r") as file:
        categories = json.load(file)
    return categories

# Load dữ liệu tại startup
try:
    df = pd.read_csv("./data/product.csv")
    full_interactions = pd.concat([
        pd.read_csv("./data/train_interactions.csv"),
        pd.read_csv("./data/val_interactions.csv"),
        pd.read_csv("./data/test_interactions.csv")
    ])
    categories = load_categories()
except FileNotFoundError as e:
    raise Exception(f"Missing data file: {e}")

# Tạo ánh xạ
cust_map, prod_map, feat_dict, seller_dict, cat_maps, cat_indices = create_mappings_and_scaler(full_interactions, df)
num_customers = len(cust_map)
num_products = len(prod_map)
num_sellers = len(seller_dict)
cat_vocab_sizes = [len(cat_maps[col]) for col in sorted(cat_maps.keys())]
num_features = 1  # price

model = NeuMF(
    num_customers=num_customers,
    num_products=num_products,
    num_sellers=num_sellers,
    cat_vocab_sizes=cat_vocab_sizes,
    num_features=num_features,
    embed_dim=32,
    mlp_layers=[128, 64, 32]
)
try:
    model.load_state_dict(torch.load("./checkpoints/best_model.pt", map_location="cpu"))
except FileNotFoundError:
    raise Exception("Missing best_model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Hàm tính Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

# Hàm tìm người dùng tương tự
def find_similar_users(new_user_products, k=5):
    new_user_set = set(new_user_products)

    # Bước 1: Lọc những dòng có tương tác tích cực với các sản phẩm new_user_products
    filtered = full_interactions[
        (full_interactions['product_id'].isin(new_user_products)) &
        (full_interactions['is_positive'] == 1)
    ]

    # Bước 2: Tính số sản phẩm trùng (intersection) theo từng customer
    overlap_count = filtered.groupby('customer_id')['product_id'].nunique()

    # Bước 3: Tính tổng số sản phẩm người dùng đó đã tương tác tích cực (union base)
    total_per_user = full_interactions[
        full_interactions['is_positive'] == 1
    ].groupby('customer_id')['product_id'].nunique()

    # Bước 4: Tính Jaccard similarity
    sim_df = overlap_count.to_frame('intersection').join(
        total_per_user.to_frame('user_total'), how='inner'
    )
    sim_df['union'] = sim_df['user_total'] + len(new_user_set) - sim_df['intersection']
    sim_df['similarity'] = sim_df['intersection'] / sim_df['union']

    # Bước 5: Sắp xếp và lấy top K người dùng tương tự nhất
    sim_df = sim_df[sim_df['similarity'] > 0]
    top_similar_users = sim_df.sort_values(by='similarity', ascending=False).head(k)

    res = top_similar_users.reset_index()[['customer_id', 'similarity']]
    res = res['customer_id'].tolist()
    return res

# Hàm lấy top 1000 sản phẩm theo danh mục
def get_top_products_by_category(new_user_products, k=1000):
    categories = set()
    for pid in new_user_products:
        product_row = df[df['id'] == pid]
        if not product_row.empty:
            for col in ['cat_level_1', 'cat_level_2', 'cat_level_3', 'cat_level_4', 'cat_level_5']:
                if col in product_row and pd.notna(product_row[col].iloc[0]) and product_row[col].iloc[0] != '<PAD>':
                    categories.add(product_row[col].iloc[0])
    
    candidate_products = df[df[['cat_level_1', 'cat_level_2', 'cat_level_3', 'cat_level_4', 'cat_level_5']].isin(categories).any(axis=1)]
    candidate_product_ids = candidate_products['id'].tolist()
    
    product_counts = full_interactions[(full_interactions['product_id'].isin(candidate_product_ids)) & (full_interactions['is_positive'] == 1)]['product_id'].value_counts()
    top_product_ids = product_counts.head(k).index.tolist()
    
    if len(top_product_ids) < k:
        popular_products = full_interactions[full_interactions['is_positive'] == 1]['product_id'].value_counts().index.tolist()
        top_product_ids.extend([pid for pid in popular_products if pid not in top_product_ids][:k - len(top_product_ids)])
    
    return top_product_ids

# Hàm tạo gợi ý
def get_recommendations(new_user_products, k=10):
    print(f"[INFO] Getting recommendations for input: {new_user_products}")
    
    if not new_user_products:
        print("[INFO] No products provided — returning popular products")
        popular_products = full_interactions[full_interactions['is_positive'] == 1]['product_id'].value_counts().head(k).index.tolist()
        return load_products(df[df['id'].isin(popular_products)])
    
    valid_product_ids = [pid for pid in new_user_products if pid in prod_map]
    print(f"[INFO] Valid product IDs: {valid_product_ids}")
    if not valid_product_ids:
        raise Exception("No valid product IDs provided")
    
    similar_users = find_similar_users(valid_product_ids, k=5)
    print(f"[INFO] Found similar users: {similar_users}")
    if not similar_users:
        print("[INFO] No similar users found — returning popular products")
        popular_products = full_interactions[full_interactions['is_positive'] == 1]['product_id'].value_counts().head(k).index.tolist()
        return load_products(df[df['id'].isin(popular_products)])
    
    similar_user_indices = [cust_map[cid] for cid in similar_users]
    print(f"[INFO] Similar user indices: {similar_user_indices}")
    
    print("[INFO] Computing average embeddings...")
    customer_gmf_embeds = model.customer_embed_gmf(torch.tensor(similar_user_indices, device=device))
    customer_mlp_embeds = model.customer_embed_mlp(torch.tensor(similar_user_indices, device=device))
    avg_gmf_embed = torch.mean(customer_gmf_embeds, dim=0).unsqueeze(0)
    avg_mlp_embed = torch.mean(customer_mlp_embeds, dim=0).unsqueeze(0)
    
    print("[INFO] Selecting top products by category...")
    top_product_ids = get_top_products_by_category(valid_product_ids, k=1000)
    top_product_indices = [prod_map[pid] for pid in top_product_ids if pid in prod_map]
    print(f"[INFO] Top product indices: {top_product_indices[:10]} ... (showing first 10)")
    
    print("[INFO] Preparing model inputs...")
    customer_idx = np.array([0] * len(top_product_indices))
    product_idx = np.array(top_product_indices)
    seller_idx = np.array([seller_dict.get(p, 0) for p in top_product_indices])
    features = np.array([feat_dict.get(p, [0, 0]) for p in top_product_indices])
    cat_levels = []
    for col in sorted(cat_indices.keys()):
        cat_levels.append(np.array([cat_indices[col].get(p, 0) for p in top_product_indices]))
    cat_levels = np.stack(cat_levels, axis=1) if cat_levels else np.zeros((len(top_product_indices), 1), dtype=np.int64)

    print("[INFO] Running model prediction...")
    customer_tensor = torch.tensor(customer_idx, dtype=torch.long, device=device)
    product_tensor = torch.tensor(product_idx, dtype=torch.long, device=device)
    seller_tensor = torch.tensor(seller_idx, dtype=torch.long, device=device)
    features_tensor = torch.tensor(features, dtype=torch.float, device=device)
    cat_levels_tensor = torch.tensor(cat_levels, dtype=torch.long, device=device)

    avg_gmf_embed = avg_gmf_embed.expand(len(top_product_indices), -1)
    avg_mlp_embed = avg_mlp_embed.expand(len(top_product_indices), -1)
    print(f"[INFO] Average GMF embedding shape: {avg_gmf_embed.shape}")
    print(f"[INFO] Average MLP embedding shape: {avg_mlp_embed.shape}")
    print("Start prediction...")

    with torch.no_grad():
        predictions = model(
            customer_tensor,
            product_tensor,
            seller_tensor,
            features_tensor,
            cat_levels_tensor,
            custom_gmf_embed=avg_gmf_embed,
            custom_mlp_embed=avg_mlp_embed
        ).cpu().numpy()
    print(f"[INFO] Predictions shape: {predictions.shape}")
    print(f"[INFO] Predictions: {predictions[:10]} ... (showing first 10)")
    top_k_indices = np.argsort(predictions)[-k:][::-1]
    top_k_product_ids = [top_product_ids[i] for i in top_k_indices]
    print(f"[INFO] Top-k recommended product IDs: {top_k_product_ids}")
    
    return load_products(df[df['id'].isin(top_k_product_ids)])


# Endpoint để lấy sản phẩm phân trang
@app.get("/products")
async def get_products(limit: int = 10, page: int = 0, category: str = "sach"):
    if category not in categories:
        return {"error": "Category not found"}
    products = load_products(df, category_name=category, category_lv=categories[category]["lv"])
    total = len(products)
    if limit <= 0 or page < 0:
        return {"error": "Invalid limit or page number"}
    if page * limit >= total:
        return {"error": "Page number out of range"}
    offset = page * limit
    paginated_products = products[offset:offset + limit]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": paginated_products
    }

# Endpoint để lấy danh mục
@app.get("/categories")
async def get_categories(category: str = "sach"):
    if category not in categories:
        return {"error": "Category not found"}

    res = categories[category].copy()
    expanded_children = []
    for child in res["children"]:
        child_name = child if isinstance(child, str) else child.get("name")
        if child_name in categories:
            child_obj = categories[child_name].copy()
            child_obj["name"] = child_name
            expanded_children.append(child_obj)

    res["children"] = expanded_children
    return res["children"]

# Endpoint để xử lý tương tác dương và trả gợi ý
@app.post("/submit-selection")
async def submit_selection(products: List[SelectedProduct]):
    try:
        product_ids = [product.id for product in products]
        recommended_products = get_recommendations(product_ids, k=10)
        return {"recommendations": recommended_products}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error processing recommendations: {str(e)}"})