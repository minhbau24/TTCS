import os
import pandas as pd
import numpy as np

def category_path_to_dataframe(path, max_level=5, pad_token='<PAD>'):
    # Tách chuỗi path thành các cấp
    parts = str(path).strip().split('\\')[1:-1]
    padded_parts = parts + [pad_token] * (max_level - len(parts))
    padded_parts = padded_parts[:max_level]  # Cắt nếu thừa

    # Tạo DataFrame 1 dòng với 5 cột
    df = pd.DataFrame([padded_parts], columns=[f'cat_level_{i+1}' for i in range(max_level)])
    return df

def read_data(file_name, cols=[], category=False):
    data = []
    for root, dirs, files in os.walk("nha-sach-tiki"):
        for file in files:
            if file == file_name:
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path, usecols=cols)
                    if category:
                        category_df = category_path_to_dataframe(file_path)
                        category_column_repeat = pd.concat([category_df] * len(df), ignore_index=True)
                        df = pd.concat([df, category_column_repeat], axis=1)
                        
                    # Chuyển đổi các cột thành kiểu dữ liệu phù hợp
                    data.append(df)
                except Exception as e:
                    print(f"{file_path}: {e}")
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

def leave_one_out_split(interactions):
    """Chia dữ liệu theo chuẩn recommender: 1 test, 1 val, còn lại train cho mỗi user"""
    interactions = interactions.sort_values(['customer_id'])
    
    train_data = []
    val_data = []
    test_data = []

    for customer_id, group in interactions.groupby('customer_id'):
        if len(group) < 3:
            continue  # bỏ qua user có ít hơn 3 tương tác

        test_data.append(group.iloc[-1])
        val_data.append(group.iloc[-2])
        train_data.append(group.iloc[:-2])

    train_df = pd.concat(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    return train_df.copy(), val_df.copy(), test_df.copy()

def generate_negative_samples(pos_df, all_products, num_negatives=4, seed=42):
    """Sinh mẫu âm với mỗi user"""
    np.random.seed(seed)
    negative_samples = []

    user_pos_items = pos_df.groupby('customer_id')['product_id'].apply(set).to_dict()

    for customer_id, pos_items in user_pos_items.items():
        neg_candidates = list(all_products - pos_items)
        num_neg_samples = num_negatives * len(pos_items)
        
        if len(neg_candidates) == 0:
            continue  # user đã tương tác hết items
        
        sampled_neg = np.random.choice(
            neg_candidates,
            size=min(num_neg_samples, len(neg_candidates)),
            replace=False
        )

        for product_id in sampled_neg:
            negative_samples.append({
                'customer_id': customer_id,
                'product_id': product_id,
                'is_positive': 0
            })

    return pd.DataFrame(negative_samples)

def generate_ranking_eval_set(pos_df, all_products, num_negatives=99, seed=42):
    """Sinh tập đánh giá ranking: 1 positive + 99 negatives cho mỗi user"""
    np.random.seed(seed)
    eval_samples = []

    for idx, row in pos_df.iterrows():
        user = row['customer_id']
        pos_item = row['product_id']
        interacted = set(pos_df[pos_df['customer_id'] == user]['product_id'])

        negatives = list(all_products - interacted - {pos_item})
        if len(negatives) < num_negatives:
            continue  # không đủ mẫu âm

        neg_items = np.random.choice(negatives, num_negatives, replace=False)

        eval_samples.append({'customer_id': user, 'product_id': pos_item, 'is_positive': 1})
        for item in neg_items:
            eval_samples.append({'customer_id': user, 'product_id': item, 'is_positive': 0})

    return pd.DataFrame(eval_samples)

# Đọc dữ liệu từ reviews.csv (có cột rating) và buy_historys.csv
reviews = read_data("data/reviews.csv", ["customer_id", "product_id", "rating"])
buy_history = read_data("data/buy_historys.csv", ["customer_id", "product_id", "seller_id"])

# Chuyển đổi kiểu dữ liệu customer_id
reviews['customer_id'] = reviews['customer_id'].astype('Int64')
buy_history['customer_id'] = buy_history['customer_id'].astype('Int64')
buy_history['seller_id'] = buy_history['seller_id'].astype('Int64')

interactions = reviews.copy()

# Đánh dấu mẫu dương dựa trên rating (rating >= 3 là mẫu dương)
# Với các dòng từ buy_history (rating là NA), coi mặc định là mẫu dương
interactions['is_positive'] = interactions['rating'].apply(lambda x: 1 if pd.isna(x) or x >= 3 else 0)

# Lọc các customer_id có ít nhất 5 tương tác dương
positive_interactions = interactions[interactions['is_positive'] == 1]
customer_count = positive_interactions['customer_id'].value_counts().reset_index()
customer_count.columns = ['customer_id', 'count']
customer_count = customer_count[customer_count['count'] >= 5]

# Lọc interactions chỉ giữ lại các khách hàng có ít nhất 5 tương tác dương
interactions = interactions[interactions['customer_id'].isin(customer_count['customer_id'])].reset_index(drop=True)

# Xóa cột rating
interactions = interactions.drop(columns=['rating'])

print(f"Số lượng khách hàng còn lại: {interactions['customer_id'].nunique()}")

# Lấy tập product_id đang còn trong interactions
valid_product_ids = set(interactions['product_id'])

# Tạo dictionary từ buy_history, nhưng chỉ lấy các product_id đang có trong interactions
buy_history_seller = buy_history[['product_id', 'seller_id']].drop_duplicates(subset=['product_id'], keep='first')
buy_history_seller = buy_history_seller[buy_history_seller['product_id'].isin(valid_product_ids)]

# Ánh xạ product_id -> seller_id
product_seller_dict = dict(zip(buy_history_seller['product_id'], buy_history_seller['seller_id']))

missing_products = set(interactions['product_id']) - set(product_seller_dict.keys())
print(f"Số lượng sản phẩm thiếu seller: {len(missing_products)}")

# Đọc dữ liệu sản phẩm (không có cột seller_id)
product = read_data("product.csv", ["id", "price", "rating_average", "name", "image_base_url"], category=True)

product['seller_id'] = np.nan  # Khởi tạo cột seller_id với giá trị NaN

# Gán cột seller_id vào product
product['seller_id'] = product['id'].map(product_seller_dict)
product['seller_id'] = product['seller_id'].astype('Int64')

# Loại bỏ các sản phẩm không có seller_id (nếu có)
product = product[product['seller_id'].notna()]
print(f"Số lượng sản phẩm còn lại: {product['id'].nunique()}")

all_products = set(interactions['product_id'].unique())

train_df, val_df_pos, test_df_pos = leave_one_out_split(interactions)

train_df['is_positive'] = 1

train_neg = generate_negative_samples(train_df, all_products, num_negatives=4)
train_full = pd.concat([train_df, train_neg], ignore_index=True)
train_full = train_full.sample(frac=1, random_state=42).reset_index(drop=True)

val_full = generate_ranking_eval_set(val_df_pos, all_products, num_negatives=99)
test_full = generate_ranking_eval_set(test_df_pos, all_products, num_negatives=99)

# Lưu file dữ liệu đã xử lý
train_full.to_csv("train_interactions.csv", index=False)
val_full.to_csv("val_interactions.csv", index=False)
test_full.to_csv("test_interactions.csv", index=False)
product.to_csv("product.csv", index=False)

print("Dữ liệu đã được lưu vào các file train_interactions.csv, val_interactions.csv, test_interactions.csv và product.csv.")
print("Số lượng mẫu trong train:", len(train_full))
print("Số lượng mẫu trong val:", len(val_full))
print("Số lượng mẫu trong test:", len(test_full))
print("Số lượng sản phẩm trong product:", len(product))