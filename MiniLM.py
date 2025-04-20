from sentence_transformers import SentenceTransformer
import pandas as pd

# Đọc dữ liệu từ file CSV dữ liệu đã crawl
df = pd.read_csv('data/qua-luu-niem/guong-mini/product.csv') # Thay thế bằng file khác cần đọc

# Load mô hình
model = SentenceTransformer('all-MiniLM-L6-v2')  # Ưu điểm: model xử lý nhanh và nhẹ

# Chỉ lấy cột mô tả 'description'
if 'description' in df.columns:
    descriptions = df['description'].fillna('')  # Xử lý giá trị NaN nếu có

    # Lặp qua từng mô tả và chuyển thành vector
    embeddings = []
    for text in descriptions:
        embedding = model.encode(text)
        embeddings.append(embedding)

    print("Số lg vector:", len(embeddings))
    print("Vector mẫu:", embeddings[0][:5])  # In 5 giá trị của vector đầu tiên để lấy mẫu quan sát
    
else:
    print("Cột 'description' không tồn tại trong DataFrame này.")