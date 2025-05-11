# 🧠 TTCS - Hệ Khuyến Nghị Sản Phẩm Tiki sử dụng Neural Collaborative Filtering

**Thương mại điện tử tại Việt Nam đang phát triển nhanh chóng**, kéo theo nhu cầu cá nhân hóa trải nghiệm mua sắm để đáp ứng sở thích đa dạng của người dùng. Tiki, một nền tảng thương mại điện tử lớn, sở hữu dữ liệu phong phú về hành vi người dùng như lịch sử mua sắm, đánh giá sản phẩm và lượt xem.

Dự án này hướng đến phát triển một **hệ khuyến nghị sản phẩm** sử dụng **Neural Collaborative Filtering (NCF)** – một phương pháp Deep Learning tiên tiến, nhằm đề xuất các sản phẩm phù hợp với từng cá nhân dựa trên lịch sử tương tác.

---

## 📂 Mục lục

- Giới thiệu
- Tính năng
- Cài đặt
- Cách sử dụng

---

## 📝 Giới thiệu

Dự án **TTCS** được thiết kế cho:

- Các nhà nghiên cứu và lập trình viên quan tâm đến hệ thống gợi ý và ứng dụng Deep Learning trong thương mại điện tử.
- Các doanh nghiệp thương mại điện tử muốn cải thiện trải nghiệm người dùng thông qua cá nhân hóa.

**Công nghệ sử dụng:**

- Python
- PyTorch
- FastAPI
- Node.js
- Pandas, NumPy, Scikit-learn

---

## ✨ Tính năng

- ✅ Thu thập dữ liệu từ Tiki (lịch sử tương tác, thông tin sản phẩm)
- ✅ Xây dựng và huấn luyện mô hình NCF để gợi ý sản phẩm
- ✅ Triển khai demo đơn giản với backend FastAPI & frontend Node.js
- ✅ Tối ưu mô hình với các đặc trưng bổ sung
- 🚧 Mở rộng demo để lọc và hiển thị sản phẩm

---

## ⚙️ Cài đặt

### 1. Clone repository

Chạy các lệnh sau để tải mã nguồn:

```bash
git clone https://github.com/yourusername/TTCS.git
cd TTCS
```

### 2. Cài đặt thư viện Python

- Cài các thư viện trong `requirements.txt` bằng lệnh:

```bash
pip install -r requirements.txt
```

### 3. Cài đặt Node.js (cho frontend demo)

- Tải Node.js từ trang chính thức: [https://nodejs.org](https://nodejs.org)
- Cài thêm `http-server`:

```bash
npm install -g http-server
```

### 4. Thiết bị

- GPU (ưu tiên): tăng tốc độ huấn luyện.
- CPU: có thể dùng nhưng chậm hơn.
- Cài driver GPU và CUDA nếu dùng GPU.

### 5. Tải dữ liệu

- Tải dữ liệu từ Google Drive: https://drive.google.com/drive/folders/1dBywhYWkEB-LVmN9XbPXDSiLU6Z4Xs_P
- Giải nén và đặt vào thư mục `data/` theo cấu trúc:

---

```text data/
├── train/
├── train_interactions.csv
├── val_interactions.csv
├── test_interactions.csv
├── product.csv
└── categories.json
```

---

## 🛠️ Cách sử dụng

### 1. Thu thập dữ liệu (tùy chọn)

- Chạy file: `python Crawl_data_tiki.py`  
  Dữ liệu sẽ được lưu vào thư mục `data/`.

---

### 2. Huấn luyện mô hình

- Chạy lệnh:

```bash
python main.py
```

Quá trình này sẽ:

- Tải dữ liệu từ `data/`
- Xử lý và chuyển sang GPU nếu có
- Huấn luyện mô hình trong 10 epoch
- Lưu mô hình vào `checkpoints/best_model.pt`

**Tham số tùy chỉnh trong `main.py`:**

- `batch_size = 1024`
- `epochs = 10`
- `patience = 3`

Sau khi huấn luyện, hệ thống sẽ đánh giá mô hình với AUC, HR@10, NDCG@10 và in kết quả ra console.

---

### 3. Chạy demo

#### Khởi động backend (FastAPI)

- Đảm bảo file `checkpoints/best_model.pt` đã tồn tại
- Chạy lệnh:

```bash
uvicorn demo.app:app --reload
```

- Backend sẽ chạy tại: `http://127.0.0.1:8000`

#### Khởi động frontend (Node.js)

- Di chuyển vào thư mục `demo/frontend`
- Chạy:

```bash
http-server
```

- Mở trình duyệt: `http://127.0.0.1:8080`

#### Tương tác demo

- Truy cập `index.html` trong trình duyệt
- Chọn sản phẩm mà bạn quan tâm
- Kết quả sản phẩm được gợi ý sẽ hiển thị trong `result.html`

---

### Demo

- Truy cập: `http://127.0.0.1:8080`
- Chọn những sản phẩm bạn quan tâm
- Nhận danh sách 10 sản phẩm gợi ý

---
