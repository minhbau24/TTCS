import requests
import json
import csv
from pathlib import Path
import random
import time
from datetime import datetime
import pytz  # Nếu bạn cần xử lý múi giờ
import pandas as pd
from bs4 import BeautifulSoup

product_url = "https://tiki.vn/api/v2/products/{}"

categors = []
headers = {"user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"}

def fetch_product_ids(url):
    """
        Đầu vào: url là API của một danh mục sản phẩm (Category)
        Đầu ra: một danh sách chứa ID của các sản phẩm trong danh mục và số lượnglượng
    """
    product_list = []
    i = 1
    while (True):
        # print("Crawl page: ", i)
        # print(url.format(i))
        response = requests.get(url.format(i), headers=headers)
        
        if (response.status_code != 200):
            break

        products = json.loads(response.text)["data"]

        if (len(products) == 0):
            break
        
        for product in products:
            product_id = str(product["id"])
            # print("Product ID: ", product_id)
            product_list.append(product_id)

        i += 1

    return product_list, i

def fetch_product_details(product_list=[]):
    """
        Lấy thông tin chi tiết của một sản phẩm từ Tiki API.
        Đầu vào: product_id
        Đầu ra: Thông tin sản phẩm dưới dạng dictionary
    """
    product_detail_list = []
    
    for product_id in product_list:
        response = requests.get(product_url.format(product_id), headers=headers)
        if (response.status_code == 200):
            product_detail_list.append(response.text)

    return product_detail_list

def normalize_product_data(product):
    """
        Đầu vào: một json
        Đầu ra: một dict hợp lệ
    """
    if not product:
        return None

    try:
        e = json.loads(product)
    except json.JSONDecodeError as err:
        # print(f"⚠️ Lỗi JSON: {err}, Dữ liệu: {product}")
        return None

    if not e or not e.get("id"):
        # print("⚠️ Không tìm thấy 'id' trong JSON:", e)
        return None

    # Xử lý mô tả
    soup = BeautifulSoup(e.get("description", "") or "", "html.parser")

    # Xử lý danh sách ảnh (tránh lỗi nếu API trả về None)
    images = e.get("image", [])
    if not isinstance(images, list):
        images = []

    # Xử lý stock_item (tránh lỗi nếu API trả về None)
    stock_item = e.get("stock_item") or {}

    API_parameters = e["id"]
    data_product = {
        "id": e["id"],
        "name": e["name"],
        "price": e.get("price", e.get("list_price", e.get("original_price", None))),
        "short_description": e.get("short_description", None),
        "description": soup.get_text(separator=" ", strip=True),
        "rating_average": e.get("rating_average"),
        "review_count": e.get("review_count", 0),
        "image_base_url": ", ".join(img.get("base_url", "N/A") for img in images),
        "image_large_url": ", ".join(img.get("large_url", "N/A") for img in images),
        "image_medium_url": ", ".join(img.get("medium_url", "N/A") for img in images),
        "image_small_url": ", ".join(img.get("small_url", "N/A") for img in images),
        "image_thumbnail_url": ", ".join(img.get("thumbnail_url", "N/A") for img in images),
        "stock": stock_item.get("qty", None),
        "quantity_sold": e.get("review_count", 0),
        # "created_at": ", ".join(f"{k}: {v if v is not None else 'N/A'}"
        #                         for k, v in (e.get("current_seller", {}) or {}).items())
    }
    
    return data_product, API_parameters

def save_product_id(product_id_file, product_list=[]):
    file = open(product_id_file, "w+")
    str = "\n".join(product_list)
    file.write(str)
    file.close()
    print("Save file: ", product_id_file)

def save_raw_product(product_data_file, product_detail_list=[]):
    with open(product_data_file, "w", encoding="utf-8") as file:  
        file.write("\n".join(product_detail_list)) 
    print("Save file:", product_data_file)
    
def save_product_list(product_file, product_json_list):
    fieldnames = product_json_list[0].keys()
    # Ghi dữ liệu dưới dạng utf-8-sig (lý do chọn utf-8-sig là để đọc file excel không bị lỗi front tiếng việt)
    with open(product_file, "w", newline="", encoding="utf-8-sig") as file:      
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # Ghi tiêu đề cột
        writer.writerows(product_json_list)  # Ghi dữ liệu

    print("Save file:", product_file)

def load_raw_product(product_data_file):
    file = open(product_data_file, "r")
    return file.readlines()

def map_json_to_customers(json_data):
    customers = {}
    
    for review in json_data.get("data", []):  # Lặp qua từng đánh giá
        created_by = review.get("created_by") or {}
        contribute_info = (created_by.get("contribute_info") or {}).get("summary") or {}

        customer_id = created_by.get("id")
        if not customer_id:
            continue  # Bỏ qua nếu không có ID khách hàng

        if customer_id not in customers:
            purchased_at = created_by.get("purchased_at")
            customers[customer_id] = {
                "id": customer_id,
                "full_name": created_by.get("full_name", ""),
                "avatar_url": created_by.get("avatar_url", ""),
                "joined_time": contribute_info.get("joined_time", ""),
                "total_review": contribute_info.get("total_review", 0),
                "total_thank": contribute_info.get("total_thank", 0),
                "purchased": created_by.get("purchased", False),
                "purchased_at": datetime.fromtimestamp(purchased_at, pytz.UTC) if isinstance(purchased_at, (int, float)) else None,
                "group_id": created_by.get("group_id")
            }

    return list(customers.values())

def map_json_to_review(json_data):
    reviews = []
    for review in json_data.get("data", []):
        created_by = review.get("created_by") or {}
        timeline = review.get("timeline") or {}
        
        created_at = timeline.get("review_created_date")
        try:
            created_at = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC) if created_at else None
        except ValueError:
            created_at = None
        
        review_data = {
            "id": review.get("id"),
            "customer_id": created_by.get("id"),
            "product_id": review.get("product_id"),
            "rating": review.get("rating", 0),
            "title": review.get("title", ""),
            "content": review.get("content", ""),
            "status": review.get("status", "unknown"),
            "thank_count": review.get("thank_count", 0),
            "comment_count": review.get("comment_count", 0),
            "images": review.get("images", []),
            "seller_id": review.get("seller", {}).get("id"),
            "seller_name": review.get("seller", {}).get("name", ""),
            "delivery_rating": review.get("delivery_rating", [])
        }
        reviews.append(review_data)

    return reviews

def map_json_to_buy_history(json_data):
    buy_histories = []
    for review in json_data.get("data", []):
        timeline = review.get("timeline") or {}
        customer_info = review.get("created_by") or {}

        purchased_at = customer_info.get("purchased_at")
        delivery_date = timeline.get("delivery_date")

        try:
            purchased_at = datetime.fromtimestamp(purchased_at, pytz.UTC) if isinstance(purchased_at, (int, float)) else None
        except ValueError:
            purchased_at = None

        try:
            delivery_date = datetime.strptime(delivery_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC) if delivery_date else None
        except ValueError:
            delivery_date = None

        buy_history_data = {
            "customer_id": customer_info.get("id"),
            "product_id": review.get("product_id"),
            "order_id": None,  # Cần lấy từ hệ thống đơn hàng nếu có
            "seller_id": review.get("seller", {}).get("id"),
            "seller_name": review.get("seller", {}).get("name", ""),
            "quantity": random.randint(1, 4),  # Giả sử mỗi lần mua chỉ có 1 sản phẩm, nếu có dữ liệu từ OrderItem thì thay đổi
            "price_at_purchase": None,  # Cần lấy từ OrderItem nếu có
            "total_price": None,  # Cần lấy từ OrderItem nếu có
            "purchased_at": purchased_at,
            "delivery_date": delivery_date,
            "review_id": review.get("id")
        }
        buy_histories.append(buy_history_data)

    return buy_histories

def fetch_user_reviews_data(folder_parent_path, product_id_list):
    """
    Thu thập dữ liệu đánh giá, thông tin khách hàng và lịch sử mua hàng từ API của Tiki.
    
    Đầu vào:
        - folder_parent_path (str): Đường dẫn thư mục lưu dữ liệu đầu ra.
        - product_id_list (list): Danh sách ID sản phẩm cần lấy dữ liệu đánh giá.

    Đầu ra:
        - Lưu 3 file CSV vào thư mục `./data/{folder_parent_path}/`:
            1. `reviews.csv`: Chứa thông tin đánh giá sản phẩm.
            2. `customers.csv`: Chứa thông tin khách hàng đã đánh giá.
            3. `buy_historys.csv`: Chứa lịch sử mua hàng của khách.

    Mô tả hoạt động:
        - Gửi request đến API lấy danh sách đánh giá của từng sản phẩm.
        - Trích xuất dữ liệu thành 3 danh sách: đánh giá, khách hàng, lịch sử mua.
        - Lưu dữ liệu vào các file CSV.
    """
    API_cmt = "https://tiki.vn/api/v2/reviews?product_id={}&page={}&limit=10"
    reviews = []
    customers = []
    buy_historys = []

    for id in product_id_list:
        i = 1
        while (True):
            response = requests.get(API_cmt.format(id, i), headers=headers)
            if (response.status_code != 200):
                break

            e = json.loads(response.text)
            if e["reviews_count"] == 0:
                break
            if i*10 > e["reviews_count"]:
                break
            
            reviews += map_json_to_review(e)
            customers += map_json_to_customers(e)
            buy_historys += map_json_to_buy_history(e)
            i += 1
    df_reviews = pd.DataFrame(reviews)
    df_customers = pd.DataFrame(customers)
    df_buy_historys = pd.DataFrame(buy_historys)
    print("number reviews: ", len(reviews))
    df_reviews.to_csv("./data/" + folder_parent_path + "/reviews.csv", index=False, encoding="utf-8-sig")
    
    df_customers.to_csv("./data/" + folder_parent_path + "/customers.csv", index=False, encoding="utf-8-sig")
    df_buy_historys.to_csv("./data/" + folder_parent_path + "/buy_historys.csv", index=False, encoding="utf-8-sig")

def fetch_and_save_product_data(folder, product_list_id):
    """
    Lấy dữ liệu chi tiết của sản phẩm, xử lý và lưu vào các file CSV/TXT.

    Đầu vào:
        - folder (str): Thư mục lưu dữ liệu.
        - product_list_id (list): Danh sách ID sản phẩm cần thu thập dữ liệu.

    Đầu ra:
        - Lưu dữ liệu vào các file trong thư mục `./data/{folder}/`:
            1. `product-id.txt`: Chứa danh sách ID sản phẩm (backup).
            2. `product.txt`: Chứa dữ liệu chi tiết sản phẩm ở dạng thô (backup).
            3. `product.csv`: Chứa dữ liệu sản phẩm đã được xử lý.

    Mô tả hoạt động:
        - Gọi API lấy thông tin chi tiết sản phẩm.
        - Chuẩn hóa dữ liệu (normalize).
        - Lọc bỏ những sản phẩm có lỗi khi lấy dữ liệu.
        - Lưu dữ liệu vào các file để sử dụng sau.
    """

    product_id_file = "./data/" + folder + "/product-id.txt"
    product_data_file = "./data/" + folder + "/product.txt"
    product_file = "./data/" + folder + "/product.csv"

    # crawl detail for each product id
    product_list = fetch_product_details(product_list_id)

    print("Number crawl product: ", len(product_list))        

    product_json_list = []
    error_query = []
    
    for i in range(len(product_list)):
        temp = normalize_product_data(product_list[i])
        if temp is not None:
            product_json, API_parameters = temp
            product_json_list.append(product_json)
        else:
            error_query.append(i)

    print(f"There are {len(error_query)} query errors")
    product_list_id = [val for i, val in enumerate(product_list_id) if i not in error_query]
    product_list = [val for i, val in enumerate(product_list) if i not in error_query]
    
    # save product id for backup
    save_product_id(product_id_file, product_list_id)

    # save product detail for backup
    save_raw_product(product_data_file, product_list)

    save_product_list(product_file, product_json_list)

def fetch_category_data(parent, data, check=False):
    """
    Thu thập dữ liệu sản phẩm từ một danh mục trên Tiki và lưu vào thư mục tương ứng.

    Đầu vào:
        - parent (str): Đường dẫn thư mục cha.
        - data (dict): Thông tin danh mục (bao gồm id, url_key, parent_id).
        - check (bool): Nếu True, sử dụng thư mục cha trực tiếp; nếu False, tạo thư mục con theo `url_key`.

    Đầu ra:
        - Lưu danh sách sản phẩm vào các file trong thư mục `./data/{folder_parent_path}/`.
        - Gọi các hàm `fetch_and_save_product_data` và `fetch_user_reviews_data` để lấy và lưu dữ liệu sản phẩm, đánh giá.

    Mô tả hoạt động:
        - Thêm danh mục vào danh sách `categories`.
        - Xây dựng đường dẫn thư mục và tạo thư mục nếu chưa có.
        - Gọi API lấy danh sách sản phẩm thuộc danh mục.
        - Gọi các hàm khác để lấy thông tin chi tiết và đánh giá của sản phẩm.
    """

    categors.append({
        "id": data.get("id", ""),
        "name": data.get("url_key", ""),
        "parent": data.get("parent_id", "")
    })
    folder_parent_path = ""
    if check:
        folder_parent_path = parent
    else:
        folder_parent_path = parent + "/" + data["url_key"].strip()
    folder_path = Path("data/{}".format(folder_parent_path))
    folder_path.mkdir(parents=True, exist_ok=True)
    
    id_childen_list = []
    i = 0
    print("url_key: {}, category: {}".format(data["url_key"], data["id"]))
    while True:
        response = requests.get("https://tiki.vn/api/personalish/v1/blocks/listings?limit=10&page={}&urlKey={}&category={}".format(i, data["url_key"], data["id"]), headers=headers)
        if response.status_code != 200:
            if i == 0:
                id_childen_list.append([])
            break
        products = json.loads(response.text)["data"]
        
        if len(products) == 0:
            if i == 0:
                id_childen_list.append([])
            break
        
        for product in products:
            product_id = str(product["id"])
            id_childen_list.append(product_id)

        i += 1
    fetch_and_save_product_data(folder_parent_path, id_childen_list)
    fetch_user_reviews_data(folder_parent_path, id_childen_list)
    print()
    time.sleep(random.uniform(2, 5))
        
def fetch_and_traverse_categories(name, category_id, parent=None):
    """
    Lấy dữ liệu danh mục từ API Tiki và duyệt đệ quy để lấy tất cả danh mục con.

    Đầu vào:
        - name (str): Tên danh mục hiện tại.
        - category_id (int): ID của danh mục hiện tại.
        - parent (str, optional): Tên danh mục cha (nếu có).

    Đầu ra:
        - Thêm danh mục vào danh sách `categories`.
        - Nếu danh mục không có danh mục con, gọi `fetch_category_data` để lấy dữ liệu sản phẩm.
        - Nếu có danh mục con, duyệt đệ quy để lấy tất cả danh mục con.

    Mô tả hoạt động:
        - Gửi request đến API để lấy danh sách danh mục con.
        - Nếu API trả về lỗi hoặc dữ liệu không hợp lệ, in thông báo lỗi.
        - Nếu danh mục không có danh mục con, gọi `fetch_category_data`.
        - Nếu có danh mục con, tiếp tục gọi đệ quy để lấy dữ liệu tất cả danh mục con.
    """

    url = f"https://tiki.vn/api/v2/categories?include=children&parent_id={category_id}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Request error: {response.status_code} - {response.text}")
        return None

    try:
        data = response.json()  # Tránh dùng json.loads(response.text)
    except json.JSONDecodeError:
        print("Lỗi: API không trả về JSON hợp lệ")
        return None

    # Kiểm tra xem "data" có tồn tại không
    if "data" not in data or not isinstance(data["data"], list):
        print(f"Lỗi: Không tìm thấy danh sách dữ liệu hợp lệ trong API ({url})")
        return None

    e = data["data"]

    if not e:  # Kiểm tra danh sách rỗng
        fetch_category_data(name, {"url_key": name, "id": category_id}, True)

    categors.append({
        "id": category_id,
        "name": name.split("/")[-1],
        "parent": parent
    })

    for data in e:
        if "children" in data:
            fetch_and_traverse_categories(name + "/" + data["url_key"].strip(), data["id"], category_id)
        else:
            fetch_category_data(name, data)


def fetch_and_save_categories(category_name, category_id, output_folder="./data"):
    """
    Thu thập danh mục sản phẩm và lưu vào file CSV.

    Đầu vào:
        - category_name (str): Tên danh mục cần lấy dữ liệu.
        - category_id (str): ID của danh mục.
        - output_folder (str, optional): Thư mục lưu file CSV (mặc định là './data').

    Đầu ra:
        - Lưu danh sách danh mục vào file CSV trong thư mục tương ứng.

    Mô tả hoạt động:
        - Gọi `fetch_and_traverse_categories` để lấy toàn bộ danh mục con.
        - Lưu danh mục vào file CSV.
    """
    categors = []
    fetch_and_traverse_categories(category_name, category_id)
    
    df = pd.DataFrame(categors)
    category_csv_path = f"{output_folder}/{category_name}/category.csv"
    df.to_csv(category_csv_path, index=False)

    print(f"Đã lưu file CSV tại: {category_csv_path}")
