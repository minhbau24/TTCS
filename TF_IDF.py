import pandas as pd
import numpy as np
import re
import math

# Tự định nghĩa 1 stopword tiếng Việt
vietnamese_stopwords = ['là', 'và', 'của', 'trong', 'một', 'các', 'được', 'với', 'đã', 'thì']

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower(), flags=re.UNICODE)

def remove_stopwords(tokens):
    return [word for word in tokens if word not in vietnamese_stopwords]

# Đọc dữ liệu từ file CSV dữ liệu đã crawl
df = pd.read_csv(r'D:\jJak\TTCS\Code\TTCS\data\qua-luu-niem\guong-mini\product.csv')

# Tiền xử lý dữ liệu
df['tokens'] = df['description'].fillna('').apply(lambda x: remove_stopwords(tokenize(x)))

# Tạo từ điển dictionary
vocab = sorted(set(word for tokens in df['tokens'] for word in tokens))
word_to_index = {word: idx for idx, word in enumerate(vocab)}
vocab_size = len(vocab)

# Đếm số document chứa từng từ (IDF)
doc_count = np.zeros(vocab_size)
for tokens in df['tokens']:
    unique_words = set(tokens)
    for word in unique_words:
        if word in word_to_index:
            doc_count[word_to_index[word]] += 1

# Tính IDF
N = len(df)
idf = np.log((N + 1) / (doc_count + 1)) + 1  # Thêm smoothing tránh chia 0

# Vector TF-IDF
def tfidf_vector(tokens):
    vec = np.zeros(vocab_size, dtype=np.float32)
    word_freq = {}
    for word in tokens:
        if word in word_to_index:
            word_freq[word] = word_freq.get(word, 0) + 1
    for word, freq in word_freq.items():
        idx = word_to_index[word]
        tf = freq / len(tokens)
        vec[idx] = tf * idf[idx]
    return vec

# Gán vector TF-IDF
df['vector'] = df['tokens'].apply(tfidf_vector)

# In mẫu
print("Số lg vector::", len(df))
print("Vector mẫu:", df['vector'].iloc[0][:5])
# In 5 giá trị của vector đầu tiên để lấy mẫu quan sát