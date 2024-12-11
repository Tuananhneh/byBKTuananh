# Import các thư viện cần thiết
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, rand_score, normalized_mutual_info_score, davies_bouldin_score
from sklearn.metrics import classification_report

# Bước 1: Tải dữ liệu IRIS
iris = load_iris()
X = iris.data  # Dữ liệu đặc trưng (features)
y_true = iris.target  # Nhãn thực sự (ground truth)

# Bước 2: Hàm phân cụm và đánh giá
def cluster_and_evaluate(n_clusters):
    # Áp dụng thuật toán KMeans với số cụm tương ứng
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_pred = kmeans.fit_predict(X)

    # Ánh xạ nhãn cho các cụm
    def map_cluster_to_class(y_true, y_pred):
        label_map = {}
        for i in range(n_clusters):
            mask = (y_pred == i)
            if np.any(mask):
                label = np.argmax(np.bincount(y_true[mask]))  # Gán nhãn lớp có nhiều điểm nhất trong cụm
                label_map[i] = label
        return np.array([label_map.get(label, -1) for label in y_pred])

    # Gán nhãn cho các cụm
    y_pred_mapped = map_cluster_to_class(y_true, y_pred)

    # Tính toán các độ đo
    f1 = f1_score(y_true, y_pred_mapped, average='weighted')
    rand_index = rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    db_index = davies_bouldin_score(X, y_pred)

    # Kết quả
    print(f"Number of Clusters: {n_clusters}")
    print(f"F1-Score: {f1:.4f}")
    print(f"RAND Index: {rand_index:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"Davies-Bouldin Index: {db_index:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_mapped, target_names=iris.target_names))
    print("-" * 60)

# Bước 3: Phân cụm với 2, 3 và 7 cụm
for n in [2, 3, 7]:
    cluster_and_evaluate(n)