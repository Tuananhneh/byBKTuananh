import numpy as np
import cv2
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, adjusted_rand_score
from sklearn.datasets import load_iris
from fcmeans import FCM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Tính F1-score và RAND index
def evaluate_clustering(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    rand_index = adjusted_rand_score(y_true, y_pred)
    return f1, rand_index

# Bước 1: Phân cụm với KMeans trên bộ dữ liệu IRIS
def evaluate_iris_clustering(n_clusters):
    # Tải bộ dữ liệu IRIS
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Phân cụm với KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_pred = kmeans.fit_predict(X_scaled)
    
    # Tính F1-score và RAND index
    f1, rand_index = evaluate_clustering(y_true, y_pred)
    print(f"KMeans Clustering with {n_clusters} clusters (IRIS Dataset)")
    print(f"F1-Score: {f1:.4f}")
    print(f"RAND Index: {rand_index:.4f}")
    print("-" * 60)

# Bước 2: Phân cụm với FCM trên bộ dữ liệu IRIS
def evaluate_fcm_iris_clustering(n_clusters):
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Phân cụm với FCM
    fcm = FCM(n_clusters=n_clusters, random_state=42)
    fcm.fit(X_scaled)
    y_pred = np.argmax(fcm.u, axis=1)  # Lấy nhãn dự đoán từ độ mạnh của mỗi cụm
    
    # Tính F1-score và RAND index
    f1, rand_index = evaluate_clustering(y_true, y_pred)
    print(f"FCM Clustering with {n_clusters} clusters (IRIS Dataset)")
    print(f"F1-Score: {f1:.4f}")
    print(f"RAND Index: {rand_index:.4f}")
    print("-" * 60)

# Bước 3: Phân cụm ảnh giao thông bằng KMeans
def evaluate_image_clustering(image_path, n_clusters):
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc ảnh, vui lòng kiểm tra đường dẫn ảnh.")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Đổi màu sắc cho đúng (BGR -> RGB)
    
    # Reshape ảnh thành dạng 2D (h * w, 3)
    img_reshaped = img_rgb.reshape((-1, 3))
    
    # Phân cụm với KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(img_reshaped)
    
    # Dự đoán nhãn cho các pixel
    kmeans_img = kmeans.predict(img_reshaped)
    
    # Tạo lại ảnh phân cụm
    kmeans_image = kmeans_img.reshape(img.shape[0], img.shape[1])
    
    # Hiển thị kết quả phân cụm
    plt.imshow(kmeans_image, cmap='viridis')
    plt.title(f"KMeans Clustering Result - {n_clusters} clusters")
    plt.show()

# Bước 4: Phân cụm ảnh giao thông bằng FCM
def evaluate_fcm_image_clustering(image_path, n_clusters):
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc ảnh, vui lòng kiểm tra đường dẫn ảnh.")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Đổi màu sắc cho đúng (BGR -> RGB)
    
    # Reshape ảnh thành dạng 2D (h * w, 3)
    img_reshaped = img_rgb.reshape((-1, 3))
    
    # Phân cụm với FCM
    fcm = FCM(n_clusters=n_clusters, random_state=42)
    fcm.fit(img_reshaped)
    fcm_img = np.argmax(fcm.u, axis=1)
    
    # Tạo lại ảnh phân cụm
    fcm_image = fcm_img.reshape(img.shape[0], img.shape[1])
    
    # Hiển thị kết quả phân cụm
    plt.imshow(fcm_image, cmap='viridis')
    plt.title(f"FCM Clustering Result - {n_clusters} clusters")
    plt.show()

# Main function
if __name__ == "__main__":
    # Bước 1: Phân cụm với KMeans và FCM trên bộ dữ liệu IRIS
    for n in [2, 3, 4]:
        evaluate_iris_clustering(n)
        evaluate_fcm_iris_clustering(n)
    
    # Bước 2: Phân cụm ảnh giao thông (giả sử bạn đã có ảnh giao thông)
    image_path = "D:\XLHA\BT7\CV2.jpg"  # Đảm bảo rằng bạn đã thay đường dẫn ảnh đúng
    for n in [2, 3, 4]:
        evaluate_image_clustering(image_path, n)
        evaluate_fcm_image_clustering(image_path, n)
