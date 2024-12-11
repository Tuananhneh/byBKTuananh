import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from fcmeans import FCM
from sklearn.metrics import f1_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Hàm chọn file ảnh từ máy tính
def select_image():
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    file_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
    return file_path

# Hàm tính F1-score và RAND index
def evaluate_clustering(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    rand_index = adjusted_rand_score(y_true, y_pred)
    return f1, rand_index

# Phân cụm ảnh giao thông bằng KMeans
def evaluate_image_clustering(image_path, n_clusters, ax, idx):
    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc ảnh, vui lòng kiểm tra đường dẫn ảnh.")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Đổi màu sắc cho đúng (BGR -> RGB)
    img_reshaped = img_rgb.reshape((-1, 3))  # Reshape ảnh thành dạng 2D
    
    # Phân cụm với KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(img_reshaped)
    
    # Dự đoán nhãn cho các pixel
    kmeans_img = kmeans.predict(img_reshaped)
    
    # Tạo lại ảnh phân cụm
    kmeans_image = kmeans_img.reshape(img.shape[0], img.shape[1])
    
    # Hiển thị kết quả phân cụm
    ax[idx].imshow(kmeans_image, cmap='viridis')
    ax[idx].set_title(f"KMeans - {n_clusters} clusters")
    ax[idx].axis('off')  # Tắt hiển thị trục

# Phân cụm ảnh giao thông bằng FCM
def evaluate_fcm_image_clustering(image_path, n_clusters, ax, idx):
    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc ảnh, vui lòng kiểm tra đường dẫn ảnh.")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Đổi màu sắc cho đúng (BGR -> RGB)
    img_reshaped = img_rgb.reshape((-1, 3))  # Reshape ảnh thành dạng 2D
    
    # Phân cụm với FCM
    fcm = FCM(n_clusters=n_clusters, random_state=42)
    fcm.fit(img_reshaped)
    fcm_img = np.argmax(fcm.u, axis=1)  # Lấy nhãn từ FCM
    
    # Tạo lại ảnh phân cụm
    fcm_image = fcm_img.reshape(img.shape[0], img.shape[1])
    
    # Hiển thị kết quả phân cụm
    ax[idx].imshow(fcm_image, cmap='viridis')
    ax[idx].set_title(f"FCM - {n_clusters} clusters")
    ax[idx].axis('off')  # Tắt hiển thị trục

# Main function
def main():
    # Chọn ảnh từ máy tính
    image_path = select_image()
    if not image_path:
        print("Không có ảnh nào được chọn.")
        return

    # Hiển thị ảnh phân cụm
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))  # Tạo 2 hàng, 3 cột cho các ảnh
    n_clusters_list = [2, 3, 4]  # Số lượng cụm cần phân tích

    # Duyệt qua từng số cụm và thực hiện phân cụm
    for i, n_clusters in enumerate(n_clusters_list):
        evaluate_image_clustering(image_path, n_clusters, ax[0], i)
        evaluate_fcm_image_clustering(image_path, n_clusters, ax[1], i)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
