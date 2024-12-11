import cv2
import numpy as np
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

# Hàm mở hộp thoại chọn file
def choose_image():
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính của Tkinter
    file_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    return file_path

# Hàm thực hiện dò biên Sobel và LoG
def edge_detection(image_path):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Toán tử Sobel
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Sobel theo hướng x
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Sobel theo hướng y
    sobel_combined = cv2.magnitude(sobelx, sobely)  # Kết hợp theo magnitude

    # Toán tử Laplace Gaussian (LoG)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)  # Làm mờ ảnh trước
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Hiển thị kết quả
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Ảnh gốc')
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Toán tử Sobel')
    plt.imshow(sobel_combined, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Toán tử Laplace Gaussian (LoG)')
    plt.imshow(laplacian, cmap='gray')

    plt.show()

# Thực hiện chọn ảnh và dò biên
image_path = choose_image()
if image_path:
    edge_detection(image_path)
