import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh từ file
image = cv2.imread('D:\XLHA\BT8\CV2.jpg', cv2.IMREAD_GRAYSCALE)

# Áp dụng Gaussian Blur để làm mờ hình ảnh
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Sử dụng Sobel để phát hiện biên
sobel_x = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)

# Sử dụng Prewitt (tạo toán tử Prewitt thủ công)
prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
prewitt_x = cv2.filter2D(gaussian_blur, -1, prewitt_kernel_x)
prewitt_y = cv2.filter2D(gaussian_blur, -1, prewitt_kernel_y)

# Đảm bảo rằng prewitt_x và prewitt_y có cùng kiểu dữ liệu và kích thước
prewitt_x = np.float32(prewitt_x)
prewitt_y = np.float32(prewitt_y)

prewitt = cv2.magnitude(prewitt_x, prewitt_y)

# Sử dụng Robert Cross (tạo toán tử Robert thủ công)
roberts_kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

roberts_x = cv2.filter2D(gaussian_blur, -1, roberts_kernel_x)
roberts_y = cv2.filter2D(gaussian_blur, -1, roberts_kernel_y)

# Đảm bảo rằng roberts_x và roberts_y có cùng kiểu dữ liệu và kích thước
roberts_x = np.float32(roberts_x)
roberts_y = np.float32(roberts_y)

# Kiểm tra kích thước của roberts_x và roberts_y để đảm bảo chúng giống nhau
if roberts_x.shape != roberts_y.shape:
    print("Kích thước của roberts_x và roberts_y không khớp!")
else:
    roberts = cv2.magnitude(roberts_x, roberts_y)

# Sử dụng Canny để phát hiện biên
canny_edges = cv2.Canny(gaussian_blur, 100, 200)

# Hiển thị kết quả
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 3, 2)
plt.imshow(sobel, cmap='gray')
plt.title('Sobel Edge Detection')

plt.subplot(2, 3, 3)
plt.imshow(prewitt, cmap='gray')
plt.title('Prewitt Edge Detection')

plt.subplot(2, 3, 4)
plt.imshow(roberts, cmap='gray')
plt.title('Roberts Edge Detection')

plt.subplot(2, 3, 5)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge Detection')

plt.tight_layout()
plt.show()
