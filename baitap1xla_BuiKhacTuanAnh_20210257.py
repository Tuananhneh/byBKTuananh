import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh từ file
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)  # Đọc ảnh dưới dạng xám

# Hàm để hiển thị ảnh
def show_image(title, image):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# 1. Ảnh âm tính (Negative)
def negative_image(image):
    negative = 255 - image
    return negative

# 2. Tăng độ tương phản (Contrast stretching)
def increase_contrast(image):
    # Dùng histogram equalization
    contrast_stretched = cv2.equalizeHist(image)
    return contrast_stretched

# 3. Biến đổi log (Log transformation)
def log_transform(image):
    # Biến đổi log yêu cầu ảnh có giá trị trong khoảng [0, 1]
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))
    log_image = np.array(log_image, dtype=np.uint8)
    return log_image

# 4. Cân bằng histogram (Histogram equalization)
def histogram_equalization(image):
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

# Thực hiện các thao tác trên ảnh
negative_img = negative_image(image)
contrast_img = increase_contrast(image)
log_img = log_transform(image)
hist_equalized_img = histogram_equalization(image)

# Hiển thị kết quả
show_image('Original Image', image)
show_image('Negative Image', negative_img)
show_image('Increased Contrast Image', contrast_img)
show_image('Log Transformed Image', log_img)
show_image('Histogram Equalized Image', hist_equalized_img)

# Lưu ảnh sau xử lý nếu cần
cv2.imwrite('negative_image.jpg', negative_img)
cv2.imwrite('contrast_image.jpg', contrast_img)
cv2.imwrite('log_image.jpg', log_img)
cv2.imwrite('histogram_equalized_image.jpg', hist_equalized_img)
