import cv2


# gives the b/w image by putting the image name without extension (withe = written text, black = background)
def get_test_image(img_name):
    file_name = f'../pics/{img_name}.jpg'
    gray_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    resize_img = cv2.resize(gray_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    kernel_size = 13
    blurred_image = cv2.GaussianBlur(resize_img, (kernel_size, kernel_size), 0)

    _, bw_image = cv2.threshold(blurred_image, 120, 255, cv2.THRESH_BINARY_INV)

    return bw_image
