import cv2
gray_image = cv2.imread('../pics/CH4.jpg', cv2.IMREAD_GRAYSCALE)
resize_img = cv2.resize(gray_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

kernel_size = 13
blurred_image = cv2.GaussianBlur(resize_img, (kernel_size, kernel_size), 0)

_retval, bw_image = cv2.threshold(blurred_image, 120, 255, cv2.THRESH_BINARY)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(bw_image, low_threshold, high_threshold)

cv2.imshow('resized image', resize_img)
cv2.imshow('blurred', blurred_image)
cv2.imshow('bw', bw_image)
cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

