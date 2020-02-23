import cv2
import numpy as np
from test import test_utils
import segmentation

# get b/w image, it's connected components and each blob image
image = test_utils.get_test_image('CH4')
_, labelled_image = cv2.connectedComponents(image)
blob_image = segmentation.get_blob_by_label_value(image, labelled_image, 2)

# resize image to a square by padding with 0s (background) around it.
row_n, col_n = blob_image.shape
if row_n > col_n:
    col_to_add = row_n - col_n
    col_left = col_to_add // 2
    col_right = col_to_add - col_left
    padded_image = np.pad(blob_image, ((0, 0), (col_left, col_right)), 'constant')
else:
    row_to_add = col_n - row_n
    row_top = row_to_add // 2
    row_bottom = row_to_add - row_top
    padded_image = np.pad(blob_image, ((row_top, row_bottom), (0, 0)), 'constant')

assert padded_image.shape[0] == padded_image.shape[1]  # my matrix is now squared.

# resize to pxp (20x20).
final_dimension = 20
if padded_image.shape[0] > final_dimension:
    resized_image = cv2.resize(padded_image, (final_dimension, final_dimension), interpolation=cv2.INTER_AREA)
else:
    resized_image = cv2.resize(padded_image, (final_dimension, final_dimension), interpolation=cv2.INTER_CUBIC)

# Add 4pad(0) each side.
# Now image will be 28x28 (20x20 + a frame of 4 "background" pixels) - as per database images.
framed_image = np.pad(resized_image, ((4, 4), (4, 4)), 'constant')

# cv2.imshow('old', blob_image)
cv2.imshow('28squared', framed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
