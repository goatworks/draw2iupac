import cv2
import numpy as np
from test import test_utils
import segmentation

image = test_utils.get_test_image('CH4')
_, labelled_image = cv2.connectedComponents(image)

# for label in range(1, labelled_image.max()+1):
#     blob_image = segmentation.get_blob_by_label_value(image, labelled_image, label)
#     cv2.imshow(f'Image #{label}', blob_image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
blob_image = segmentation.get_blob_by_label_value(image, labelled_image, 2)

row_n, col_n = blob_image.shape
if row_n > col_n:
    col_to_be_added = row_n - col_n
    padded_image = np.pad(blob_image, ((0, 0), (col_to_be_added, 0)), 'constant')
else:
    row_to_be_added = col_n - row_n
    padded_image = np.pad(blob_image, ((row_to_be_added, 0), (0, 0)), 'constant')

assert padded_image.shape[0] == padded_image.shape[1]  # my matrix is now squared.

# resize to pxp (64x64)
final_dimension = 64
if padded_image.shape[0] > final_dimension:
    resized_image = cv2.resize(padded_image, (final_dimension, final_dimension), interpolation=cv2.INTER_AREA)
else:
    resized_image = cv2.resize(padded_image, (final_dimension, final_dimension), interpolation=cv2.INTER_CUBIC)

cv2.imshow('old', blob_image)
cv2.imshow('64squared', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()