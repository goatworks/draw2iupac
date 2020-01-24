import cv2
import numpy as np
from test import test_utils
import segmentation

# get b/w image and it's connected components
image = test_utils.get_test_image('CH4')

cv2.imshow('test_image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

_, labelled_image = cv2.connectedComponents(image)

# print all chunks
for label in range(1, labelled_image.max()+1):
    blob_image = segmentation.get_blob_by_label_value(image, labelled_image, label)
    cv2.imshow(f'Image #{label}', blob_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
