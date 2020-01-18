import cv2
import numpy as np
from test import test_utils

image = test_utils.get_test_image('CH4')

cv2.imshow('test_image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# TODO: fare cv2.connectedComponents()
