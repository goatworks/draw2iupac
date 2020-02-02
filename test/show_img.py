import cv2
import numpy
import sys

cv2.imshow('lalala', numpy.load(sys.argv[1]))
cv2.waitKey()
cv2.destroyAllWindows()
