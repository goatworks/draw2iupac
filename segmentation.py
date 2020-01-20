import numpy as np


# extracts the part of a image (blob) matching the specified value in the corresponding labelled image.
def get_blob_by_label_value(img, labelled_img, value):
    blob = np.where(labelled_img == value)
    top_left_corner_this_blob = blob[0].min(), blob[1].min()
    bottom_right_corner_this_blob = blob[0].max(), blob[1].max()
    blob_image = img[top_left_corner_this_blob[0]:bottom_right_corner_this_blob[0],
                     top_left_corner_this_blob[1]:bottom_right_corner_this_blob[1]]

    return blob_image


