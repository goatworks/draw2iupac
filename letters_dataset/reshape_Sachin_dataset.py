import csv
import os
import string
import numpy as np
import cv2
import torch

dataset_folder = '../letters_dataset'
file_path = os.path.join(dataset_folder, 'A_Z Handwritten Data.csv')
output_folder = os.path.join(dataset_folder, 'balanced_Sachin')
os.mkdir(output_folder)

# Check how many datapoints per letter the Sachin dataset contains.
letter_count = []
with open(file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)  # quotechar='|'
    count = 0
    previous_letter = None
    for row in reader:
        current_letter = row[0]
        if current_letter != previous_letter:
            letter_count.append(count)
            count = 0
        previous_letter = current_letter
        count += 1
    letter_count.append(count)
letter_count = letter_count[1:]
print(letter_count, '\n', min(letter_count))

# Divide the Sachin dataset by letter and balance it on the lowest occurrence.
# Save each letter/datapoint as a binary black and white nparray 28x28.
with open(file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    count = 0
    previous_letter = None
    for row in reader:
        current_letter = int(row[0])
        # create image folder when encountering a new letter
        this_letter_folder_name = os.path.join(output_folder, string.ascii_uppercase[current_letter])
        if current_letter != previous_letter:
            os.makedirs(this_letter_folder_name)
            count = 0
            previous_letter = current_letter
        if count >= min(letter_count):
            continue
        # create image as numpy matrix
        image_array = np.asarray(row[1:])
        image_matrix = image_array.reshape((28, 28))
        new_image = image_matrix.astype('uint8')
        # Experimented about blurring and/or thresholding to black and white. Decided not to.
        # makes image black and white (white = written text, black = background).
        # kernel_size = 3
        # blurred_image = cv2.GaussianBlur(new_image, (kernel_size, kernel_size), 0)
        # no cv2.TRESH_BINARY_INV reqired as the images are already with white written and black background.
        # _, bw_image = cv2.threshold(blurred_image, 120, 255, cv2.THRESH_BINARY)
        # save image in folder with increasing filename
        new_image_path = os.path.join(this_letter_folder_name, str(count))
        np.save(new_image_path, new_image)
        count += 1


print(':)')
