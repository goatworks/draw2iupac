import torch
import torch.nn as nn
# import torchvision
# import cv2
import numpy as np
import string
import os
import random

from neural_network import NeuralNetwork


# Get all the images (numpy arrays) in the said folder and transform them in a normalised PyTorch tensor.
# Gives a list of lists of all the tensors labelled with the number corresponding to the letter they are.
def get_dataset_from_letter_folder(folder_path, label):  # -> List[List]
    list_of_labelled_images = []
    list_of_img_in_folder = os.listdir(folder_path)
    list_of_img_in_folder.sort()  # To allow repeatable experiments.
    for file in list_of_img_in_folder:
        path = os.path.join(folder_path, file)
        image = np.load(path)
        image = image.astype(np.float32) / 255
        image = torch.from_numpy(image)
        labelled_image = [image, label]
        list_of_labelled_images.append(labelled_image)
    return list_of_labelled_images


# Create dataset as required by torch (a big list of [tensor, label]).
all_labelled_pics = []
for i, letter in enumerate(string.ascii_uppercase):
    # if letter in ['C', 'H', 'I', 'O', 'N']:
    folder_name = os.path.join('..', 'letters_dataset', 'balanced_Sachin', letter)
    one_letter_labelled_pics = get_dataset_from_letter_folder(folder_name, i)
    all_labelled_pics.append(one_letter_labelled_pics)

# Split data in training and testing datasets; create their iterators.
random.seed(5)  # To allow repeatable experiments.
train_dataset = []
test_dataset = []
for sublist in all_labelled_pics:
    n = int(len(sublist) * 0.90)
    random.shuffle(sublist)
    train_dataset += sublist[:n]
    test_dataset += sublist[n:]

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=42, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=7, shuffle=False)


# Training parameters, Loss function and optimizer.
input_size = 28 * 28
hidden_size1 = 150
hidden_size2 = 50
output_size = 26
num_epochs = 15
momentum = 0.9

learning_rate = 0.01

model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)

loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# To allow repeatable experiments.
random_seed = 42
torch.manual_seed(random_seed)
torch.backends.cudnn.enabled = False


# Train the network.
total_step = len(train_loader)
for epoch in range(0, num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28)
        # show_batch(images)
        output = model(images)
        loss = loss_function(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


# Test the network.
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on test images: {} %'.format(100 * correct / total))

print('Che sudata !')

NN_path = os.path.join('..', 'test', 'my_model')
torch.save(model, NN_path)
