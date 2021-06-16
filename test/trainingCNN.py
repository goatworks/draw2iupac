import torch
import torch.nn as nn
# import torchvision
# import cv2
import numpy as np
import string
import os
import random

from convulutional_neural_network import Conv_NeuralNetwork


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
        image = image.reshape([1, 28, 28])
        labelled_image = [image, label]
        list_of_labelled_images.append(labelled_image)
    return list_of_labelled_images

print("Let's strat")
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

conv1_in_chan = 1
conv1_out_chan = 10
conv2_out_chan = 20
conv_kernel_size = 3
pool_kernel_size = 2

hidden_size1 = 980
hidden_size2 = 150
output_size = 26

num_epochs = 7
momentum = 0.9

learning_rate = 0.01

model = Conv_NeuralNetwork(conv1_in_chan, conv1_out_chan, conv2_out_chan, conv_kernel_size, pool_kernel_size,
                           hidden_size1, hidden_size2, output_size)

loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# To allow repeatable experiments.
random_seed = 42
torch.manual_seed(random_seed)
torch.backends.cudnn.enabled = False

print("start training")
# Train the network.
total_step = len(train_loader)
loss_evolution = []

for epoch in range(0, num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # images = images.reshape(-1, 28*28)
        # show_batch(images)
        output = model(images)
        loss = loss_function(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_evolution.append(loss.item())

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

print("start testing")
# Test the network.
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        # images = images.reshape(-1, 28*28)
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on test images: {} %'.format(100 * correct / total))

print('Che sudata !')

NN_path = os.path.join('..', 'test', 'my_model')
torch.save(model, NN_path)

from matplotlib import pyplot as plt
plt.plot(loss_evolution)
plt.show()

print("End.")
