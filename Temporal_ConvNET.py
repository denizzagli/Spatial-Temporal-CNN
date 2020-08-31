import os
import cv2
import json
import numpy as np
from torch.utils import data
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

def get_infos(input_file, dataset_type):
    info = []
    path = "MS-ASL/" + input_file
    videos_name = get_videos_name(dataset_type)
    with open(path) as json_file:
        dataset = json.load(json_file)
        for data in dataset:
            if int(data["label"]) <= 9:
                temp_list = []
                url = "watch" + data["url"].split("watch")[1]
                if url in list(videos_name.keys()):
                    start_time = float(data["start_time"])
                    end_time = float(data["end_time"])
                    label = int(data["label"])
                    temp_list.append(url)
                    temp_list.append(start_time)
                    temp_list.append(end_time)
                    temp_list.append(label)
                    info.append(temp_list)
    return info

def get_videos_name(dataset_type):
    name_dict = {}
    folder_path, dirs, files = next(os.walk("HW4_DATA/videos/" + dataset_type))
    for file in files:
        temp = file.split(".")
        name_dict[temp[0]] = file
    return name_dict

train_info = get_infos("MSASL_train.json", "train")
val_info = get_infos("MSASL_val.json", "val")
test_info = get_infos("MSASL_test.json", "test")

def get_flow_paths(dataset_type, infos):
    labels = []
    for info in infos:
        labels.append(int(info[3]))
    folder_path, dirs, files = next(os.walk("Optical_Flows/" + dataset_type))
    paths = []
    for directory in dirs:
        folder_path, dirs2, files = next(os.walk("Optical_Flows/" + dataset_type + "/" + str(directory)))
        files_list = []
        for input_file in files:
            files_list.append(folder_path + "/" + input_file)
        paths.append(files_list)
    temp_dict = {}
    for path in paths:
        temp_list = path[0].split("/")
        temp_dict[int(temp_list[2])] = path
    temp_list = list(temp_dict.keys())
    temp_list.sort()
    temp_list2 = []
    for item in temp_list:
        temp_list2.append(temp_dict[item])
    paths = temp_list2
    return paths, labels

paths_train, labels_train = get_flow_paths("train", train_info)
paths_val, labels_val = get_flow_paths("val", val_info)
paths_test, labels_test = get_flow_paths("test", test_info)

def path_to_images(paths):
    images = []
    for paths2 in paths:
        temp_list = []
        for path in paths2:
            image = Image.open(path).convert('L')
            image = image.resize((224, 224), Image.ANTIALIAS)
            image = np.asarray(image).tolist()
            temp_list.append(image)
        images.append(temp_list)
    images = np.array(images)
    return images

train_images = path_to_images(paths_train)
val_images = path_to_images(paths_val)
test_images = path_to_images(paths_test)

# Dataset Class

class Dataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

# Model Class

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(200704, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# Parameters

params = {'batch_size': 4, 'shuffle': True, 'num_workers': 0}
max_epochs = 3

# Data Loader

training_set = Dataset(train_images, labels_train)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(val_images, labels_val)
validation_generator = data.DataLoader(validation_set, **params)

test_set = Dataset(test_images, labels_test)
test_generator = data.DataLoader(test_set, **params)

# CUDA optimization

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Model creation

model = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
model = model.cuda()
criterion = criterion.cuda()

# Training part

total_step = len(paths_train)
total_step_val = len(paths_val)
loss_list = []
acc_list = []
train_loss = []
validation_loss = []
train_accuracy = []
validation_accuracy = []
for epoch in range(max_epochs):
    step_train = 0
    step_validation = 0
    for local_batch, local_labels in training_generator:
        # Model computations
        # local_batch = local_batch.unsqueeze(1)
        local_batch = local_batch.type(torch.cuda.FloatTensor)
        local_labels = local_labels.type(torch.cuda.LongTensor)
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Run the forward pass
        outputs = model(local_batch)
        loss = criterion(outputs, local_labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = local_labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == local_labels).sum().item()
        acc_list.append(correct / total)

        if True:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, max_epochs, step_train, int(total_step / params['batch_size']), loss.item(),
                          (correct / total) * 100))
        train_loss.append(loss.item())
        train_accuracy.append((correct / total) * 100)
        step_train = step_train + 1

    for local_batch, local_labels in validation_generator:

        # Model computations
        # local_batch = local_batch.unsqueeze(1)
        local_batch = local_batch.type(torch.cuda.FloatTensor)
        local_labels = local_labels.type(torch.cuda.LongTensor)
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Run the forward pass
        outputs = model(local_batch)
        loss = criterion(outputs, local_labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # Track the accuracy
        total = local_labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == local_labels).sum().item()
        acc_list.append(correct / total)

        if True:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, max_epochs, step_validation, int(total_step_val / params['batch_size']), loss.item(),
                          (correct / total) * 100))
        validation_loss.append(loss.item())
        validation_accuracy.append((correct / total) * 100)
        step_validation = step_validation + 1

# Testing Part

test_loss = []
test_accuracy = []
step_test = 0
total_step = len(paths_test)
actual = []
predicted_values = []
for local_batch, local_labels in test_generator:
    # Model computations
    #local_batch = local_batch.unsqueeze(1)
    local_batch = local_batch.type(torch.cuda.FloatTensor)
    local_labels = local_labels.type(torch.cuda.LongTensor)
    local_batch, local_labels = local_batch.to(device), local_labels.to(device)

    # Run the forward pass
    outputs = model(local_batch)
    loss = criterion(outputs, local_labels)
    loss_list.append(loss.item())

    # Backprop and perform Adam optimisation
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # Track the accuracy
    total = local_labels.size(0)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == local_labels).sum().item()
    predicted = predicted.cpu().detach().numpy()
    local_labels = local_labels.cpu().detach().numpy()
    print(predicted)
    for item in predicted:
        predicted_values.append(item)
    for item in local_labels:
        actual.append(item)
    print(local_labels)
    acc_list.append(correct / total)

    if True:
        print('Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                .format(step_test + 1, int(total_step / params['batch_size']), loss.item(),
                        (correct / total) * 100))
    test_loss.append(loss.item())
    test_accuracy.append((correct / total) * 100)
    step_test = step_test + 1

# Experimental Results

accuracy = sum(test_accuracy) / len(test_accuracy)
print("%" + "%.2f" % accuracy)
accuracy = sum(train_accuracy) / len(train_accuracy)
print("%" + "%.2f" % accuracy)
accuracy = sum(validation_accuracy) / len(validation_accuracy)
print("%" + "%.2f" % accuracy)
plt.plot(train_loss, label='Training Loss')
plt.legend()
plt.show()
plt.plot(validation_loss, label='Validation Loss')
plt.legend()
plt.show()
plt.plot(train_accuracy, label='Training Accuracy')
plt.legend()
plt.show()
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.legend()
plt.show()
