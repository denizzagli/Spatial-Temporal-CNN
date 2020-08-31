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

def get_frame_paths_and_labels(dataset_type, infos):
    labels = []
    paths = []
    for info in infos:
        labels.append(int(info[3]))
    folder_path, dirs, files = next(os.walk("Video_Frames/" + dataset_type))
    for directory in dirs:
        folder_path, dirs2, files = next(os.walk("Video_Frames/" + dataset_type + "/" + str(directory)))
        temp_file = files[int(len(files) / 2)]
        path = folder_path + "/" + temp_file
        paths.append(path)
    temp_dict = {}
    for path in paths:
        temp_list = path.split("/")
        temp_dict[int(temp_list[2])] = path
    temp_list = list(temp_dict.keys())
    temp_list.sort()
    temp_list2 = []
    for item in temp_list:
        temp_list2.append(temp_dict[item])
    paths = temp_list2
    return paths, labels

paths_train, labels_train = get_frame_paths_and_labels("train", train_info)
paths_val, labels_val = get_frame_paths_and_labels("val", val_info)
paths_test, labels_test = get_frame_paths_and_labels("test", test_info)

def path_to_image(paths):
    images = []
    for path in paths:
        image = Image.open(path).convert('RGB').split()
        r = image[0].resize((224, 224), Image.ANTIALIAS)
        g = image[1].resize((224, 224), Image.ANTIALIAS)
        b = image[2].resize((224, 224), Image.ANTIALIAS)
        r = np.asarray(r).tolist()
        g = np.asarray(g).tolist()
        b = np.asarray(b).tolist()
        temp_list = []
        temp_list.append(r)
        temp_list.append(g)
        temp_list.append(b)
        images.append(temp_list)
    images = np.array(images)
    return images

train_images = path_to_image(paths_train)
val_images = path_to_image(paths_val)
test_images = path_to_image(paths_test)

def get_flow_paths(dataset_type, infos):
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
    return paths

paths_train_flow = get_flow_paths("train", train_info)
paths_val_flow = get_flow_paths("val", val_info)
paths_test_flow = get_flow_paths("test", test_info)

def path_to_images_flow(paths):
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

train_images_flow = path_to_images_flow(paths_train_flow)
val_images_flow = path_to_images_flow(paths_val_flow)
test_images_flow = path_to_images_flow(paths_test_flow)

# Dataset class

class Dataset(data.Dataset):
    def __init__(self, spatial_images, flow_images, labels):
        self.spatial_images = spatial_images
        self.flow_images = flow_images
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        spatial_image = self.spatial_images[index]
        flow_image = self.flow_images[index]
        label = self.labels[index]
        return spatial_image, flow_image, label

# Model classes (For spatial and temporal)

class CNN_Spatial(nn.Module):
    def __init__(self):
        super(CNN_Spatial, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
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

class CNN_Temporal(nn.Module):
    def __init__(self):
        super(CNN_Temporal, self).__init__()
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

training_set = Dataset(train_images, train_images_flow, labels_train)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(val_images, val_images_flow, labels_val)
validation_generator = data.DataLoader(validation_set, **params)

test_set = Dataset(test_images, test_images_flow, labels_test)
test_generator = data.DataLoader(test_set, **params)

# CUDA optimization

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Model creation

model_spatial = CNN_Spatial()
model_temporal = CNN_Temporal()
criterion_spatial = torch.nn.CrossEntropyLoss()
criterion_temporal = torch.nn.CrossEntropyLoss()
optimizer_spatial = torch.optim.Adam(model_spatial.parameters(), lr=0.00002)
optimizer_temporal = torch.optim.Adam(model_temporal.parameters(), lr=0.00002)
model_spatial = model_spatial.cuda()
model_temporal = model_temporal.cuda()
criterion_spatial = criterion_spatial.cuda()
criterion_temporal = criterion_temporal.cuda()

# Training part

total_step = len(paths_train)
total_step_val = len(paths_val)
loss_list_spatial = []
loss_list_temporal = []
acc_list = []
train_loss = []
validation_loss = []
train_accuracy = []
validation_accuracy = []
for epoch in range(max_epochs):
    step_train = 0
    step_validation = 0
    for local_batch_spatial, local_batch_flow, local_labels in training_generator:
        # Model computations
        # local_batch = local_batch.unsqueeze(1)
        local_batch_spatial = local_batch_spatial.type(torch.cuda.FloatTensor)
        local_batch_flow =  local_batch_flow.type(torch.cuda.FloatTensor)
        local_labels = local_labels.type(torch.cuda.LongTensor)
        local_batch_spatial, local_batch_flow, local_labels = local_batch_spatial.to(device), local_batch_flow.to(device), local_labels.to(device)

        # Run the forward pass
        outputs_spatial = model_spatial(local_batch_spatial)
        outputs_temporal = model_temporal(local_batch_flow)
        outputs_sum = outputs_spatial + outputs_temporal
        loss_spatial = criterion_spatial(outputs_spatial, local_labels)
        loss_temporal = criterion_temporal(outputs_temporal, local_labels)
        loss_total = loss_spatial + loss_temporal
        loss_list_spatial.append(loss_spatial.item())
        loss_list_temporal.append(loss_temporal.item())

        # Backprop and perform Adam optimisation
        optimizer_spatial.zero_grad()
        optimizer_temporal.zero_grad()
        loss_total.backward()
        optimizer_spatial.step()
        optimizer_temporal.step()

        # Track the accuracy
        total = local_labels.size(0)
        _, predicted = torch.max(outputs_sum.data, 1)
        correct = (predicted == local_labels).sum().item()
        acc_list.append(correct / total)

        if True:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, max_epochs, step_train, int(total_step / params['batch_size']), loss_total.item(),
                          (correct / total) * 100))
        train_loss.append(loss_total.item())
        train_accuracy.append((correct / total) * 100)
        step_train = step_train + 1

    for local_batch_spatial, local_batch_flow, local_labels in validation_generator:

        # Model computations
        # local_batch = local_batch.unsqueeze(1)
        local_batch_spatial = local_batch_spatial.type(torch.cuda.FloatTensor)
        local_batch_flow = local_batch_flow.type(torch.cuda.FloatTensor)
        local_labels = local_labels.type(torch.cuda.LongTensor)
        local_batch_spatial, local_batch_flow, local_labels = local_batch_spatial.to(device), local_batch_flow.to(device), local_labels.to(device)

        # Run the forward pass
        outputs_spatial = model_spatial(local_batch_spatial)
        outputs_temporal = model_temporal(local_batch_flow)
        outputs_sum = outputs_spatial + outputs_temporal
        loss_spatial = criterion_spatial(outputs_spatial, local_labels)
        loss_temporal = criterion_temporal(outputs_temporal, local_labels)
        loss_total = loss_spatial + loss_temporal
        loss_list_spatial.append(loss_spatial.item())
        loss_list_temporal.append(loss_temporal.item())

        # Backprop and perform Adam optimisation
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # Track the accuracy
        total = local_labels.size(0)
        _, predicted = torch.max(outputs_sum.data, 1)
        correct = (predicted == local_labels).sum().item()
        acc_list.append(correct / total)

        if True:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, max_epochs, step_validation, int(total_step_val / params['batch_size']), loss_total.item(),
                          (correct / total) * 100))
        validation_loss.append(loss_total.item())
        validation_accuracy.append((correct / total) * 100)
        step_validation = step_validation + 1

# Testing part

test_loss = []
test_accuracy = []
step_test = 0
total_step = len(paths_test)
actual = []
predicted_values = []
for local_batch_spatial, local_batch_flow, local_labels in test_generator:
    # Model computations
    #local_batch = local_batch.unsqueeze(1)
    local_batch_spatial = local_batch_spatial.type(torch.cuda.FloatTensor)
    local_batch_flow = local_batch_flow.type(torch.cuda.FloatTensor)
    local_labels = local_labels.type(torch.cuda.LongTensor)
    local_batch_spatial, local_batch_flow, local_labels = local_batch_spatial.to(device), local_batch_flow.to(device), local_labels.to(device)

    # Run the forward pass
    outputs_spatial = model_spatial(local_batch_spatial)
    outputs_temporal = model_temporal(local_batch_flow)
    outputs_sum = outputs_spatial + outputs_temporal
    loss_spatial = criterion_spatial(outputs_spatial, local_labels)
    loss_temporal = criterion_temporal(outputs_temporal, local_labels)
    loss_total = loss_spatial + loss_temporal
    loss_list_spatial.append(loss_spatial.item())
    loss_list_temporal.append(loss_temporal.item())

    # Backprop and perform Adam optimisation
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # Track the accuracy
    total = local_labels.size(0)
    _, predicted = torch.max(outputs_sum.data, 1)
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
                .format(step_test + 1, int(total_step / params['batch_size']), loss_total.item(),
                        (correct / total) * 100))
    test_loss.append(loss_total.item())
    test_accuracy.append((correct / total) * 100)
    step_test = step_test + 1

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
