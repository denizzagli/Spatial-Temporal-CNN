# Using the dataset, I create a new dataset suitable for the desired tasks.

import os
import cv2
import json
import numpy as np

# Reads the necessary information from the JSON file.

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

# Reads the video names in the dataset.

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

# Frames are created according to the desired time intervals.

def generate_frames(infos, dataset_type):
    counter = 0
    videos_name = get_videos_name(dataset_type)
    for info in infos:
        info[2] -= 0.2
        os.mkdir("Video_Frames/" + dataset_type + "/" + str(counter))
        video_path = "HW4_DATA/videos/" + dataset_type +"/" + videos_name[info[0]]
        start_time_ms = int(float(info[1]) * 1000)
        stop_time_ms = int(float(info[2]) * 1000)
        vidcap = cv2.VideoCapture(video_path)
        count = 0
        success = True
        while success and vidcap.get(cv2.CAP_PROP_POS_MSEC) < start_time_ms:
            success, image = vidcap.read()
        while success and vidcap.get(cv2.CAP_PROP_POS_MSEC) <= stop_time_ms:
            success, image = vidcap.read()
            frame_path = "Video_Frames/" + dataset_type + "/" + str(counter) + "/" + str(count) + ".jpg"
            cv2.imwrite(frame_path, image)
            count += 1
        counter += 1

generate_frames(train_info, "train")
generate_frames(val_info, "val")
generate_frames(test_info, "test")

# Flows are created according to the desired time intervals.

os.mkdir("Optical_Flows")

def generate_optical_flows(dataset_type):
    os.mkdir("Optical_Flows/" + dataset_type)
    folder_path, dirs, files = next(os.walk("Video_Frames/" + dataset_type))
    dir_index = 0
    for directory in dirs:
        folder_path2, dirs2, files = next(os.walk("Video_Frames/" + dataset_type + "/" + directory))
        file_size = len(files)
        counter = int(file_size/10)
        index = 0
        indices = []
        for count in range(10):
            indices.append(str(index))
            index = index + counter
        dir_name = "Optical_Flows/" + dataset_type + "/" + str(dir_index)
        os.mkdir(dir_name)
        for index in range(0, 9):
            image1_path = "Video_Frames/" + dataset_type + "/" + directory + "/" + str(indices[index]) + ".jpg"
            image2_path = "Video_Frames/" + dataset_type + "/" + directory + "/" + str(indices[index + 1]) + ".jpg"
            image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
            flow = cv2.calcOpticalFlowFarneback(image1, image2, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
            flow = np.transpose(flow,(2,0,1))
            cv2.imwrite(dir_name + "/" + str(index) + "-x.jpg", flow[0])
            cv2.imwrite(dir_name + "/" + str(index) + "-y.jpg", flow[1])
        print(str(dir_index) + "/" + str(len(dirs)))
        dir_index += 1

generate_optical_flows("train")
generate_optical_flows("val")
generate_optical_flows("test")
