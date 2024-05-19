# 用预训练好的MobileNetV2模型提取每个视频每一帧的特征，作为SMART模型的输入
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import MobileNetV2, MobileNet_V2_Weights
import torch.nn as nn
import cv2 as cv
import os
from tqdm import tqdm
import json
from einops import rearrange

device =  torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# 加载模型
model = MobileNetV2()
model.load_state_dict(state_dict=torch.load(
    '/data4/zst/uav/smart/saved_model/MobileNet_v2.pth'))
model.classifier = nn.Identity()
model = model.to(device)
model.eval()
# 加载需要的数据变换
weight = MobileNet_V2_Weights.DEFAULT
preprocess = weight.transforms()

data_folder = '/data4/zst/uav/smart/data/ucf_101_frame'

labels = []
features = [] # 由于视频帧数不同，以列表方式存储特征，而不是另外增加维度对齐存储
class_list = os.listdir(data_folder)
class_list.sort()
video_bar = tqdm(enumerate(class_list))

for id, class_ in video_bar:
    class_folder = os.path.join(data_folder, class_)
    labels += [id] * len(os.listdir(class_folder))

    video_list = os.listdir(class_folder)
    video_list.sort()
    class_par = tqdm(video_list)

    for video in class_par:
        class_par.set_description(f'Processing {class_} videos')
        video_folder = os.path.join(class_folder, video)
        video_frames = []

        frame_list = os.listdir(video_folder)
        frame_list.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))
        for frame in (frame_list):
            frame_path = os.path.join(video_folder, frame)
            image = cv.imread(cv.samples.findFile(frame_path))
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB) #cv2 读入是BGR格式，直接转化tensor会导致图像理解问题，需要用官方文档指定的PIL（RGB）格式。
            image = rearrange(torch.from_numpy(image), 'h w c -> c h w')
            image = preprocess(image)
            video_frames.append(image)
        with torch.no_grad():
            video_frames = model(torch.stack(video_frames).to(device))
        features.append(video_frames.cpu().tolist())


labels = torch.Tensor(labels)
print(labels.shape)
torch.save(labels, '/data4/zst/uav/smart/data/labels.pt')


with open('/data4/zst/uav/smart/data/video_feature.json', 'w') as f:
    json.dump(features, f)

