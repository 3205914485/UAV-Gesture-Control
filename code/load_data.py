import torch

import os
from PIL import Image
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def get_transformer():
    # 定义预处理转换
    preprocess = transforms.Compose([
        transforms.Resize(256),  # 先缩放图像
        transforms.CenterCrop(224),  # 再裁剪到 224 x 224
        transforms.ToTensor(),  # 将图像转换为 torch.Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])
    return preprocess
class VideoDataset(Dataset):

    def __init__(self,data_path="datasets/demo_dataset",transformer=get_transformer()):
        super().__init__()
        self.data_path=data_path
        self.transformer=transformer
        self.frames=[]
        for classes in os.listdir(data_path):
            classes=data_path+"/"+classes
            for video_file in os.listdir(classes):
                video_file=classes+"/"+video_file
                frame_files = [video_file+"/"+f for f in sorted(os.listdir(video_file)) if f .endswith(('jpg','png','jepg'))]
                self.frames.append(frame_files)

    def __len__(self):
        return len(self.frames)
    def __getitem__(self, idx):
        video_frames=self.frames[idx]
        images=[]
        for frame_path in video_frames:
            image=Image.open(frame_path)
            if self.transformer:
                image=self.transformer(image)
            images.append(image)
        images=torch.stack(images)
        return images

def get_dataloader(Dataset:Dataset,batch_size=8)->DataLoader:
    return DataLoader(Dataset,batch_size=8, shuffle=True)