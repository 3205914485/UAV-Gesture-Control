import torch
import torch.nn as nn
from torchvision import models
from single_selector import SingleFrameSelector
from global_selector import GlobalSelector


class SmartModel(nn.Module):
    def __init__(self, num_classes):
        super(SmartModel, self).__init__()
        # 初始化MobileNet作为特征提取器
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        # 取消MobileNet的最后一层，只用于特征提取
        self.feature_extractor = nn.Sequential(
            *list(self.mobilenet.children())[:-1])
        self.feature_extractor.eval()  # 设置为评估模式

        # 假设SingleFrameSelector和GlobalSelector的输入特征维度为1280
        self.single_frame_selector = SingleFrameSelector(
            input_dim=1280, num_classes=num_classes)
        self.global_selector = GlobalSelector(
            num_features=1280, num_classes=num_classes)

    def forward(self, x):
        # 使用MobileNet提取特征
        with torch.no_grad():
            features = self.feature_extractor(x)
            features = features.view(features.size(
                0), features.size(1), -1)  # 调整特征形状以匹配选择器的输入

        # 使用SingleFrameSelector选择帧
        selected_frame_features = self.single_frame_selector(features)

        # 使用GlobalSelector进行最终的分类决策
        predictions = self.global_selector(selected_frame_features)

        return predictions
