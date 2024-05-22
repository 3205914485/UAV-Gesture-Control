import torch
import torch.nn as nn
from torchvision import models
from single_selector import SingleFrameSelector, load_singleframe_selector
from global_selector import GlobalSelector, load_global_selector
from einops import rearrange


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
        # self.single_frame_selector = SingleFrameSelector(input_dim=1280)
        # self.global_selector = GlobalSelector(num_features=1280, num_classes=num_classes)
        self.single_frame_selector = load_singleframeselector(
            model_path="models/single_frame_selector", num_features=1280)
        self.global_selector = load_globalselector(
            model_path="models/global_frame_selector", num_features=1280, num_classes=10)
        self.final_frame_selector = FrameSelector()

    def forward(self, x) -> torch.Tensor:
        # 使用MobileNet提取特征
        with torch.no_grad():
            features = self.feature_extractor(x)
            features = features.view(features.size(
                0), features.size(1), -1)  # 调整特征形状以匹配选择器的输出
        # 使用SingleFrameSelector选择帧
        selected_frame_features = self.single_frame_selector(features)

        # 使用GlobalSelector进行最终的分类决策
        global_select_result = self.global_selector.get_frame_importance(
            features)
        # 使用final_selector来综合上述结果,从而生成最终选帧
        predictions = self.final_frame_selector(
            selected_frame_features, global_select_result)

        selected_frame = self.get_selected_frame(predictions, x)

        return predictions, selected_frame

    def get_selected_frame(self, predictions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = rearrange(x, "b s f -> (b s) f", b=batch_size)
        predictions = rearrange(predictions, "b s f -> (b s) f", b=batch_size)
        selected_feature = []
        for i in predictions:
            if i:
                selected_feature.append(x[i])
        selected_feature = torch.stack(selected_feature)
        selected_feature = rearrange(
            selected_feature, "(b s) f -> b s f", b=batch_size)
        return selected_feature
    # def predict_top_classes(self,inputs:torch.Tensor)-> torch.Tensor:
    #
    #     batch_size=inputs.shape[0]
    #     inputs=rearrange(inputs,'b s f -> (b s) f')
    #     frame_nums=inputs.shape[0]
    #     outputs=self.mobilenet(inputs)
    #     _,indices=torch.topk(outputs,10,dim=1)
    #
    #     #脑子抽了，但是这几段代码实现挺有意义的
    #     # flat_indices = indices.view(-1)
    #     # selected_frames = []
    #     # for i in range(frame_nums):
    #     #     selected_frames.append(inputs[i][flat_indices[i]])
    #     # selected_frames = torch.stack(selected_frames)
    #     # selected_frames=rearrange(selected_frames,'(b s) 1 -> b s 1',b=batch_size)
    #     return indices


def load_smart():
    return SmartModel().to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
