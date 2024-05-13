import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader

# Assuming the existence of SingleFrameSelector, GlobalSelector, and a dataset class VideoDataset
from single_selector import SingleFrameSelector
from global_selector import GlobalSelector
import wandb

import os
# ##这一串是我登录wandb的秘钥
# os.environ["WANDB_API_KEY"] ='4b00240cca79b70a0ed3661fbb18d5b38f33f129'
# ##详细配置看configs.py
# train_config=TrainConfig()
#
# wandb_config=dict(
#     epoch=train_config.epoch,
#     lr=train_config.lr,
#     batch_size=train_config.batch_size,
# )
#
# wandb.init(
#     project="SMART_MAIN_TRAIN",
#     config=wandb_config
# )

from smart import SmartModel


# Main training function
# def train_model(dataset_path, num_epochs=10, batch_size=32, learning_rate=0.001):
#     # Load dataset
#     dataset = VideoDataset(dataset_path)
#     ##这里下面要加一下训练集和测试集的区分
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     # Initialize MobileNet for feature extraction
#     mobilenet = mobilenet_v2(pretrained=True)
#     mobilenet.eval()  # Set to evaluation mode
# ##num_classes先随便取的
#     smartmodel=SmartModel(num_classes=10)

    # Initialize selectors
#    single_frame_selector = SingleFrameSelector(input_dim=1280, num_classes=dataset.num_classes)  # Adjust input_dim based on MobileNet
#    global_selector = GlobalSelector(num_features=1280, num_classes=dataset.num_classes)  # Adjust num_features based on MobileNet

    # Optimizer (for simplicity, we use a single optimizer for both selectors)
    # optimizer = optim.Adam(smartmodel.parameters(), lr=learning_rate)
    #还要设置criterion

    # Training loop
    # for epoch in range(num_epochs):
    #     for batch in dataloader:
    #         # Extract features for each frame using MobileNet and combine with language features
    #         # combined_features = []
    #         # for frame in batch['frames']:
    #         #     visual_features = mobilenet(frame)
    #         #     top_classes = predict_top_classes(visual_features)  # Implement this function based on MobileNet's output
    #         #     language_features = get_language_features(top_classes, glove_embeddings)
    #         #     combined_features.append(torch.cat((visual_features, language_features), dim=1))
    #         #
    #         # combined_features = torch.stack(combined_features)
    #         #
    #         # # Use selectors to compute importance scores and select frames
    #         # frame_importance_scores = single_frame_selector.get_frame_importance(combined_features)
    #         # selected_frames = select_top_frames(frame_importance_scores)  # Implement based on your selection criteria
    #         #
    #
    #         # # Compute loss and update model (simplified for demonstration)
    #         selected_frames=smartmodel(batch)
    #         loss = compute_loss(selected_frames, batch['labels'])  # Define compute_loss based on your problem
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     ##在这里接下来要写一下evaluate函数,log是在evaluate里面的
    #     wandb.log()
    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    #
    # print("Training complete.")

from load_data import VideoDataset





from load_data import get_dataloader, get_transformer,VideoDataset
from smart import load_smart

# Placeholder functions for parts of the pipeline not detailed in this example
if __name__ == "__main__":
    dataset_path = "../datasets/demo_dataset"
    dataset=VideoDataset(data_path=dataset_path)
    dataloader=get_dataloader(Dataset=dataset)
    model=load_smart()
    for batch in dataloader:
        predictions,frames=model(batch)






