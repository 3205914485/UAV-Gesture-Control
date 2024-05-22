import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from einops import rearrange, repeat, reduce
from PIL import Image
from torch.utils.data import Dataset
import argparse
from utils import read_split_data
import numpy as np
class SingleFrameSelector(nn.Module):
    r"""
        Args:
            num_features(int): the number of expected feature in the input `x` 

        Inputs:
            x(Tensor): Tensor of shape(batch_size, seq_len, num_features)

        Outputs: 
        """

    def __init__(self, num_features: int):
        super(SingleFrameSelector, self).__init__()

        # Define a simple MLP with 2 layers
        self.fc1 = nn.Linear(num_features, num_features // 2)
        # Output one score per frame, thus the second parameter is 1
        self.fc2 = nn.Linear(num_features // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, num_features) / training only need (bs,n_features)
                              representing feature vectors of frames.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len) / training only need (bs,1)
                          representing the contribution score for each frame.
        """

        # batch_size = x.shape[0]
        # x = rearrange(x, 'b s f -> (b s) f')
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)  # For scores between 0 and 1
        # or
        # x = torch.relu(x)  # For non-negative scores
        # x = rearrange(x, '(b s) 1 -> b s', b=batch_size)
        return x

    def get_frame_importance(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the importance score of each frame based on the confidence
        of the ground truth class.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_features)
                              representing feature vectors of frames.

        Returns:
            torch.Tensor: The importance score of each frame, of shape (batch_size,)
        """
        confidences = self.forward(x)
        return confidences


def load_singleframe_selector(model_path,
                              num_features,
                              eval=True,
                              device=torch.device(
                                  "cuda" if torch.cuda.is_available() else "cpu")
                              ):
    model = SingleFrameSelector(num_features=num_features).to(device)
    model.load_state_dict(torch.load(model_path))
    if eval:
        model.eval()
    return model




class MyDataSet(Dataset):
    def __init__(self, images_path: list, scores: np.array, transform=None):
        self.images_path = images_path
        self.scores = scores
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx]).convert('RGB')
        score = self.scores[idx]

        if self.transform:
            img = self.transform(img)

        return img, score

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    scores = np.load('./data/scores.npy')
    scores = scores.astype(np.float32)
    scores_train = scores[:len(train_images_label)]
    scores_val = scores[len(train_images_label):]

    # 使用MobileNetV2的默认transformations
    weights = MobileNet_V2_Weights.DEFAULT
    transform = weights.transforms()

    train_dataset = MyDataSet(images_path=train_images_path, scores=scores_train, transform=transform)
    val_dataset = MyDataSet(images_path=val_images_path, scores=scores_val, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # 定义特征提取器
    feature_extractor = models.MobileNetV2()
    feature_extractor.load_state_dict(state_dict=torch.load(
        '/data4/zst/uav/smart/saved_model/MobileNet_v2.pth'))
    feature_extractor.classifier = nn.Identity()
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    num_features = 1280  # MobileNetV2的特征数量
    single_frame_selector = SingleFrameSelector(num_features).to(device)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.AdamW(single_frame_selector.parameters(), lr=args.lr)

    best_valid_loss = float('inf')
    best_model_path = './data/best_single_frame_selector.pt'

    for epoch in range(args.epochs):
        single_frame_selector.train()
        total_loss = 0
        for train_nodes, train_labels in tqdm(train_loader):
            train_nodes, train_labels = train_nodes.to(device), train_labels.to(device)
            features = feature_extractor(train_nodes)
            features = features.view(features.size(0), -1)
            scores = single_frame_selector(features)
            loss = loss_func(scores.squeeze(), train_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation phase
        single_frame_selector.eval()
        valid_loss = 0
        with torch.no_grad():
            for valid_nodes, valid_labels in tqdm(val_loader):
                valid_nodes, valid_labels = valid_nodes.to(device), valid_labels.to(device)
                features = feature_extractor(valid_nodes)
                features = features.view(features.size(0), -1)
                scores = single_frame_selector(features)
                loss = loss_func(scores.squeeze(), valid_labels)
                valid_loss += loss.item()

        valid_loss /= len(val_loader)
        print(f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {avg_loss}, Valid Loss: {valid_loss}')

        # Save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(single_frame_selector.state_dict(), best_model_path)
            print(f'Best model saved with validation loss: {best_valid_loss}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=101)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data-path', type=str, default="./data/ucf_101_frame")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    train(opt)

