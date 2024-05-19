import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils import generate_frame_pairs
from einops import rearrange, repeat, reduce
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class GlobalSelector(nn.Module):
    r"""
        Args:
            num_features(int): the number of expected feature in the input `x` 
            num_classes(int): the number of video classes
            lstm_layers(int): the number of stacking LSTM layers
            dropout(float): the dropout rate in LSTM
                Default: `0.3`

        Inputs:
            x(Tensor): Tensor of shape(batch_size, seq_len, num_features)

        Outputs: 
        """

    def __init__(self, num_features: int, num_classes: int, lstm_layers: int = 1, dropout: float = 0.3) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.sigmoid = nn.Sigmoid()
        self.attention_linear1 = nn.Linear(2*num_features, 1)
        self.spatial_linear = nn.Linear(4*num_features, 1)
        self.attention_linear2 = nn.Linear(2*num_features, 1)
        self.temporal_linear = nn.Linear(4*num_features, 1)
        self.lstm = nn.LSTM(input_size=2*num_features,
                            hidden_size=2*num_features,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout)
        self.lstm_layers = lstm_layers
        self.classifier = nn.Linear(2*num_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Classify each frame to train the model.

        Returns: out(Tensor)
            out: tensor of shape (batch_size, seq_len, num_classes), the classification results of each frame of each sample
        """
        # x = generate_frame_pairs(x)
        # batch_size, seq_len, num_features = x.shape
        # alpha = self.sigmoid(self.attention_linear1(x.reshape(-1, num_features))).reshape(
        #     batch_size, seq_len, 1)
        # global_x1 = (alpha * x).sum(dim=1) / \
        #     alpha.sum(dim=1)  # (batch_size, num_features)
        # global_x1 = global_x1.unsqueeze(
        #     1).expand(-1, seq_len, -1).reshape(-1, num_features)
        # beta = self.sigmoid(self.spatial_linear(
        #     torch.cat((x.reshape(-1, num_features), global_x1), dim=1))).reshape(
        #     batch_size, seq_len, 1)
        # omega = (beta * x).cumsum(dim=1)  # (batch_size, seq_len, num_features)

        # h0 = torch.zeros((self.lstm_layers, batch_size, num_features))
        # c0 = torch.zeros((self.lstm_layers, batch_size, num_features))
        # h, (hn, cn) = self.lstm(omega, (h0, c0))
        # lamda = self.attention_linear2(
        #     h.reshape(-1, num_features)).reshape(-1, seq_len, 1)
        # lamda = F.softmax(lamda, dim=1)
        # global_x2 = (lamda * omega).sum(dim=1) / \
        #     lamda.sum(dim=1)  # (batch_size, num_features)
        # global_x2 = global_x2.unsqueeze(
        #     1).expand(-1, seq_len, -1).reshape(-1, num_features)
        # gamma = self.sigmoid(self.temporal_linear(
        #     torch.cat((omega.reshape(-1, num_features), global_x2), dim=1))).reshape(
        #     batch_size, seq_len, 1)
        # c = (gamma * h).cumsum(dim=1).reshape(-1, num_features)
        # out = self.classifier(c).reshape(batch_size, seq_len, self.num_classes)
        # return out

        x = generate_frame_pairs(x)
        batch_size, seq_len, num_features = x.shape
        alpha = self.sigmoid(self.attention_linear1(
            rearrange(x, 'b s f -> (b s) f')))
        alpha = rearrange(alpha, '(b s) 1 -> b s 1', b=batch_size)
        global_x1 = torch.einsum('b s f, b s o -> b f', x, alpha) / \
            reduce(alpha, 'b s 1 -> b 1', 'sum')
        beta = self.sigmoid(self.spatial_linear(
            torch.cat((rearrange(x, 'b s f -> (b s) f'), repeat(global_x1, 'b f -> (b s) f', s=seq_len)), dim=1)))
        beta = rearrange(beta, '(b s) 1 -> b s 1', b=batch_size)
        omega = torch.einsum('b s f, b s o -> b s f', x,
                             torch.cumsum(beta, dim=1))

        h0 = torch.zeros((self.lstm_layers, batch_size,
                         num_features)).to(x.device)
        c0 = torch.zeros((self.lstm_layers, batch_size,
                         num_features)).to(x.device)
        h, (hn, cn) = self.lstm(omega, (h0, c0))
        lamda = self.attention_linear2(rearrange(h, 'b s f -> (b s) f'))
        lamda = F.softmax(
            rearrange(lamda, '(b s) 1 -> b s 1', b=batch_size), dim=1)
        global_x2 = torch.einsum('b s f, b s o -> b f', omega,
                                 lamda) / reduce(lamda, 'b f 1 -> b 1', 'sum')
        gamma = self.sigmoid(self.temporal_linear(
            torch.cat((rearrange(omega, 'b s f -> (b s) f'), repeat(global_x2, 'b f -> (b s) f', s=seq_len)), dim=1)))
        gamma = rearrange(gamma, '(b s) 1 -> b s 1', b=batch_size)
        c = torch.einsum('b s f, b s o -> b s f', h,
                         torch.cumsum(gamma, dim=1))
        out = self.classifier(rearrange(c, 'b s f -> (b s) f'))
        return rearrange(out, '(b s) num_classes -> b num_classes s', b=batch_size)

    def get_frame_importance(self, x: Tensor) -> Tensor:
        r"""
        Compute the temporal relation-attention weights of each frame.

        Returns: gamma(Tensor)
            gamma: tensor of shape (batch_size, seq_len), the  temporal relation-attention weights of each frame of each sample
        """
        x = generate_frame_pairs(x)
        batch_size, seq_len, num_features = x.shape
        alpha = self.sigmoid(self.attention_linear1(
            rearrange(x, 'b s f -> (b s) f')))
        alpha = rearrange(alpha, '(b s) 1 -> b s 1', b=batch_size)
        global_x1 = torch.einsum('b s f, b s o -> b f', x, alpha) / \
            reduce(alpha, 'b s 1 -> b 1', 'sum')
        beta = self.sigmoid(self.spatial_linear(
            torch.cat((rearrange(x, 'b s f -> (b s) f'), repeat(global_x1, 'b f -> (b s) f', s=seq_len)), dim=1)))
        beta = rearrange(beta, '(b s) 1 -> b s 1', b=batch_size)
        omega = torch.einsum('b s f, b s o -> b s f', x,
                             torch.cumsum(beta, dim=1))

        h0 = torch.zeros((self.lstm_layers, batch_size,
                         num_features)).to(x.device)
        c0 = torch.zeros((self.lstm_layers, batch_size,
                         num_features)).to(x.device)
        h, (hn, cn) = self.lstm(omega, (h0, c0))
        lamda = self.attention_linear2(rearrange(h, 'b s f -> (b s) f'))
        lamda = F.softmax(
            rearrange(lamda, '(b s) 1 -> b s 1', b=batch_size), dim=1)
        global_x2 = torch.einsum('b s f, b s o -> b f', omega,
                                 lamda) / reduce(lamda, 'b f 1 -> b 1', 'sum')
        gamma = self.sigmoid(self.temporal_linear(
            torch.cat((rearrange(omega, 'b s f -> (b s) f'), repeat(global_x2, 'b f -> (b s) f', s=seq_len)), dim=1)))
        gamma = rearrange(gamma, '(b s) 1 -> b s 1', b=batch_size)
        return gamma


def load_global_selector(model_path,
                         num_features,
                         num_classes,
                         eval=True,
                         device=torch.device(
                             "cuda" if torch.cuda.is_available() else "cpu")
                         ):
    model = GlobalSelector(num_features=num_features,
                           num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    if eval:
        model.eval()
    return model


class MyDataSet(Dataset):
    def __init__(self, data, mode):
        path = '/data3/whr/zst/uav/smart/data/'
        mask = torch.load(path+f'mask/{mode}_mask.pt')
        self.data = data[mask]
        self.labels = torch.load(path+'labels.pt')[mask]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("Starting loading features...")
    with open('../data/video_feature.json') as f:
        data = json.load(f)
    # data = torch.rand(13320, 10, 1280)
    print('Finished loading.')
    train_dataset = MyDataSet(data, mode='train')
    val_dataset = MyDataSet(data, mode='valid')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, pin_memory=True, num_workers=4)
    global_frame_selector = GlobalSelector(num_features=1280,
                                           num_classes=101,
                                           lstm_layers=3).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        global_frame_selector.parameters(), lr=args.lr)

    best_valid_loss = float('inf')
    best_model_path = '/data3/whr/zst/uav/smart/data/best_global_frame_selector.pt'

    print("Starting Training.")
    for epoch in range(args.epochs):
        global_frame_selector.train()
        
        y_true = []
        y_pred = []
        total_loss = 0
        
        for train_feature, train_labels in tqdm(train_loader):
            train_feature = train_feature.to(device)
            train_labels = repeat(train_labels, 'b -> b s', s=train_feature.size(1)).to(device)
            out = global_frame_selector(train_feature)

            loss = loss_func(out, train_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            y_true.append(train_labels)
            y_pred.append(out.argmax(dim=1))

        y_pred = [tensor.cpu().numpy() for tensor in y_pred]
        y_pred = np.concatenate(y_pred, axis=0)

        y_true = [tensor.cpu().numpy() for tensor in y_true]
        y_true = np.concatenate(y_true, axis=0)

        avg_loss = total_loss / len(train_loader)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        print(
            f"Epoch {epoch+1}: train loss: {avg_loss:.4f}, f1: {f1:.4f}, auc: {auc:.4f}, acc: {acc:.4f}")

        # Validation phase
        global_frame_selector.eval()
        
        y_true = []
        y_pred = []
        total_loss = 0
        
        with torch.no_grad():
            for valid_feature, valid_labels in tqdm(val_loader):
                valid_feature = valid_feature.to(device)
                valid_labels = repeat(
                    train_labels, 'b -> b s', s=valid_feature.size(1)).to(device)
                out = global_frame_selector(valid_feature)

                total_loss += loss.item()

                y_true.append(valid_labels)
                y_pred.append(out.argmax(dim=1))

        y_pred = [tensor.cpu().numpy() for tensor in y_pred]
        y_pred = np.concatenate(y_pred, axis=0)

        y_true = [tensor.cpu().numpy() for tensor in y_true]
        y_true = np.concatenate(y_true, axis=0)

        avg_loss = total_loss / len(val_loader)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        print(
            f"Epoch {epoch+1}: valid loss: {avg_loss:.4f}, f1: {f1:.4f}, auc: {auc:.4f}, acc: {acc:.4f}")

        # Save the best model
        if avg_loss < best_valid_loss:
            best_valid_loss = avg_loss
            torch.save(global_frame_selector.state_dict(), best_model_path)
            print(f'Best model saved with validation loss: {best_valid_loss}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=101)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--data-path', type=str,
                        default="./data/ucf_101_frame")
    parser.add_argument('--device', default='cuda:1',
                        help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    train(opt)
