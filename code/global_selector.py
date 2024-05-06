import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple, Union
from utils import generate_frame_pairs
from einops import rearrange, repeat, reduce


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
        global_x1 = torch.einsum('b s f, b s 1 -> b f', x, alpha) / \
            reduce(alpha, 'b s 1 -> b 1', 'sum')
        beta = self.sigmoid(self.spatial_linear(
            torch.cat((rearrange(x, 'b s f -> (b s) f'), repeat(global_x1, 'b f -> (b s) f', s=seq_len)), dim=1)))
        omega = torch.einsum('b s f, b s 1 -> b s f', x,
                             torch.cumsum(beta, dim=1))

        h0 = torch.zeros((self.lstm_layers, batch_size, num_features))
        c0 = torch.zeros((self.lstm_layers, batch_size, num_features))
        h, (hn, cn) = self.lstm(omega, (h0, c0))
        lamda = self.attention_linear2(rearrange(h, 'b s f -> (b s) f'))
        lamda = F.softmax(
            rearrange(lamda, '(b s) 1 -> b s 1', b=batch_size), dim=1)
        global_x2 = torch.einsum('b s f, b s 1 -> b f', omega,
                                 lamda) / reduce(lamda, 'b f 1 -> b 1', 'sum')
        gamma = self.sigmoid(self.temporal_linear(
            torch.cat((rearrange(omega, 'b s f -> (b s) f'), repeat(global_x2, 'b f -> (b s) f', s=seq_len)), dim=1)))
        c = torch.einsum('b s f, b s 1 -> b s f', h,
                         torch.cumsum(gamma, dim=1))
        out = self.classifier(rearrange(c, 'b s f -> (b s) f'))
        return rearrange(out, '(b s) num_classes -> b s num_classes', b=batch_size)

    def get_frame_importance(self, x: Tensor) -> Tensor:
        r"""
        Compute the temporal relation-attention weights of each frame.

        Returns: gamma(Tensor)
            gamma: tensor of shape (batch_size, seq_len), the  temporal relation-attention weights of each frame of each sample 
        """
        x = generate_frame_pairs(x)
        batch_size, seq_len, num_features = x.shape
        alpha = self.sigmoid(self.attention_linear1(x.reshape(-1, num_features))).reshape(
            batch_size, seq_len, 1)
        global_x1 = (alpha * x).sum(dim=1) / \
            alpha.sum(dim=1)  # (batch_size, num_features)
        global_x1 = global_x1.unsqueeze(
            1).expand(-1, seq_len, -1).reshape(-1, num_features)
        beta = self.sigmoid(self.spatial_linear(
            torch.cat((x.reshape(-1, num_features), global_x1), dim=1))).reshape(
            batch_size, seq_len, 1)
        omega = (beta * x).cumsum(dim=1)  # (batch_size, seq_len, num_features)

        h0 = torch.zeros((self.lstm_layers, batch_size, num_features))
        c0 = torch.zeros((self.lstm_layers, batch_size, num_features))
        h, (hn, cn) = self.lstm(omega, (h0, c0))
        lamda = self.attention_linear2(
            h.reshape(-1, num_features)).reshape(-1, seq_len, 1)
        lamda = F.softmax(lamda, dim=1)
        global_x2 = (lamda * omega).sum(dim=1) / \
            lamda.sum(dim=1)  # (batch_size, num_features)
        global_x2 = global_x2.unsqueeze(
            1).expand(-1, seq_len, -1).reshape(-1, num_features)
        gamma = self.sigmoid(self.temporal_linear(
            torch.cat((omega.reshape(-1, num_features), global_x2), dim=1))).reshape(
            batch_size, seq_len)
        return gamma


# model = GlobalSelector(20, 2, 2)
# x = torch.randn((16, 10, 20))
# print(model.einops_forward(x).shape)
