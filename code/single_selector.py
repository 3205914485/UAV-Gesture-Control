import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

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
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, num_features)
                              representing feature vectors of frames.
        
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len)
                          representing the contribution score for each frame.
        """

        batch_size = x.shape[0]
        x = rearrange(x, 'b s f -> (b s) f')
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)  # For scores between 0 and 1
        # or
        # x = torch.relu(x)  # For non-negative scores
        x = rearrange(x, '(b s) 1 -> b s', b=batch_size)
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

def load_singleframeselector(model_path,
                             num_features,
                             eval=True,
                             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                             ):
    model=SingleFrameSelector(num_features=num_features).to(device)
    model.load_state_dict(torch.load(model_path))
    if eval:
        model.eval()
    return model