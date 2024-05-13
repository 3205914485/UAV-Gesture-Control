import torch
from torch import nn
from einops import rearrange,repeat


class FrameSelector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,single_score:torch.Tensor,global_relation:torch.Tensor):
        N=global_relation.shape[2]
        relation_score=repeat(single_score,"b s -> b s N",N=N)
        final_score=(relation_score*global_relation).sum(dim=1)/single_score.sum(dim=1)
        return final_score

##测试用代码
# A=torch.Tensor([[[1,2,3]
#               ,[2,3,4]
#               ,[3,4,5]]])
# B=torch.Tensor([[0.3,0.3,0.4]])
# frame_selector=FrameSelector()
# print(frame_selector(B,A))