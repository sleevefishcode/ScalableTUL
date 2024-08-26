import torch
import torch.nn as nn
import torch.nn.functional as F




class ProjectionLayer(nn.Module):
    def __init__(self,embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.projection = nn.Linear(self.embed_size,int(self.embed_size/2))         


    def forward(self, x):
        x = self.projection(x)

        return x