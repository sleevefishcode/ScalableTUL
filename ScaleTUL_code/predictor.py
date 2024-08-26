import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictorLayer(nn.Module):
    def __init__(self,embed_size,user_embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.user_embed_size = int(embed_size/2)
        # self.fc = nn.Linear(self.embed_size,self.user_embed_size)
        self.fc = nn.Sequential(
             nn.Linear(self.embed_size, self.embed_size*2),
             nn.ReLU(),
             nn.Linear(self.embed_size*2, self.embed_size),
             nn.ReLU(),
             nn.Linear(self.embed_size, self.user_embed_size)
        )

    def forward(self, x):
        x = self.fc(x)

        return x