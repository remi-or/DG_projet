import torch
from torch import nn, Tensor
from itertools import chain

class GridDES(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(13, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1), nn.ReLU(inplace=True),
        )
        for layer in self.body:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, 1.414)
                nn.init.zeros_(layer.bias)
        
    def forward(self,
                x: Tensor, # B, 13
                ) -> Tensor: # B, 1
        return self.body(x)

class MeshDES_base(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.stage1mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1024), nn.ReLU(inplace=True))
        self.stage2mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1), nn.ReLU(inplace=True),
        )
        for layer in chain(self.stage1mlp, self.stage2mlp):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, 1.414)
                nn.init.zeros_(layer.bias)

    def forward(self,
                x: Tensor, # B, M, 4
                ) -> Tensor:
        x = self.stage1mlp(x) # B, 1024, 4
        x = x.max(dim=-2)[0] # this covers the case [1024, 4] as well as [B, 1024, 4]
        x = self.stage2mlp(x)
        return x


class MeshDES_ror(nn.Module):

    def __init__(self,
                 pooling: str = 'mean',
                 leak: float = 0.2,
                 mask_cst: float = -1.0,
                 ) -> None:
        assert pooling in ['mean', 'max']
        self.pooling = pooling
        self.mask_cst = mask_cst
        super().__init__()
        self.stage1mlp = nn.Sequential(
            nn.Linear(4, 64), nn.LeakyReLU(leak, inplace=True),
            nn.Linear(64, 128), nn.LeakyReLU(leak, inplace=True),
            nn.Linear(128, 512), nn.LeakyReLU(leak, inplace=True),
            nn.Linear(512, 1024), nn.LeakyReLU(leak, inplace=True),
        )
        self.stage2mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.LeakyReLU(leak, inplace=True),
            nn.Linear(512, 256), nn.LeakyReLU(leak, inplace=True),
            nn.Linear(256, 1), nn.LeakyReLU(leak, inplace=True),
        )
        for layer in chain(self.stage1mlp, self.stage2mlp):
            if isinstance(layer, nn.Linear):
                gain = (2 / (1 + leak**2)) ** 0.5
                nn.init.xavier_normal_(layer.weight, gain)
                nn.init.zeros_(layer.bias)

    def forward(self,
                x: Tensor, # B, M, 4
                ) -> Tensor:
        mask = (x[:, :, 3] == self.mask_cst) # B, M
        x = self.stage1mlp(x) # B, M, 1024
        # It seems avg-pooling propagates gradient better than max-pooling
        if self.pooling == 'mean':
            x[mask].fill_(0) # B, M, 1024
            x = x.sum(dim=-2) / mask.sum(dim=-1).reshape((-1, 1)) # B, 1024
        elif self.pooling == 'max':
            x[mask].fill_(-torch.inf) # B, M, 1024
            x = x.max(dim=-2)[0] # B, 1024
        x = self.stage2mlp(x) # B, 1
        return x