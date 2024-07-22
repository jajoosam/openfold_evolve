import torch
import torch.nn as nn

class MSAFeatsModule(nn.Module):
    def __init__(self, msa_feats):
        super(MSAFeatsModule, self).__init__()
        self.msa_cluster_bias = nn.Parameter(torch.zeros_like(msa_feats[3]))

    def forward(self):
        return self.msa_cluster_bias