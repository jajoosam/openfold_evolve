import torch
import torch.nn as nn

from utils import get_fape_loss

class DropoutMaskModule(nn.Module):
    def __init__(self, num_residues, channels_msa, channels_msa_extra, channels_pair, dropout_rate=0.15, init="halves", device="cuda"):
        super(DropoutMaskModule, self).__init__()

        self.dropout_rate = dropout_rate
        self.num_residues = num_residues
        self.channels_msa = channels_msa
        self.channels_msa_extra = channels_msa_extra
        self.channels_pair = channels_pair

        self.init = init

        if(init == "binarized"):
            self.msa_dropout_mask = nn.Parameter(
                ((torch.rand(num_residues, channels_msa)) > dropout_rate).float()
            )

            self.pair_dropout_mask = nn.Parameter(
                ((torch.rand(4, num_residues, channels_pair)) > dropout_rate).float()
            )

            self.extra_msa_dropout_mask = nn.Parameter(
                ((torch.rand(num_residues, channels_msa_extra)) > dropout_rate).float()
            )
        
        if(init == "uniform"):
            self.msa_dropout_mask = nn.Parameter(
                torch.rand(num_residues, channels_msa) - 0.5
            )
            self.pair_dropout_mask = nn.Parameter(
                torch.rand(4, num_residues, channels_pair) - 0.5
            )
            self.extra_msa_dropout_mask = nn.Parameter(
                torch.rand(num_residues, channels_msa_extra) - 0.5
            )
        if(init == "ones"):
            self.msa_dropout_mask = nn.Parameter(
                torch.ones(num_residues, channels_msa)
            )
            self.pair_dropout_mask = nn.Parameter(
                torch.ones(4, num_residues, channels_pair)
            )
            self.extra_msa_dropout_mask = nn.Parameter(
                torch.ones(num_residues, channels_msa_extra)
            )
        if(self.init == "halves"):
            self.msa_dropout_mask = nn.Parameter(
                torch.ones(num_residues, channels_msa) * 0.5
            )
            self.pair_dropout_mask = nn.Parameter(
                torch.ones(4, num_residues, channels_pair) * 0.5
            )
            self.extra_msa_dropout_mask = nn.Parameter(
                torch.ones(num_residues, channels_msa_extra) * 0.5
            )   
            
        if(self.init == "zeros"):
            self.msa_dropout_mask = nn.Parameter(
                torch.zeros(self.num_residues, self.channels_msa)
            )
            self.pair_dropout_mask = nn.Parameter(
                torch.zeros(4, self.num_residues, self.channels_pair)
            )
            self.extra_msa_dropout_mask = nn.Parameter(
                torch.zeros(self.num_residues, self.channels_msa_extra)
            )
    def reinitialize(self):
        if(self.init == "binarized"):
            self.msa_dropout_mask = nn.Parameter(
                ((torch.rand(self.num_residues, self.channels_msa)) > self.dropout_rate).float()
            )    
            self.pair_dropout_mask = nn.Parameter(
                ((torch.rand(4, self.num_residues, self.channels_pair)) > self.dropout_rate).float()
            )
            self.extra_msa_dropout_mask = nn.Parameter(
                ((torch.rand(self.num_residues, self.channels_msa_extra)) > self.dropout_rate).float()
            )
        
        if(self.init == "uniform"):
            self.msa_dropout_mask = nn.Parameter(
                torch.rand(self.num_residues, self.channels_msa) - 0.5
            )
            self.pair_dropout_mask = nn.Parameter(
                torch.rand(4, self.num_residues, self.channels_pair) - 0.5
            )
            self.extra_msa_dropout_mask = nn.Parameter(
                torch.rand(self.num_residues, self.channels_msa_extra) - 0.5
            )
        
        if(self.init == "zeros"):
            self.msa_dropout_mask = nn.Parameter(
                torch.zeros(self.num_residues, self.channels_msa)
            )
            self.pair_dropout_mask = nn.Parameter(
                torch.zeros(4, self.num_residues, self.channels_pair)
            )
            self.extra_msa_dropout_mask = nn.Parameter(
                torch.zeros(self.num_residues, self.channels_msa_extra)
            )

    def forward(self):
        return



class FapeHistoryLoss(nn.Module):
    def __init__(self, device, alpha=2):
        self.history = []
        self.device = device
        self.alpha = alpha

    def add_to_history(self, coords, score):
        self.history.append((coords.detach().clone(), score.detach().clone()))

    def compute_loss(self, current_coords):
        if not self.history:
            return torch.tensor(0.0, device=self.device)

        total_loss = torch.tensor(0.0, device=self.device)

        for past_coords, score in self.history:
            # Calculate the distance between the current distogram and the past one
            fape_loss = get_fape_loss(past_coords, current_coords).mean()

            weight = score**2
            weighted_distance = fape_loss * weight

            total_loss += weighted_distance

        # Normalize by the number of models in history
        return -(total_loss / len(self.history)).mean()
