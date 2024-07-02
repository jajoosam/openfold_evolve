import torch
import torch.nn as nn

from utils import get_fape_loss

class DropoutMaskModule(nn.Module):
    def __init__(self, num_residues, channels_msa, channels_msa_extra, channels_pair):
        super(DropoutMaskModule, self).__init__()


        self.msa_dropout_mask = nn.Parameter(
            ((torch.rand(num_residues, channels_msa)) > 0.15).float()
        )

        self.pair_dropout_mask = nn.Parameter(
            ((torch.rand(4, num_residues, channels_pair)) > 0.15).float()
        )

        self.extra_msa_dropout_mask = nn.Parameter(
            ((torch.rand(num_residues, channels_msa_extra)) > 0.15).float()
        )

    def forward(self):
        return



class FapeHistoryLoss(nn.Module):
    def __init__(self, device, alpha=2):
        self.history = []
        self.device = device
        self.alpha = alpha

    def add_to_history(self, coords, tm_score):
        self.history.append((coords.detach().clone(), tm_score.detach().clone()))

    def compute_loss(self, current_coords):
        if not self.history:
            return torch.tensor(0.0, device=self.device)

        total_loss = torch.tensor(0.0, device=self.device)

        for past_coords, tm_score in self.history:
            # Calculate the distance between the current distogram and the past one
            fape_loss = get_fape_loss(past_coords, current_coords).mean()

            weight = torch.exp(self.alpha * (1 - tm_score)) - 1
            weighted_distance = fape_loss * weight

            total_loss += weighted_distance

        # Normalize by the number of models in history
        return -(total_loss / len(self.history)).mean()
