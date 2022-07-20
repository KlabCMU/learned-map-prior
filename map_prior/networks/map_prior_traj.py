import torch
import torch.nn as nn


class MapPriorTraj(nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )

    def forward(self, traj, deep_map):
        _, (deep_traj, _) = self.lstm(traj[:, ::self.sample_rate, :])
        deep_traj = deep_traj[0, ...]
        deep_traj = deep_traj / \
            torch.linalg.norm(deep_traj, dim=1, keepdim=True)
        deep_traj = deep_traj.squeeze(0)
        conv_res = torch.einsum('c,chw->hw', deep_traj, deep_map)
        return conv_res
