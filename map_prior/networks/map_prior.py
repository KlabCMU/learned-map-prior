import torch
import torch.nn as nn
from .unet_small import UNet as UNet


class MapPrior(nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate
        self.mapnet = UNet()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
        )

    def forward(self, traj, map):
        _, (deep_traj, _) = self.lstm(traj[:, ::self.sample_rate, :])
        deep_traj = deep_traj[0, ...]
        deep_traj = deep_traj / \
            torch.linalg.norm(deep_traj, dim=1, keepdims=True)
        deep_map = self.mapnet(map[:, None, :, :], return_features=True)
        deep_map = deep_map / torch.linalg.norm(deep_map, dim=1, keepdims=True)
        conv_res = torch.einsum('bc,bchw->bhw', deep_traj, deep_map)
        return {
            "convolution result": conv_res,
            "deep map": deep_map,
            "deep traj vector": deep_traj
        }
