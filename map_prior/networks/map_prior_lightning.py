import torch
import torch.nn as nn
import pytorch_lightning as pl

from .unet_small import UNet as UNet


class MapPriorLightning(pl.LightningModule):
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)

    def loss(self, maps, y, wgt=None):
        se = torch.pow(maps - y, 2)

        if wgt is not None:
            se = wgt*se
            assert torch.all(se >= 0)
            assert torch.all(wgt >= 0)
        mse = torch.mean(se)
        return mse

    def forward(self, traj, map):
        _, (deep_traj, _) = self.lstm(traj[:, ::self.sample_rate, :])
        deep_traj = deep_traj[0, ...]
        deep_traj = deep_traj / \
            torch.linalg.norm(deep_traj, dim=1, keepdim=True)
        deep_map = self.mapnet(map[:, None, :, :], return_features=True)
        deep_map = deep_map / torch.linalg.norm(deep_map, dim=1, keepdim=True)
        conv_res = torch.einsum('bc,bchw->bhw', deep_traj, deep_map)
        return {
            "convolution result": conv_res,
            "deep map": deep_map,
            "deep traj vector": deep_traj
        }

    def training_step(self, data):
        input_traj = data["normalized input trajectory"]
        map = data["omap"]
        target_scores = data["ground truth probability target"]

        net_out = self.forward(input_traj, map)
        predicted_scores = net_out['convolution result']

        loss = self.loss(predicted_scores, target_scores, wgt=data['weights'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, data, data_idx):
        input_traj = data["normalized input trajectory"]
        map = data["omap"]
        target_scores = data["ground truth probability target"]

        net_out = self.forward(input_traj, map)
        predicted_scores = net_out['convolution result']

        loss = self.loss(predicted_scores, target_scores, wgt=data['weights'])
        self.log('val_loss', loss)
        return loss
