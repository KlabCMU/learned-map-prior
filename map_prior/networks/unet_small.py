import torch
import torch.nn as nn


class Conv3x3(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_feat,
                out_feat,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ELU(),
            nn.Dropout(p=0.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_feat,
                out_feat,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ELU(),
        )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UpSample(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpSample, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv = nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        outputs = self.up(inputs)
        # outputs = self.deconv(inputs)
        out = torch.cat([outputs, down_outputs], 1)
        return out


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = nn.Sequential(Conv3x3(1, 32))
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            Conv3x3(32, 64),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            Conv3x3(64, 128),
        )
        self.bottom = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),
            Conv3x3(128, 256),
            nn.BatchNorm2d(256),
        )

        self.up1 = UpSample(256,  128)
        self.upconv1 = nn.Sequential(
            Conv3x3(256 + 128, 128),
            nn.BatchNorm2d(128),
        )
        self.up2 = UpSample(128, 64)
        self.upconv2 = nn.Sequential(
            Conv3x3(128 + 64, 64),
            nn.BatchNorm2d(64),
        )
        self.up3 = UpSample(64, 32)
        self.upconv3 = nn.Sequential(
            Conv3x3(64 + 32, 32),
            nn.BatchNorm2d(32),
        )
        self.final = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input, return_features=False):
        down1_feat = self.down1(input)
        down2_feat = self.down2(down1_feat)
        down3_feat = self.down3(down2_feat)
        bottom_feat = self.bottom(down3_feat)

        up1_feat = self.up1(bottom_feat, down3_feat)
        up1_feat = self.upconv1(up1_feat)
        up2_feat = self.up2(up1_feat, down2_feat)
        up2_feat = self.upconv2(up2_feat)
        up3_feat = self.up3(up2_feat, down1_feat)
        up3_feat = self.upconv3(up3_feat)

        if return_features:
            outputs = up3_feat
        else:
            outputs = self.final(up3_feat)

        return outputs
