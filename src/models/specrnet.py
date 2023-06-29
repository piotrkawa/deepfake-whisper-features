"""
This file contains implementation of SpecRNet architecture.
We base our codebase on the implementation of RawNet2 by Hemlata Tak (tak@eurecom.fr).
It is available here: https://github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-RawNet2/model.py
"""
from typing import Dict

import torch.nn as nn

from src import frontends


def get_config(input_channels: int) -> Dict:
    return {
        "filts": [input_channels, [input_channels, 20], [20, 64], [64, 64]],
        "nb_fc_node": 64,
        "gru_node": 64,
        "nb_gru_layer": 2,
        "nb_classes": 1,
    }


class Residual_block2D(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            padding=1,
            kernel_size=3,
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=0,
                kernel_size=1,
                stride=1,
            )

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d(2)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


class SpecRNet(nn.Module):
    def __init__(self, input_channels, **kwargs):
        super().__init__()
        config = get_config(input_channels=input_channels)

        self.device = kwargs.get("device", "cuda")

        self.first_bn = nn.BatchNorm2d(num_features=config["filts"][0])
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(
            Residual_block2D(nb_filts=config["filts"][1], first=True)
        )
        self.block2 = nn.Sequential(Residual_block2D(nb_filts=config["filts"][2]))
        config["filts"][2][0] = config["filts"][2][1]
        self.block4 = nn.Sequential(Residual_block2D(nb_filts=config["filts"][2]))
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc_attention0 = self._make_attention_fc(
            in_features=config["filts"][1][-1], l_out_features=config["filts"][1][-1]
        )
        self.fc_attention2 = self._make_attention_fc(
            in_features=config["filts"][2][-1], l_out_features=config["filts"][2][-1]
        )
        self.fc_attention4 = self._make_attention_fc(
            in_features=config["filts"][2][-1], l_out_features=config["filts"][2][-1]
        )

        self.bn_before_gru = nn.BatchNorm2d(num_features=config["filts"][2][-1])
        self.gru = nn.GRU(
            input_size=config["filts"][2][-1],
            hidden_size=config["gru_node"],
            num_layers=config["nb_gru_layer"],
            batch_first=True,
            bidirectional=True,
        )

        self.fc1_gru = nn.Linear(
            in_features=config["gru_node"] * 2, out_features=config["nb_fc_node"] * 2
        )

        self.fc2_gru = nn.Linear(
            in_features=config["nb_fc_node"] * 2,
            out_features=config["nb_classes"],
            bias=True,
        )

        self.sig = nn.Sigmoid()

    def _compute_embedding(self, x):
        x = self.first_bn(x)
        x = self.selu(x)

        x0 = self.block0(x)
        y0 = self.avgpool(x0).view(x0.size(0), -1)
        y0 = self.fc_attention0(y0)
        y0 = self.sig(y0).view(y0.size(0), y0.size(1), -1)
        y0 = y0.unsqueeze(-1)
        x = x0 * y0 + y0

        x = nn.MaxPool2d(2)(x)

        x2 = self.block2(x)
        y2 = self.avgpool(x2).view(x2.size(0), -1)
        y2 = self.fc_attention2(y2)
        y2 = self.sig(y2).view(y2.size(0), y2.size(1), -1)
        y2 = y2.unsqueeze(-1)
        x = x2 * y2 + y2

        x = nn.MaxPool2d(2)(x)

        x4 = self.block4(x)
        y4 = self.avgpool(x4).view(x4.size(0), -1)
        y4 = self.fc_attention4(y4)
        y4 = self.sig(y4).view(y4.size(0), y4.size(1), -1)
        y4 = y4.unsqueeze(-1)
        x = x4 * y4 + y4

        x = nn.MaxPool2d(2)(x)

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = nn.AdaptiveAvgPool2d((1, None))(x)
        x = x.squeeze(-2)
        x = x.permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1_gru(x)
        x = self.fc2_gru(x)
        return x

    def forward(self, x):
        x = self._compute_embedding(x)
        return x

    def _make_attention_fc(self, in_features, l_out_features):
        l_fc = []
        l_fc.append(nn.Linear(in_features=in_features, out_features=l_out_features))
        return nn.Sequential(*l_fc)


class FrontendSpecRNet(SpecRNet):
    def __init__(self, input_channels, **kwargs):
        super().__init__(input_channels, **kwargs)

        self.device = kwargs['device']

        frontend_name = kwargs.get("frontend_algorithm", [])
        self.frontend = frontends.get_frontend(frontend_name)
        print(f"Using {frontend_name} frontend")

    def _compute_frontend(self, x):
        frontend = self.frontend(x)
        if frontend.ndim < 4:
            return frontend.unsqueeze(1)  # (bs, 1, n_lfcc, frames)
        return frontend # (bs, n, n_lfcc, frames)

    def forward(self, x):
        x = self._compute_frontend(x)
        x = self._compute_embedding(x)
        return x


if __name__ == "__main__":
    print("Definition of model")
    device = "cuda"

    input_channels = 1
    config = {
        "filts": [input_channels, [input_channels, 20], [20, 64], [64, 64]],
        "nb_fc_node": 64,
        "gru_node": 64,
        "nb_gru_layer": 2,
        "nb_classes": 1,
    }

    def count_parameters(model) -> int:
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return pytorch_total_params
    model = FrontendSpecRNet(input_channels=1, device=device, frontend_algorithm=["lfcc"])
    model = model.to(device)
    print(count_parameters(model))
