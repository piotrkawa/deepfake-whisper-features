"""
This file contains implementation of RawNet3 architecture.
The original implementation can be found here: https://github.com/Jungjee/RawNet/tree/master/python/RawNet3
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid_filterbanks import Encoder, ParamSincFB  # pip install asteroid_filterbanks


class RawNet3(nn.Module):
    def __init__(self, block, model_scale, context, summed, C=1024, **kwargs):
        super().__init__()

        nOut = kwargs["nOut"]

        self.context = context
        self.encoder_type = kwargs["encoder_type"]
        self.log_sinc = kwargs["log_sinc"]
        self.norm_sinc = kwargs["norm_sinc"]
        self.out_bn = kwargs["out_bn"]
        self.summed = summed

        self.preprocess = nn.Sequential(
            PreEmphasis(), nn.InstanceNorm1d(1, eps=1e-4, affine=True)
        )
        self.conv1 = Encoder(
            ParamSincFB(
                C // 4,
                251,
                stride=kwargs["sinc_stride"],
            )
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C // 4)

        self.layer1 = block(
            C // 4, C, kernel_size=3, dilation=2, scale=model_scale, pool=5
        )
        self.layer2 = block(
            C, C, kernel_size=3, dilation=3, scale=model_scale, pool=3
        )
        self.layer3 = block(C, C, kernel_size=3, dilation=4, scale=model_scale)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)

        if self.context:
            attn_input = 1536 * 3
        else:
            attn_input = 1536
        print("self.encoder_type", self.encoder_type)
        if self.encoder_type == "ECA":
            attn_output = 1536
        elif self.encoder_type == "ASP":
            attn_output = 1
        else:
            raise ValueError("Undefined encoder")

        self.attention = nn.Sequential(
            nn.Conv1d(attn_input, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, attn_output, kernel_size=1),
            nn.Softmax(dim=2),
        )

        self.bn5 = nn.BatchNorm1d(3072)

        self.fc6 = nn.Linear(3072, nOut)
        self.bn6 = nn.BatchNorm1d(nOut)

        self.mp3 = nn.MaxPool1d(3)

    def forward(self, x):
        """
        :param x: input mini-batch (bs, samp)
        """

        with torch.cuda.amp.autocast(enabled=False):
            x = self.preprocess(x)
            x = torch.abs(self.conv1(x))
            if self.log_sinc:
                x = torch.log(x + 1e-6)
            if self.norm_sinc == "mean":
                x = x - torch.mean(x, dim=-1, keepdim=True)
            elif self.norm_sinc == "mean_std":
                m = torch.mean(x, dim=-1, keepdim=True)
                s = torch.std(x, dim=-1, keepdim=True)
                s[s < 0.001] = 0.001
                x = (x - m) / s

        if self.summed:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(self.mp3(x1) + x2)
        else:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)

        x = self.layer4(torch.cat((self.mp3(x1), x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        if self.context:
            global_x = torch.cat(
                (
                    x,
                    torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                    torch.sqrt(
                        torch.var(x, dim=2, keepdim=True).clamp(
                            min=1e-4, max=1e4
                        )
                    ).repeat(1, 1, t),
                ),
                dim=1,
            )
        else:
            global_x = x

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt(
            (torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4)
        )

        x = torch.cat((mu, sg), 1)

        x = self.bn5(x)

        x = self.fc6(x)

        if self.out_bn:
            x = self.bn6(x)

        return x


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97) -> None:
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert (
            len(input.size()) == 2
        ), "The number of dimensions of input tensor must be 2!"
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), "reflect")
        return F.conv1d(input, self.flipped_filter)


class AFMS(nn.Module):
    """
    Alpha-Feature map scaling, added to the output of each residual block[1,2].

    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    [2] AMFS    : https://www.koreascience.or.kr/article/JAKO202029757857763.page
    """

    def __init__(self, nb_dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((nb_dim, 1)))
        self.fc = nn.Linear(nb_dim, nb_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        y = self.sig(self.fc(y)).view(x.size(0), x.size(1), -1)

        x = x + self.alpha
        x = x * y
        return x


class Bottle2neck(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=None,
        dilation=None,
        scale=4,
        pool=False,
    ):

        super().__init__()

        width = int(math.floor(planes / scale))

        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)

        self.nums = scale - 1

        convs = []
        bns = []

        num_pad = math.floor(kernel_size / 2) * dilation

        for i in range(self.nums):
            convs.append(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=num_pad,
                )
            )
            bns.append(nn.BatchNorm1d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)

        self.relu = nn.ReLU()

        self.width = width

        self.mp = nn.MaxPool1d(pool) if pool else False
        self.afms = AFMS(planes)

        if inplanes != planes:  # if change in number of filters
            self.residual = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out += residual
        if self.mp:
            out = self.mp(out)
        out = self.afms(out)

        return out


def prepare_model():
    model = RawNet3(
        Bottle2neck,
        model_scale=8,
        context=True,
        summed=True,
        encoder_type="ECA",
        nOut=1,  # number of slices
        out_bn=False,
        sinc_stride=10,
        log_sinc=True,
        norm_sinc="mean",
        grad_mult=1,
    )
    return model


if __name__ == "__main__":
    model = RawNet3(
        Bottle2neck,
        model_scale=8,
        context=True,
        summed=True,
        encoder_type="ECA",
        nOut=1,  # number of slices
        out_bn=False,
        sinc_stride=10,
        log_sinc=True,
        norm_sinc="mean",
        grad_mult=1,
    )
    gpu = False

    model.eval()
    print("RawNet3 initialised & weights loaded!")

    if torch.cuda.is_available():
        print("Cuda available, conducting inference on GPU")
        model = model.to("cuda")
        gpu = True

    audios = torch.rand(32, 64_600)

    out = model(audios)
    print(out.shape)
