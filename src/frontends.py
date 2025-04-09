import torch
from torch import Tensor, nn
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2BertModel,
    Wav2Vec2Model,
)
import torchaudio

SAMPLING_RATE = 16_000
WINDOW_LENGTH = 400  # int((25 / 1_000) * SAMPLING_RATE)
HOP_LENGTH = 160  # int((10 / 1_000) * SAMPLING_RATE)

device = "cuda" if torch.cuda.is_available() else "cpu"




# LFCC_FN = torchaudio.transforms.LFCC(
#     sample_rate=SAMPLING_RATE,
#     n_lfcc=128,
#     speckwargs={
#         "n_fft": 512,
#         "win_length": win_length,
#         "hop_length": hop_length,
#     },
# ).to(device)




# def get_frontend(
#     frontends: list[str],
# ) -> torchaudio.transforms.MFCC | torchaudio.transforms.LFCC | Callable:
#     if "mfcc" in frontends:
#         return prepare_mfcc_double_delta
#     elif "lfcc" in frontends:
#         return prepare_lfcc_double_delta
#     raise ValueError(f"{frontends} frontend is not supported!")


# def prepare_lfcc_double_delta(input: Tensor) -> Tensor:
#     if input.ndim < 4:
#         input = input.unsqueeze(1)  # (bs, 1, n_lfcc, frames)
#     x = LFCC_FN(input)
#     delta = delta_fn(x)
#     double_delta = delta_fn(delta)
#     x = torch.cat((x, delta, double_delta), 2)  # -> [bs, 1, 128 * 3, 1500]
#     return x[:, :, :, :3000]  # (bs, n, n_lfcc * 3, frames)


class MFCCDoubleDelta:
    def __init__(
        self, 
        sampling_rate: int = SAMPLING_RATE,
        win_length: int = WINDOW_LENGTH,
        hop_length: int = HOP_LENGTH,
        n_mfcc: int = 128,
        device: str = "cuda",
    ):
        self.n_mfcc = n_mfcc
        self.mfcc_fn =  torchaudio.transforms.MFCC(
            sample_rate=sampling_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 512,
                "win_length": win_length,
                "hop_length": hop_length,
            },
        ).to(device)
        self.delta_fn = torchaudio.transforms.ComputeDeltas(
            win_length=win_length,
            mode="replicate",
        )

    def __call__(self, x: Tensor) -> Tensor:
        if x.ndim < 4:
            x = x.unsqueeze(1)  # (bs, 1, n_lfcc, frames)
        x = self.mfcc_fn(x)
        delta = self.delta_fn(x)
        double_delta = self.delta_fn(delta)
        x = torch.cat((x, delta, double_delta), 2)  # -> [bs, 1, 128 * 3, 1500]
        return x

    @property
    def output_dim(self) -> int:
        return self.n_mfcc * 3


class Wav2Vec2BERT(nn.Module):
    def __init__(self, freeze_encoder: bool = False, device: str = "cuda"):
        super().__init__()
        self.model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        self.preprocessor = AutoFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        self.freeze_encoder = freeze_encoder
        self.device = device

        if self.freeze_encoder:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, x: Tensor, sample_rate: int = 16_000):
        x_pp = self.preprocessor(
            x.cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt"
        )
        x_pp = x_pp.to(self.device)
        # self.model.to(device)
        x = self.model(**x_pp)
        return x.last_hidden_state

    @property
    def output_dim(self) -> int:
        return self.model.config.hidden_size



class Wav2Vec2_XLSR(nn.Module):
    def __init__(self, freeze_encoder: bool = False, device: str = "cuda"):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.preprocessor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53"
        )
        self.freeze_encoder = freeze_encoder
        self.device = device

        if self.freeze_encoder:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, x: Tensor, sample_rate: int = 16_000):
        x_pp = self.preprocessor(
            x.cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt"
        )
        x_pp = x_pp.to(self.device)
        # self.model.to(device)
        x = self.model(**x_pp)
        return x.last_hidden_state


    @property
    def output_dim(self) -> int:
        return self.model.config.hidden_size
