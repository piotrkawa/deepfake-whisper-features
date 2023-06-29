from typing import List, Union, Callable

import torch
import torchaudio

SAMPLING_RATE = 16_000
win_length = 400  # int((25 / 1_000) * SAMPLING_RATE)
hop_length = 160  # int((10 / 1_000) * SAMPLING_RATE)

device = "cuda" if torch.cuda.is_available() else "cpu"

MFCC_FN = torchaudio.transforms.MFCC(
    sample_rate=SAMPLING_RATE,
    n_mfcc=128,
    melkwargs={
        "n_fft": 512,
        "win_length": win_length,
        "hop_length": hop_length,
    },
).to(device)


LFCC_FN = torchaudio.transforms.LFCC(
    sample_rate=SAMPLING_RATE,
    n_lfcc=128,
    speckwargs={
        "n_fft": 512,
        "win_length": win_length,
        "hop_length": hop_length,
    },
).to(device)

MEL_SCALE_FN = torchaudio.transforms.MelScale(
    n_mels=80,
    n_stft=257,
    sample_rate=SAMPLING_RATE,
).to(device)

delta_fn = torchaudio.transforms.ComputeDeltas(
    win_length=400,
    mode="replicate",
)


def get_frontend(
    frontends: List[str],
) -> Union[torchaudio.transforms.MFCC, torchaudio.transforms.LFCC, Callable,]:
    if "mfcc" in frontends:
        return prepare_mfcc_double_delta
    elif "lfcc" in frontends:
        return prepare_lfcc_double_delta
    raise ValueError(f"{frontends} frontend is not supported!")


def prepare_lfcc_double_delta(input):
    if input.ndim < 4:
        input = input.unsqueeze(1)  # (bs, 1, n_lfcc, frames)
    x = LFCC_FN(input)
    delta = delta_fn(x)
    double_delta = delta_fn(delta)
    x = torch.cat((x, delta, double_delta), 2)  # -> [bs, 1, 128 * 3, 1500]
    return x[:, :, :, :3000]  # (bs, n, n_lfcc * 3, frames)


def prepare_mfcc_double_delta(input):
    if input.ndim < 4:
        input = input.unsqueeze(1)  # (bs, 1, n_lfcc, frames)
    x = MFCC_FN(input)
    delta = delta_fn(x)
    double_delta = delta_fn(delta)
    x = torch.cat((x, delta, double_delta), 2)  # -> [bs, 1, 128 * 3, 1500]
    return x[:, :, :, :3000]  # (bs, n, n_lfcc * 3, frames)
