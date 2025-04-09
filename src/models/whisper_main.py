from collections import OrderedDict
from pathlib import Path
# Based on https://github.com/openai/whisper/blob/main/whisper/model.py
from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Iterable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


def exact_div(x, y):
    assert x % y == 0
    return x // y


# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
N_FRAMES = exact_div(
    N_SAMPLES, HOP_LENGTH
)  # 3000: number of frames in a mel spectrogram input

WHISPER_WEIGHTS_DIR = Path("assets/whisper_weights")

if not WHISPER_WEIGHTS_DIR.exists():
    import whisper
    from collections import OrderedDict

    def extract_and_save_encoder(size):
        model = whisper.load_model(size)
        model_ckpt = OrderedDict()

        model_ckpt['model_state_dict'] = OrderedDict()

        for key, value in model.encoder.state_dict().items():
            model_ckpt['model_state_dict'][f'encoder.{key}'] = value

        model_ckpt['dims'] = model.dims
        torch.save(model_ckpt, WHISPER_WEIGHTS_DIR / f"{size}.pt")

    model_sizes = [
        "tiny",
        "small",
        "base",
        "large",
        "turbo"
    ]
    for size in model_sizes:
        extract_and_save_encoder(size)


def pad_or_trim(
    array: Union[torch.Tensor, np.ndarray],
    length: int = N_SAMPLES,
    *,
    axis: int = -1,
) -> torch.Tensor:
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if not torch.is_tensor(array):
        array = torch.from_numpy(array)

    if array.shape[axis] > length:
        array = array.index_select(
            dim=axis, index=torch.arange(length, device=array.device)
        )

    if array.shape[axis] < length:
        # pad multiple times
        num_repeats = int(length / array.shape[axis]) + 1
        array = torch.tile(array, (1, num_repeats))[:, :length]
    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(
        os.path.join(os.path.dirname(__file__), "assets/mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(audio: torch.Tensor, n_mels: int = N_MELS):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[:, :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10_000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv)

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)
        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )

    def forward(self, mel: torch.Tensor):
        return self.encoder(mel)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def output_dim(self):
        return self.dims.n_audio_state


def get_whisper_dims(model_size: str) -> ModelDimensions:
    # TODO: proper dims
    match model_size:
        case "tiny":
            return ModelDimensions(
                n_mels=80,
                n_audio_ctx=1500,
                n_audio_state=384,
                n_audio_head=6,
                n_audio_layer=4, 
                n_vocab=51865, 
                n_text_ctx=448, 
                n_text_state=384,
                n_text_head=6, 
                n_text_layer=4
            )
        case "small":
            return ModelDimensions(
                n_mels=80,
                n_audio_ctx=1500,
                n_audio_state=768,
                n_audio_head=12,
                n_audio_layer=12,
                n_vocab=51865,
                n_text_ctx=448,
                n_text_state=768,
                n_text_head=12,
                n_text_layer=12
            )
        case "base":
            return ModelDimensions(
                n_mels=80,
                n_audio_ctx=1500,
                n_audio_state=512,
                n_audio_head=8, 
                n_audio_layer=6,
                n_vocab=51865,
                n_text_ctx=448,  
                n_text_state=512,
                n_text_head=8,
                n_text_layer=6
            )
        case "large":
            return ModelDimensions(
                n_mels=128,
                n_audio_ctx=1500,
                n_audio_state=1280,
                n_audio_head=20,
                n_audio_layer=32,
                n_vocab=51866,
                n_text_ctx=448,
                n_text_state=1280,
                n_text_head=20,
                n_text_layer=32
            )
        case "turbo":
            return ModelDimensions(
                n_mels=128,
                n_audio_ctx=1500,
                n_audio_state=1280,
                n_audio_head=20,
                n_audio_layer=32,
                n_vocab=51866,
                n_text_ctx=448,
                n_text_state=1280,
                n_text_head=20,
                n_text_layer=4
            )
        case _:
            raise ValueError(f"Unsupported model size: {model_size}")


def get_whisper_model(model_size: str = "tiny"):
    dims = get_whisper_dims(model_size)
    model = Whisper(dims)
    state_dict = torch.load(WHISPER_WEIGHTS_DIR / f"{model_size}.pt")
    model.load_state_dict(state_dict['model_state_dict'])
    return model
