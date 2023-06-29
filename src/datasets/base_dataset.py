"""Base dataset classes."""
import logging
import math
import random

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


LOGGER = logging.getLogger(__name__)

SAMPLING_RATE = 16_000
APPLY_NORMALIZATION = True
APPLY_TRIMMING = True
APPLY_PADDING = True
FRAMES_NUMBER = 480_000  # <- originally 64_600


SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]


class SimpleAudioFakeDataset(Dataset):
    def __init__(
        self,
        subset,
        transform=None,
        return_label: bool = True,
        return_meta: bool = False,
    ):
        self.transform = transform
        self.samples = pd.DataFrame()

        self.subset = subset
        self.allowed_attacks = None
        self.partition_ratio = None
        self.seed = None
        self.return_label = return_label
        self.return_meta = return_meta

    def split_samples(self, samples_list):
        if isinstance(samples_list, pd.DataFrame):
            samples_list = samples_list.sort_values(by=list(samples_list.columns))
            samples_list = samples_list.sample(frac=1, random_state=self.seed)
        else:
            samples_list = sorted(samples_list)
            random.seed(self.seed)
            random.shuffle(samples_list)

        p, s = self.partition_ratio
        subsets = np.split(
            samples_list, [int(p * len(samples_list)), int((p + s) * len(samples_list))]
        )
        return dict(zip(["train", "test", "val"], subsets))[self.subset]

    def df2tuples(self):
        tuple_samples = []
        for i, elem in self.samples.iterrows():
            tuple_samples.append(
                (str(elem["path"]), elem["label"], elem["attack_type"])
            )

        self.samples = tuple_samples
        return self.samples

    def __getitem__(self, index) -> T_co:
        if isinstance(self.samples, pd.DataFrame):
            sample = self.samples.iloc[index]

            path = str(sample["path"])
            label = sample["label"]
            attack_type = sample["attack_type"]
            if type(attack_type) != str and math.isnan(attack_type):
                attack_type = "N/A"
        else:
            path, label, attack_type = self.samples[index]

        waveform, sample_rate = torchaudio.load(path, normalize=APPLY_NORMALIZATION)
        real_sec_length = len(waveform[0]) / sample_rate

        waveform, sample_rate = apply_preprocessing(waveform, sample_rate)

        return_data = [waveform, sample_rate]
        if self.return_label:
            label = 1 if label == "bonafide" else 0
            return_data.append(label)

        if self.return_meta:
            return_data.append(
                (
                    attack_type,
                    path,
                    self.subset,
                    real_sec_length,
                )
            )
        return return_data

    def __len__(self):
        return len(self.samples)


def apply_preprocessing(
    waveform,
    sample_rate,
):
    if sample_rate != SAMPLING_RATE and SAMPLING_RATE != -1:
        waveform, sample_rate = resample_wave(waveform, sample_rate, SAMPLING_RATE)

    # Stereo to mono
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform[:1, ...]

    # Trim too long utterances...
    if APPLY_TRIMMING:
        waveform, sample_rate = apply_trim(waveform, sample_rate)

    # ... or pad too short ones.
    if APPLY_PADDING:
        waveform = apply_pad(waveform, FRAMES_NUMBER)

    return waveform, sample_rate


def resample_wave(waveform, sample_rate, target_sample_rate):
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, [["rate", f"{target_sample_rate}"]]
    )
    return waveform, sample_rate


def resample_file(path, target_sample_rate, normalize=True):
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
        path, [["rate", f"{target_sample_rate}"]], normalize=normalize
    )

    return waveform, sample_rate


def apply_trim(waveform, sample_rate):
    (
        waveform_trimmed,
        sample_rate_trimmed,
    ) = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, SOX_SILENCE)

    if waveform_trimmed.size()[1] > 0:
        waveform = waveform_trimmed
        sample_rate = sample_rate_trimmed

    return waveform, sample_rate


def apply_pad(waveform, cut):
    """Pad wave by repeating signal until `cut` length is achieved."""
    waveform = waveform.squeeze(0)
    waveform_len = waveform.shape[0]

    if waveform_len >= cut:
        return waveform[:cut]

    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]

    return padded_waveform
