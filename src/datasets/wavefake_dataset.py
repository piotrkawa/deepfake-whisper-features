from pathlib import Path

import pandas as pd

from src.datasets.base_dataset import SimpleAudioFakeDataset

WAVEFAKE_SPLIT = {
    "train": ['multi_band_melgan', 'melgan_large', 'parallel_wavegan', 'waveglow', 'full_band_melgan', 'melgan', 'hifiGAN'],
    "test":  ['multi_band_melgan', 'melgan_large', 'parallel_wavegan', 'waveglow', 'full_band_melgan', 'melgan', 'hifiGAN'],
    "val":   ['multi_band_melgan', 'melgan_large', 'parallel_wavegan', 'waveglow', 'full_band_melgan', 'melgan', 'hifiGAN'],
    "partition_ratio": [0.7, 0.15],
    "seed": 45
}


class WaveFakeDataset(SimpleAudioFakeDataset):

    fake_data_path = "generated_audio"
    jsut_real_data_path = "real_audio/jsut_ver1.1/basic5000/wav"
    ljspeech_real_data_path = "real_audio/LJSpeech-1.1/wavs"

    def __init__(self, path, subset="train", transform=None):
        super().__init__(subset, transform)
        self.path = Path(path)

        self.fold_subset = subset
        self.allowed_attacks = WAVEFAKE_SPLIT[subset]
        self.partition_ratio = WAVEFAKE_SPLIT["partition_ratio"]
        self.seed = WAVEFAKE_SPLIT["seed"]

        self.samples = pd.concat([self.get_fake_samples(), self.get_real_samples()], ignore_index=True)

    def get_fake_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        samples_list = list((self.path / self.fake_data_path).glob("*/*.wav"))
        samples_list = self.filter_samples_by_attack(samples_list)
        samples_list = self.split_samples(samples_list)

        for sample in samples_list:
            samples["user_id"].append(None)
            samples["sample_name"].append("_".join(sample.stem.split("_")[:-1]))
            samples["attack_type"].append(self.get_attack_from_path(sample))
            samples["label"].append("spoof")
            samples["path"].append(sample)

        return pd.DataFrame(samples)

    def filter_samples_by_attack(self, samples_list):
        return [s for s in samples_list if self.get_attack_from_path(s) in self.allowed_attacks]

    def get_real_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        samples_list = list((self.path / self.jsut_real_data_path).glob("*.wav"))
        samples_list += list((self.path / self.ljspeech_real_data_path).glob("*.wav"))
        samples_list = self.split_samples(samples_list)

        for sample in samples_list:
            samples["user_id"].append(None)
            samples["sample_name"].append(sample.stem)
            samples["attack_type"].append("-")
            samples["label"].append("bonafide")
            samples["path"].append(sample)

        return pd.DataFrame(samples)

    @staticmethod
    def get_attack_from_path(path):
        folder_name = path.parents[0].relative_to(path.parents[1])
        return str(folder_name).split("_", maxsplit=1)[-1]


