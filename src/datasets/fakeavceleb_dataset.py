from pathlib import Path

import pandas as pd

from src.datasets.base_dataset import SimpleAudioFakeDataset

FAKEAVCELEB_SPLIT = {
    "train": ['faceswap-wav2lip', 'fsgan-wav2lip', 'wav2lip', 'rtvc'],
    "test":  ['faceswap-wav2lip', 'fsgan-wav2lip', 'wav2lip', 'rtvc'],
    "val":   ['faceswap-wav2lip', 'fsgan-wav2lip', 'wav2lip', 'rtvc'],
    "partition_ratio": [0.7, 0.15],
    "seed": 45
}


class FakeAVCelebDataset(SimpleAudioFakeDataset):

    audio_folder = "FakeAVCeleb-audio"
    audio_extension = ".mp3"
    metadata_file = Path(audio_folder) / "meta_data.csv"
    subsets = ("train", "dev", "eval")

    def __init__(self, path, subset="train", transform=None):
        super().__init__(subset, transform)
        self.path = path

        self.subset = subset
        self.allowed_attacks = FAKEAVCELEB_SPLIT[subset]
        self.partition_ratio = FAKEAVCELEB_SPLIT["partition_ratio"]
        self.seed = FAKEAVCELEB_SPLIT["seed"]

        self.metadata = self.get_metadata()

        self.samples = pd.concat([self.get_fake_samples(), self.get_real_samples()], ignore_index=True)

    def get_metadata(self):
        md = pd.read_csv(Path(self.path) / self.metadata_file)
        md["audio_type"] = md["type"].apply(lambda x: x.split("-")[-1])
        return md

    def get_fake_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        for attack_name in self.allowed_attacks:
            fake_samples = self.metadata[
                (self.metadata["method"] == attack_name) & (self.metadata["audio_type"] == "FakeAudio")
            ]

            samples_list = fake_samples.iterrows()
            samples_list = self.split_samples(samples_list)

            for _, sample in samples_list:
                samples["user_id"].append(sample["source"])
                samples["sample_name"].append(Path(sample["filename"]).stem)
                samples["attack_type"].append(sample["method"])
                samples["label"].append("spoof")
                samples["path"].append(self.get_file_path(sample))

        return pd.DataFrame(samples)

    def get_real_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        samples_list = self.metadata[
            (self.metadata["method"] == "real") & (self.metadata["audio_type"] == "RealAudio")
        ]

        samples_list = self.split_samples(samples_list)

        for index, sample in samples_list.iterrows():
            samples["user_id"].append(sample["source"])
            samples["sample_name"].append(Path(sample["filename"]).stem)
            samples["attack_type"].append("-")
            samples["label"].append("bonafide")
            samples["path"].append(self.get_file_path(sample))

        return pd.DataFrame(samples)

    def get_file_path(self, sample):
        path = "/".join([self.audio_folder, *sample["path"].split("/")[1:]])
        return Path(self.path) / path / Path(sample["filename"]).with_suffix(self.audio_extension)

