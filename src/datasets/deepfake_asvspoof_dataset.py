import logging
from pathlib import Path

import pandas as pd

from src.datasets.base_dataset import SimpleAudioFakeDataset

DF_ASVSPOOF_SPLIT = {
    "partition_ratio": [0.7, 0.15],
    "seed": 45
}

LOGGER = logging.getLogger()

class DeepFakeASVSpoofDataset(SimpleAudioFakeDataset):

    protocol_file_name = "keys/CM/trial_metadata.txt"
    subset_dir_prefix = "ASVspoof2021_DF_eval"
    subset_parts = ("part00", "part01", "part02", "part03")

    def __init__(self, path, subset="train", transform=None):
        super().__init__(subset, transform)
        self.path = path

        self.partition_ratio = DF_ASVSPOOF_SPLIT["partition_ratio"]
        self.seed = DF_ASVSPOOF_SPLIT["seed"]

        self.flac_paths = self.get_file_references()
        self.samples = self.read_protocol()

        self.transform = transform
        LOGGER.info(f"Spoof: {len(self.samples[self.samples['label'] == 'spoof'])}")
        LOGGER.info(f"Original: {len(self.samples[self.samples['label'] == 'bonafide'])}")

    def get_file_references(self):
        flac_paths = {}
        for part in self.subset_parts:
            path = Path(self.path) / f"{self.subset_dir_prefix}_{part}" / self.subset_dir_prefix / "flac"
            flac_list = list(path.glob("*.flac"))

            for path in flac_list:
                flac_paths[path.stem] = path

        return flac_paths

    def read_protocol(self):
        samples = {
            "sample_name": [],
            "label": [],
            "path": [],
            "attack_type": [],
        }

        real_samples = []
        fake_samples = []
        with open(Path(self.path) / self.protocol_file_name, "r") as file:
            for line in file:
                label = line.strip().split(" ")[5]

                if label == "bonafide":
                    real_samples.append(line)
                elif label == "spoof":
                    fake_samples.append(line)

        fake_samples = self.split_samples(fake_samples)
        for line in fake_samples:
            samples = self.add_line_to_samples(samples, line)

        real_samples = self.split_samples(real_samples)
        for line in real_samples:
            samples = self.add_line_to_samples(samples, line)

        return pd.DataFrame(samples)

    def add_line_to_samples(self, samples, line):
        _, sample_name, _, _, _, label, _, _ = line.strip().split(" ")
        samples["sample_name"].append(sample_name)
        samples["label"].append(label)
        samples["attack_type"].append(label)

        sample_path = self.flac_paths[sample_name]
        assert sample_path.exists()
        samples["path"].append(sample_path)

        return samples

