from pathlib import Path

import pandas as pd
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from src.datasets.base_dataset import SimpleAudioFakeDataset

ASVSPOOF_SPLIT = {
    "train": ['A01', 'A07', 'A08', 'A02', 'A09', 'A10', 'A03', 'A04', 'A05', 'A06', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19'],
    "test":  ['A01', 'A07', 'A08', 'A02', 'A09', 'A10', 'A03', 'A04', 'A05', 'A06', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19'],
    "val":   ['A01', 'A07', 'A08', 'A02', 'A09', 'A10', 'A03', 'A04', 'A05', 'A06', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19'],
    "partition_ratio": [0.7, 0.15],
    "seed": 45,
}


class ASVSpoofDataset(SimpleAudioFakeDataset):

    protocol_folder_name = "ASVspoof2019_LA_cm_protocols"
    subset_dir_prefix = "ASVspoof2019_LA_"
    subsets = ("train", "dev", "eval")

    def __init__(self, path, subset="train", transform=None):
        super().__init__(subset, transform)
        self.path = path

        self.allowed_attacks = ASVSPOOF_SPLIT[subset]
        self.partition_ratio = ASVSPOOF_SPLIT["partition_ratio"]
        self.seed = ASVSPOOF_SPLIT["seed"]

        self.samples = pd.DataFrame()

        for subset in self.subsets:
            subset_dir = Path(self.path) / f"{self.subset_dir_prefix}{subset}"
            subset_protocol_path = self.get_protocol_path(subset)
            subset_samples = self.read_protocol(subset_dir, subset_protocol_path)

            self.samples = pd.concat([self.samples, subset_samples])

        self.transform = transform

    def get_protocol_path(self, subset):
        paths = list((Path(self.path) / self.protocol_folder_name).glob("*.txt"))
        for path in paths:
            if subset in Path(path).stem:
                return path

    def read_protocol(self, subset_dir, protocol_path):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        real_samples = []
        fake_samples = []
        with open(protocol_path, "r") as file:
            for line in file:
                attack_type = line.strip().split(" ")[3]

                if attack_type == "-":
                    real_samples.append(line)
                elif attack_type in self.allowed_attacks:
                    fake_samples.append(line)

                if attack_type not in self.allowed_attacks:
                    continue

        fake_samples = self.split_samples(fake_samples)
        for line in fake_samples:
            samples = self.add_line_to_samples(samples, line, subset_dir)

        real_samples = self.split_samples(real_samples)
        for line in real_samples:
            samples = self.add_line_to_samples(samples, line, subset_dir)

        return pd.DataFrame(samples)

    @staticmethod
    def add_line_to_samples(samples, line, subset_dir):
        user_id, sample_name, _, attack_type, label = line.strip().split(" ")
        samples["user_id"].append(user_id)
        samples["sample_name"].append(sample_name)
        samples["attack_type"].append(attack_type)
        samples["label"].append(label)

        assert (subset_dir / "flac" / f"{sample_name}.flac").exists()
        samples["path"].append(subset_dir / "flac" / f"{sample_name}.flac")

        return samples

class ASVSpoof2019DatasetOriginal(ASVSpoofDataset):

    subsets = {"train": "train", "test": "dev", "val": "eval"}

    protocol_folder_name = "ASVspoof2019_LA_cm_protocols"
    subset_dir_prefix = "ASVspoof2019_LA_"
    subset_dirs_attacks = {
        "train": ["A01", "A02", "A03", "A04", "A05", "A06"],
        "dev":  ["A01", "A02", "A03", "A04", "A05", "A06"],
        "eval": [
            "A07", "A08", "A09", "A10", "A11",  "A12", "A13", "A14", "A15",
            "A16", "A17", "A18", "A19"
        ]
    }


    def __init__(self, path, fold_subset="train"):
        """
        Initialise object. Skip __init__ of ASVSpoofDataset doe to different 
        logic, but follow SimpleAudioFakeDataset constructor.
        """
        super(ASVSpoofDataset, self).__init__(float('inf'), fold_subset)
        self.path = path
        subset = self.subsets[fold_subset]
        self.allowed_attacks = self.subset_dirs_attacks[subset]
        subset_dir = Path(self.path) / f"{self.subset_dir_prefix}{subset}"
        subset_protocol_path = self.get_protocol_path(subset)
        self.samples = self.read_protocol(subset_dir, subset_protocol_path)

    def read_protocol(self, subset_dir, protocol_path):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        real_samples = []
        fake_samples = []

        with open(protocol_path, "r") as file:
            for line in file:
                attack_type = line.strip().split(" ")[3]
                if attack_type == "-":
                    real_samples.append(line)
                elif attack_type in self.allowed_attacks:
                    fake_samples.append(line)
                else:
                    raise ValueError(
                        "Tried to load attack that shouldn't be here!"
                    )

        for line in fake_samples:
            samples = self.add_line_to_samples(samples, line, subset_dir)
        for line in real_samples:
            samples = self.add_line_to_samples(samples, line, subset_dir)

        return pd.DataFrame(samples)

