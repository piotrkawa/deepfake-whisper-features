import numpy as np
import pandas as pd
from pathlib import Path

from src.datasets.base_dataset import SimpleAudioFakeDataset


class InTheWildDataset(SimpleAudioFakeDataset):

    def __init__(
        self,
        path,
        subset="train",
        transform=None,
        seed=None,
        partition_ratio=(0.7, 0.15),
        split_strategy="random"
    ):
        super().__init__(subset=subset, transform=transform)
        self.path = path
        self.read_samples()
        self.partition_ratio = partition_ratio
        self.seed = seed


    def read_samples(self):
        path = Path(self.path)
        meta_path = path / "meta.csv"

        self.samples = pd.read_csv(meta_path)
        self.samples["path"] = self.samples["file"].apply(lambda n: str(path / n))
        self.samples["file"] = self.samples["file"].apply(lambda n: Path(n).stem)
        self.samples["label"] = self.samples["label"].map({"bona-fide": "bonafide", "spoof": "spoof"})
        self.samples["attack_type"] = self.samples["label"].map({"bonafide": "-", "spoof": "X"})
        self.samples.rename(columns={'file': 'sample_name', 'speaker': 'user_id'}, inplace=True)


    def split_samples_per_speaker(self, samples):
        speaker_list = pd.Series(samples["user_id"].unique())
        speaker_list = speaker_list.sort_values()
        speaker_list = speaker_list.sample(frac=1, random_state=self.seed)
        speaker_list = list(speaker_list)

        p, s = self.partition_ratio
        subsets = np.split(speaker_list, [int(p * len(speaker_list)), int((p + s) * len(speaker_list))])
        speaker_subset = dict(zip(['train', 'test', 'val'], subsets))[self.subset]
        return self.samples[self.samples["user_id"].isin(speaker_subset)]


if __name__ == "__main__":
    dataset = InTheWildDataset(
        path="../datasets/release_in_the_wild",
        subset="val",
        seed=242,
        split_strategy="per_speaker"
    )

    print(len(dataset))
    print(len(dataset.samples["user_id"].unique()))
    print(dataset.samples["user_id"].unique())

    print(dataset[0])
