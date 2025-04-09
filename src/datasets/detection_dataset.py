import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.datasets.base_dataset import SimpleAudioFakeDataset
from src.datasets.deepfake_asvspoof_dataset import DeepFakeASVSpoofDataset
from src.datasets.fakeavceleb_dataset import FakeAVCelebDataset
from src.datasets.wavefake_dataset import WaveFakeDataset
from src.datasets.mlaad_dataset import MLADDataset
from src.datasets.asvspoof_dataset import ASVSpoof2019DatasetOriginal


LOGGER = logging.getLogger()


class DetectionDataset(SimpleAudioFakeDataset):
    def __init__(
        self,
        dataset_map: dict[str, Path],
        subset: str = "val",
        transform=None,
        oversample: bool = True,
        undersample: bool = False,
        return_label: bool = True,
        reduced_number: Optional[int] = None,
        return_meta: bool = False,
    ):
        super().__init__(
            subset=subset,
            transform=transform,
            return_label=return_label,
            return_meta=return_meta,
        )
        datasets = self._init_datasets(
            dataset_map=dataset_map,
            subset=subset,
        )
        self.samples = pd.concat([ds.samples for ds in datasets], ignore_index=True)

        if oversample:
            self.oversample_dataset()
        elif undersample:
            self.undersample_dataset()

        if reduced_number:
            LOGGER.info(f"Using reduced number of samples - {reduced_number}!")
            self.samples = self.samples.sample(
                min(len(self.samples), reduced_number),
                random_state=42,
            )

    def _init_datasets(
        self,
        dataset_map: dict[str, Path],
        subset: str,
    ) -> List[SimpleAudioFakeDataset]:

        datasets = []

        for name, path in dataset_map.items():
            match name:
                case "wavefake":
                    ds = WaveFakeDataset(path, subset=subset)
                    datasets.append(ds)

                case "fakeavceleb":
                    ds = FakeAVCelebDataset(path, subset=subset)
                    datasets.append(ds)

                case "asvspoof_2019":
                    ds = ASVSpoof2019DatasetOriginal(path, fold_subset=subset)
                    datasets.append(ds)

                case "asvspoof_2021_df":
                    ds = DeepFakeASVSpoofDataset(path, subset=subset)
                    datasets.append(ds)

                case "mlaad":
                    ds = MLADDataset(path, subset=subset)
                    datasets.append(ds)
                case _:
                    raise ValueError(f"Dataset {name} not supported!")
        return datasets

    def oversample_dataset(self):
        samples = self.samples.groupby(by=["label"])
        bona_length = len(samples.groups["bonafide"])
        spoof_length = len(samples.groups["spoof"])

        diff_length = spoof_length - bona_length

        if diff_length < 0:
            raise NotImplementedError

        if diff_length > 0:
            bonafide = samples.get_group("bonafide").sample(diff_length, replace=True)
            self.samples = pd.concat([self.samples, bonafide], ignore_index=True)

    def undersample_dataset(self):
        samples = self.samples.groupby(by=["label"])
        bona_length = len(samples.groups["bonafide"])
        spoof_length = len(samples.groups["spoof"])

        if spoof_length < bona_length:
            raise NotImplementedError

        if spoof_length > bona_length:
            spoofs = samples.get_group("spoof").sample(bona_length, replace=True)
            self.samples = pd.concat(
                [samples.get_group("bonafide"), spoofs], ignore_index=True
            )

    def get_bonafide_only(self):
        samples = self.samples.groupby(by=["label"])
        self.samples = samples.get_group("bonafide")
        return self.samples

    def get_spoof_only(self):
        samples = self.samples.groupby(by=["label"])
        self.samples = samples.get_group("spoof")
        return self.samples
