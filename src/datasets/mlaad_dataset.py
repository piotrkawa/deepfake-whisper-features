from pathlib import Path

import pandas as pd
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from src.datasets.base_dataset import SimpleAudioFakeDataset



class MLADDataset(SimpleAudioFakeDataset):
    def __init__(
        self,
        path: str | Path,
        subset="train",
        supported_languages: list[str] | None = None,
        transform=None
    ):
        super().__init__(subset, transform)

        if supported_languages is None:
            supported_languages = ["en"]

        supported_languages = supported_languages
        self.path = Path(path)
        # mailabs_path = self.path / "real"

        lang_dirs = [x for x in (self.path / "fake").glob("*") if x.is_dir()]
        # dirs

        if supported_languages:
            lang_dirs = [x for x in lang_dirs if x.name in supported_languages]

        # Get MLAAD samples
        all_mlaad_meta = pd.DataFrame()
        for dir in lang_dirs:
            lang_models_dirs = [x for x in dir.rglob("*") if x.is_dir()]

            for model_dir in lang_models_dirs:
                # print(model_dir / "meta.csv")
                meta = pd.read_csv(model_dir / "meta.csv", delimiter="|")
                meta["label"] = "spoof"
                meta["user_id"] = "?"
                meta["sample_name"] = meta["path"].apply(lambda x: Path(x).stem)
                meta["path"] = meta["path"].apply(lambda x: self.path / x)
                all_mlaad_meta = pd.concat([all_mlaad_meta, meta], ignore_index=True)

        all_mlaad_meta.rename({"model_name": "attack_type"}, inplace=True)
        all_mlaad_meta.drop(columns=["duration", "training_data", "is_original_language"], inplace=True)

        # Get corresponding M-AILABS samples
        all_mailabs_meta = all_mlaad_meta.copy()
        all_mailabs_meta["label"] = "bonafide"
        all_mailabs_meta["path"] = all_mailabs_meta["original_file"].apply(lambda x: self.path / "real" / x)

        self.samples = pd.concat([all_mlaad_meta, all_mailabs_meta], ignore_index=True)

        for row in self.samples.itertuples():
            assert Path(row.path).exists(), f"Not found: '{row.path}'"

        self.transform = transform
