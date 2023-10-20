# Improved DeepFake Detection Using Whisper Features

The following repository contains code for our paper called "Improved DeepFake Detection Using Whisper Features".

The paper is available [here](https://www.isca-speech.org/archive/interspeech_2023/kawa23b_interspeech.html).


## Before you start

### Whisper
To download Whisper encoder used in training run `download_whisper.py`.

### Datasets

Download appropriate datasets:
* [ASVspoof2021 DF subset](https://zenodo.org/record/4835108) (**Please note:** we use [this keys&metadata file](https://www.asvspoof.org/resources/DF-keys-stage-1.tar.gz)),
* [In-The-Wild dataset](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild).



### Dependencies
Install required dependencies using (we assume you're using conda and the target env is active):
```bash
bash install.sh
```

List of requirements:
```
python=3.8
pytorch==1.11.0
torchaudio==0.11
asteroid-filterbanks==0.4.0
librosa==0.9.2
openai whisper (git+https://github.com/openai/whisper.git@7858aa9c08d98f75575035ecd6481f462d66ca27)
```

### Supported models

The following list concerns models and its names to select it supported by this repository:
* SpecRNet - `specrnet`,
* (Whisper) SpecRNet - `whisper_specrnet`,
* (Whisper + LFCC/MFCC) SpecRNet - `whisper_frontend_specrnet`,
* LCNN - `lcnn`,
* (Whisper) LCNN - `whisper_lcnn`,
* (Whisper + LFCC/MFCC) LCNN -`whisper_frontend_lcnn`,
* MesoNet - `mesonet`,
* (Whisper) MesoNet - `whisper_mesonet`,
* (Whisper + LFCC/MFCC) MesoNet - `whisper_frontend_mesonet`,
* RawNet3 - `rawnet3`.

To select appropriate front-end please specify it in the config file.

### Pretrained models

All models reported in paper are available [here](https://drive.google.com/drive/folders/1YWMC64MW4HjGUX1fnBaMkMIGgAJde9Ch?usp=sharing).

### Configs

Both training and evaluation scripts are configured with the use of CLI and `.yaml` configuration files.
e.g.:
```yaml
data:
  seed: 42

checkpoint: 
  path: "trained_models/lcnn/ckpt.pth",

model:
  name: "lcnn"
  parameters:
    input_channels: 1
    frontend_algorithm: ["lfcc"]
  optimizer:
    lr: 0.0001
    weight_decay: 0.0001
```

Other example configs are available under `configs/training/`.

## Full train and test pipeline 

To perform full pipeline of training and testing please use `train_and_test.py` script.

```
usage: train_models.py [-h] [--asv_path ASV_PATH] [--in_the_wild_path IN_THE_WILD_PATH] [--config CONFIG] [--train_amount TRAIN_AMOUNT] [--test_amount TEST_AMOUNT] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--ckpt CKPT] [--cpu]

Arguments: 
    --asv_path          Path to the ASVSpoof2021 DF root dir
    --in_the_wild_path  Path to the In-The-Wild root dir
    --config            Path to the config file
    --train_amount      Number of samples to train on (default: 100000)
    --valid_amount      Number of samples to validate on (default: 25000)
    --test_amount       Number of samples to test on (default: None - all)
    --batch_size        Batch size (default: 8)
    --epochs            Number of epochs (default: 10)
    --ckpt              Path to saved models (default: 'trained_models')
    --cpu               Force using CPU
```

e.g.:
```bash
python train_models.py --asv_path ../datasets/deep_fakes/ASVspoof2021/DF --in_the_wild_path ../datasets/release_in_the_wild --config configs/training/whisper_specrnet.yaml --batch_size 8 --epochs 10 --train_amount 100000 --test_amount 25000
```


## Finetune and test pipeline

To perform finetuning as presented in paper please use `train_and_test.py` script.

e.g.:
```
python train_and_test.py --asv_path ../datasets/deep_fakes/ASVspoof2021/DF --in_the_wild_path ../datasets/release_in_the_wild --config configs/finetuning/whisper_specrnet.yaml --batch_size 8 --epochs 5  --train_amount 100000 --valid_amount 25000
```
Please remember about decreasing the learning rate!


## Other scripts

To use separate scripts for training and evaluation please refer to respectively `train_models.py` and `evaluate_models.py`.


## Acknowledgments

We base our codebase on [Attack Agnostic Dataset repo](https://github.com/piotrkawa/attack-agnostic-dataset).
Apart from the dependencies mentioned in Attack Agnostic Dataset repository we also include: 
* [RawNet3 implementation](https://github.com/Jungjee/RawNet).



## Citation

If you use this code in your research please use the following citation:
```
@inproceedings{kawa23b_interspeech,
  author={Piotr Kawa and Marcin Plata and Michał Czuba and Piotr Szymański and Piotr Syga},
  title={{Improved DeepFake Detection Using Whisper Features}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={4009--4013},
  doi={10.21437/Interspeech.2023-1537}
}
```
