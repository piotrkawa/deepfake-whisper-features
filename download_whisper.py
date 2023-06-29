# pip install git+https://github.com/openai/whisper.git
from collections import OrderedDict
import whisper
import torch

from src.commons import WHISPER_MODEL_WEIGHTS_PATH

def download_whisper():
    model = whisper.load_model("tiny.en")
    return model


def extract_and_save_encoder(model):
    model_ckpt = OrderedDict()

    model_ckpt['model_state_dict'] = OrderedDict()

    for key, value in model.encoder.state_dict().items():
        model_ckpt['model_state_dict'][f'encoder.{key}'] = value

    model_ckpt['dims'] = model.dims
    torch.save(model_ckpt, WHISPER_MODEL_WEIGHTS_PATH)


if __name__ == "__main__":
    model = download_whisper()
    print("Downloaded Whisper model!")
    extract_and_save_encoder(model)
    print(f"Saved encoder at '{WHISPER_MODEL_WEIGHTS_PATH}'")