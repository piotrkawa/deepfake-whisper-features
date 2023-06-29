import argparse
import logging
from pathlib import Path

import torch
import yaml

import train_models
import evaluate_models
from src.commons import set_seed


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
LOGGER.addHandler(ch)


def parse_args():
    parser = argparse.ArgumentParser()

    ASVSPOOF_DATASET_PATH = "../datasets/ASVspoof2021/DF"
    IN_THE_WILD_DATASET_PATH = "../datasets/release_in_the_wild"

    parser.add_argument(
        "--asv_path",
        type=str,
        default=ASVSPOOF_DATASET_PATH,
        help="Path to ASVspoof2021 dataset directory",
    )
    parser.add_argument(
        "--in_the_wild_path",
        type=str,
        default=IN_THE_WILD_DATASET_PATH,
        help="Path to In The Wild dataset directory",
    )
    default_model_config = "config.yaml"
    parser.add_argument(
        "--config",
        help="Model config file path (default: config.yaml)",
        type=str,
        default=default_model_config,
    )

    default_train_amount = None
    parser.add_argument(
        "--train_amount",
        "-a",
        help=f"Amount of files to load for training.",
        type=int,
        default=default_train_amount,
    )

    default_valid_amount = None
    parser.add_argument(
        "--valid_amount",
        "-va",
        help=f"Amount of files to load for testing.",
        type=int,
        default=default_valid_amount,
    )

    default_test_amount = None
    parser.add_argument(
        "--test_amount",
        "-ta",
        help=f"Amount of files to load for testing.",
        type=int,
        default=default_test_amount,
    )

    default_batch_size = 8
    parser.add_argument(
        "--batch_size",
        "-b",
        help=f"Batch size (default: {default_batch_size}).",
        type=int,
        default=default_batch_size,
    )

    default_epochs = 10  # it was 5 originally
    parser.add_argument(
        "--epochs",
        "-e",
        help=f"Epochs (default: {default_epochs}).",
        type=int,
        default=default_epochs,
    )

    default_model_dir = "trained_models"
    parser.add_argument(
        "--ckpt",
        help=f"Checkpoint directory (default: {default_model_dir}).",
        type=str,
        default=default_model_dir,
    )

    parser.add_argument("--cpu", "-c", help="Force using cpu?", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # TRAIN MODEL

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config["data"].get("seed", 42)
    # fix all seeds
    set_seed(seed)

    if not args.cpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_dir = Path(args.ckpt)
    model_dir.mkdir(parents=True, exist_ok=True)

    evaluation_config_path, model_path = train_models.train_nn(
        datasets_paths=[
            args.asv_path,
        ],
        device=device,
        amount_to_use=(args.train_amount, args.valid_amount),
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_dir=model_dir,
        config=config,
    )

    with open(evaluation_config_path, "r") as f:
        config = yaml.safe_load(f)

    evaluate_models.evaluate_nn(
        model_paths=config["checkpoint"].get("path", []),
        batch_size=args.batch_size,
        datasets_paths=[args.in_the_wild_path],
        model_config=config["model"],
        amount_to_use=args.test_amount,
        device=device,
    )
