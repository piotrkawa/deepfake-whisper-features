import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import sys

import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from src import metrics, commons
from src.models import models
from src.datasets.base_dataset import SimpleAudioFakeDataset
from src.datasets.in_the_wild_dataset import InTheWildDataset


def get_dataset(
    datasets_paths: List[Union[Path, str]],
    amount_to_use: Optional[int],
) -> SimpleAudioFakeDataset:
    data_val = InTheWildDataset(
        subset="foo",
        path=datasets_paths[0],
    )
    return data_val


def evaluate_nn(
    model_paths: List[Path],
    datasets_paths: List[Union[Path, str]],
    model_config: Dict,
    device: str,
    amount_to_use: Optional[int] = None,
    batch_size: int = 8,
):
    logging.info("Loading data...")
    model_name, model_parameters = model_config["name"], model_config["parameters"]

    # Load model architecture
    model = models.get_model(
        model_name=model_name,
        config=model_parameters,
        device=device,
    )
    # If provided weights, apply corresponding ones (from an appropriate fold)
    if len(model_paths):
        model.load_state_dict(torch.load(model_paths))
    model = model.to(device)

    data_val = get_dataset(
        datasets_paths=datasets_paths,
        amount_to_use=amount_to_use,
    )

    logging.info(
        f"Testing '{model_name}' model, weights path: '{model_paths}', on {len(data_val)} audio files."
    )
    test_loader = DataLoader(
        data_val,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=3,
    )

    batches_number = len(data_val) // batch_size
    num_correct = 0.0
    num_total = 0.0

    y_pred = torch.Tensor([]).to(device)
    y = torch.Tensor([]).to(device)
    y_pred_label = torch.Tensor([]).to(device)

    for i, (batch_x, _, batch_y) in enumerate(test_loader):
        model.eval()
        if i % 10 == 0:
            print(f"Batch [{i}/{batches_number}]")

        with torch.no_grad():
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            num_total += batch_x.size(0)

            batch_pred = model(batch_x).squeeze(1)
            batch_pred = torch.sigmoid(batch_pred)
            batch_pred_label = (batch_pred + 0.5).int()

            num_correct += (batch_pred_label == batch_y.int()).sum(dim=0).item()

            y_pred = torch.concat([y_pred, batch_pred], dim=0)
            y_pred_label = torch.concat([y_pred_label, batch_pred_label], dim=0)
            y = torch.concat([y, batch_y], dim=0)

    eval_accuracy = (num_correct / num_total) * 100

    precision, recall, f1_score, support = precision_recall_fscore_support(
        y.cpu().numpy(), y_pred_label.cpu().numpy(), average="binary", beta=1.0
    )
    auc_score = roc_auc_score(y_true=y.cpu().numpy(), y_score=y_pred.cpu().numpy())

    # For EER flip values, following original evaluation implementation
    y_for_eer = 1 - y

    thresh, eer, fpr, tpr = metrics.calculate_eer(
        y=y_for_eer.cpu().numpy(),
        y_score=y_pred.cpu().numpy(),
    )

    eer_label = f"eval/eer"
    accuracy_label = f"eval/accuracy"
    precision_label = f"eval/precision"
    recall_label = f"eval/recall"
    f1_label = f"eval/f1_score"
    auc_label = f"eval/auc"

    logging.info(
        f"{eer_label}: {eer:.4f}, {accuracy_label}: {eval_accuracy:.4f}, {precision_label}: {precision:.4f}, {recall_label}: {recall:.4f}, {f1_label}: {f1_score:.4f}, {auc_label}: {auc_score:.4f}"
    )


def main(args):
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    LOGGER.addHandler(ch)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if not args.cpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config["data"].get("seed", 42)
    # fix all seeds - this should not actually change anything
    commons.set_seed(seed)

    evaluate_nn(
        model_paths=config["checkpoint"].get("path", []),
        datasets_paths=[
            args.in_the_wild_path,
        ],
        model_config=config["model"],
        amount_to_use=args.amount,
        device=device,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    # If assigned as None, then it won't be taken into account
    IN_THE_WILD_DATASET_PATH = "../datasets/release_in_the_wild"

    parser.add_argument(
        "--in_the_wild_path", type=str, default=IN_THE_WILD_DATASET_PATH
    )

    default_model_config = "config.yaml"
    parser.add_argument(
        "--config",
        help="Model config file path (default: config.yaml)",
        type=str,
        default=default_model_config,
    )

    default_amount = None
    parser.add_argument(
        "--amount",
        "-a",
        help=f"Amount of files to load from each directory (default: {default_amount} - use all).",
        type=int,
        default=default_amount,
    )

    parser.add_argument("--cpu", "-c", help="Force using cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
