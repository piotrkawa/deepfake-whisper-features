import logging
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from src import metrics
from src.models import models
from src.datasets.base_dataset import SimpleAudioFakeDataset
from src.datasets.in_the_wild_dataset import InTheWildDataset


def get_dataset(
    datasets_paths: list[Path | str],
    amount_to_use: int | None,
) -> SimpleAudioFakeDataset:
    data_val = InTheWildDataset(
        subset="foo",
        path=datasets_paths[0],
    )
    return data_val


def evaluate_nn(
    model_paths: list[Path],
    datasets_paths: list[Path | str],
    model_config: dict[str, Any],
    device: str,
    amount_to_use: int | None = None,
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
