import logging
import time
from pathlib import Path

import torch
import yaml

from src.datasets.detection_dataset import DetectionDataset
from src.models import models
from copy import deepcopy
from typing import Optional

import torch
from torch.utils.data import DataLoader


LOGGER = logging.getLogger(__name__)



def save_model(
    model: torch.nn.Module,
    model_dir: Path | str,
    name: str,
) -> None:
    full_model_dir = Path(f"{model_dir}/{name}")
    full_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{full_model_dir}/ckpt.pth")


def get_datasets(
    datasets_paths: dict[str, Path],
    amount_to_use: tuple[int | None, int | None],
) -> tuple[DetectionDataset, DetectionDataset]:
    data_train = DetectionDataset(
        dataset_map=datasets_paths,
        subset="train",
        reduced_number=amount_to_use[0],
        oversample=True,
    )
    data_test = DetectionDataset(
        dataset_map=datasets_paths,
        subset="test",
        reduced_number=amount_to_use[1],
        oversample=True,
    )

    return data_train, data_test


def train_nn(
    datasets_paths: dict,
    batch_size: int,
    epochs: int,
    device: str,
    config: dict,
    model_dir: Path | None = None,
    amount_to_use: tuple[int | None, int | None] = (None, None),
    config_save_path: str = "configs",
) -> tuple[str, str]:
    logging.info("Loading data...")
    model_config = config["model"]
    model_name = model_config["name"]
    optimizer_config = model_config["optimizer"]

    timestamp = time.time()
    checkpoint_path = ""

    data_train, data_test = get_datasets(
        datasets_paths=datasets_paths,
        amount_to_use=amount_to_use,
    )

    current_model = models.get_model(
        model_name=model_name,
        config=model_config["parameters"],
        device=device,
    )

    # If provided weights, apply corresponding ones (from an appropriate fold)
    model_path = config["checkpoint"]["path"]
    if model_path:
        current_model.load_state_dict(torch.load(model_path))
        logging.info(
            f"Finetuning '{model_name}' model, weights path: '{model_path}', on {len(data_train)} audio files."
        )
        if config["model"]["parameters"].get("freeze_encoder"):
            for param in current_model.whisper_model.parameters():
                param.requires_grad = False
    else:
        logging.info(f"Training '{model_name}' model from scratch on {len(data_train)} audio files.")
    current_model = current_model.to(device)


    current_model = train(
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        optimizer_fn=torch.nn.Adam,
        optimizer_kwargs=optimizer_config["kwargs"],
        dataset=data_train,
        model=current_model,
        test_dataset=data_test,
    )

    if model_dir is not None:
        save_name = f"model__{model_name}__{timestamp}"
        save_model(
            model=current_model,
            model_dir=model_dir,
            name=save_name,
        )
        checkpoint_path = str(model_dir.resolve() / save_name / "ckpt.pth")

    # Save config for testing
    if model_dir is not None:
        config["checkpoint"] = {"path": checkpoint_path}
        config_name = f"model__{model_name}__{timestamp}.yaml"
        config_save_path = str(Path(config_save_path) / config_name)
        with open(config_save_path, "w") as f:
            yaml.dump(config, f)
        logging.info("Test config saved at location '{}'!".format(config_save_path))
    return config_save_path, checkpoint_path


def forward_and_loss(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_out = model(batch_x)
    batch_loss = criterion(batch_out, batch_y)
    return batch_out, batch_loss


def train(
    epochs: int,
    batch_size: int,
    device: str,
    optimizer_fn,
    optimizer_kwargs,
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    test_len: Optional[float] = None,
    test_dataset: Optional[torch.utils.data.Dataset] = None,
) -> torch.nn.Module:
    if test_dataset is not None:
        train = dataset
        test = test_dataset
    else:
        test_len = int(len(dataset) * test_len)
        train_len = len(dataset) - test_len
        lengths = [train_len, test_len]
        train, test = torch.utils.data.random_split(dataset, lengths)

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=6,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=6,
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optim = optimizer_fn(model.parameters(), **optimizer_kwargs)

    best_model = None
    best_acc = 0

    LOGGER.info(f"Starting training for {epochs} epochs!")

    forward_and_loss_fn = forward_and_loss

    # TODO: add normal scheduler strategy

    for epoch in range(epochs):
        LOGGER.info(f"Epoch num: {epoch}")

        running_loss = 0
        num_correct = 0.0
        num_total = 0.0
        model.train()

        for i, (batch_x, _, batch_y) in enumerate(train_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)

            batch_y = batch_y.unsqueeze(1).type(torch.float32).to(device)

            batch_out, batch_loss = forward_and_loss_fn(
                model, criterion, batch_x, batch_y
            )
            batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
            num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

            running_loss += batch_loss.item() * batch_size

            if i % 100 == 0:
                LOGGER.info(
                    f"[{epoch:04d}][{i:05d}]: {running_loss / num_total} {num_correct/num_total*100}"
                )

            optim.zero_grad()
            batch_loss.backward()
            optim.step()

        running_loss /= num_total
        train_accuracy = (num_correct / num_total) * 100

        LOGGER.info(
            f"Epoch [{epoch+1}/{self.epochs}]: train/loss: {running_loss}, train/accuracy: {train_accuracy}"
        )

        test_running_loss = 0.0
        num_correct = 0.0
        num_total = 0.0
        model.eval()
        eer_val = 0

        for batch_x, _, batch_y in test_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)

            with torch.no_grad():
                batch_pred = model(batch_x)

            batch_y = batch_y.unsqueeze(1).type(torch.float32).to(device)
            batch_loss = criterion(batch_pred, batch_y)

            test_running_loss += batch_loss.item() * batch_size

            batch_pred = torch.sigmoid(batch_pred)
            batch_pred_label = (batch_pred + 0.5).int()
            num_correct += (batch_pred_label == batch_y.int()).sum(dim=0).item()

        if num_total == 0:
            num_total = 1

        test_running_loss /= num_total
        test_acc = 100 * (num_correct / num_total)
        LOGGER.info(
            f"Epoch [{epoch+1}/{epochs}]: test/loss: {test_running_loss}, test/accuracy: {test_acc}, test/eer: {eer_val}"
        )

        if best_model is None or test_acc > best_acc:
            best_acc = test_acc
            best_model = deepcopy(model.state_dict())

        LOGGER.info(
            f"[{epoch:04d}]: {running_loss} - train acc: {train_accuracy} - test_acc: {test_acc}"
        )

    if best_model is not None:
        model.load_state_dict(best_model)
    return model
