"""A generic training wrapper."""
from copy import deepcopy
import logging
from typing import Callable, List, Optional

import torch
from torch.utils.data import DataLoader


LOGGER = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        device: str = "cpu",
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 1e-3},
        use_scheduler: bool = False,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.epoch_test_losses: List[float] = []
        self.use_scheduler = use_scheduler


def forward_and_loss(model, criterion, batch_x, batch_y, **kwargs):
    batch_out = model(batch_x)
    batch_loss = criterion(batch_out, batch_y)
    return batch_out, batch_loss


class GDTrainer(Trainer):
    def train(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        test_len: Optional[float] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
    ):
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
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        )
        test_loader = DataLoader(
            test,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        )

        criterion = torch.nn.BCEWithLogitsLoss()
        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        best_model = None
        best_acc = 0

        LOGGER.info(f"Starting training for {self.epochs} epochs!")

        forward_and_loss_fn = forward_and_loss

        if self.use_scheduler:
            batches_per_epoch = len(train_loader) * 2  # every 2nd epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optim,
                T_0=batches_per_epoch,
                T_mult=1,
                eta_min=5e-6,
                # verbose=True,
            )
        use_cuda = self.device != "cpu"

        for epoch in range(self.epochs):
            LOGGER.info(f"Epoch num: {epoch}")

            running_loss = 0
            num_correct = 0.0
            num_total = 0.0
            model.train()

            for i, (batch_x, _, batch_y) in enumerate(train_loader):
                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(self.device)

                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)

                batch_out, batch_loss = forward_and_loss_fn(
                    model, criterion, batch_x, batch_y, use_cuda=use_cuda
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
                if self.use_scheduler:
                    scheduler.step()

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
                batch_x = batch_x.to(self.device)

                with torch.no_grad():
                    batch_pred = model(batch_x)

                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)
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
                f"Epoch [{epoch+1}/{self.epochs}]: test/loss: {test_running_loss}, test/accuracy: {test_acc}, test/eer: {eer_val}"
            )

            if best_model is None or test_acc > best_acc:
                best_acc = test_acc
                best_model = deepcopy(model.state_dict())

            LOGGER.info(
                f"[{epoch:04d}]: {running_loss} - train acc: {train_accuracy} - test_acc: {test_acc}"
            )

        model.load_state_dict(best_model)
        return model
