import os
from time import time
from typing import NamedTuple, Any, Sized, TypedDict, cast
import torch as t
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
import time
from model import LadderFModule, LadderModel
import pickle


class LadderHP(TypedDict):
    learning_rate: float  # MUST BE A POWER OF 2. eg learning_rate of -2 means 0.01
    batch_size: int
    ladder_loss_saturation: float
    ladder_loss_acceleration: float
    ladder_loss_delay: float
    ladder_corruption_std: float


class LadderConfig(TypedDict):
    input_shape: tuple[int, int]
    output_classes: int

    train_dataset: Dataset
    test_dataset: Dataset

    epochs: int

    shuffle: bool
    optim: Any


class Metrics(TypedDict):
    train_metrics_epoch_wide: t.Tensor
    train_losses: t.Tensor
    train_accuracies: t.Tensor
    train_confusion_matrices: t.Tensor

    val_metrics_epoch_wide: t.Tensor
    val_losses: t.Tensor
    val_accuracies: t.Tensor
    val_confusion_matrices: t.Tensor


class Ladder:
    def init_dataloaders(self):
        batch_size = max(16, int(2 ** self.hp["batch_size"]))

        self.train_dataloader = DataLoader(
            self.cfg["train_dataset"],
            batch_size=batch_size,
            shuffle=self.cfg["shuffle"],
        )
        self.validation_dataloader = DataLoader(
            self.cfg["test_dataset"], batch_size=batch_size, shuffle=self.cfg["shuffle"]
        )

    def init_device(self):
        self.device = t.device("cpu")

    def init_model_hps(self, m: nn.Module):
        if isinstance(m, LadderFModule) or isinstance(m, LadderModel):
            m.corruption_std = 10 ** self.hp["ladder_corruption_std"]
        return

    def init_model(self):
        self.model = LadderModel(
            self.cfg["input_shape"], self.cfg["output_classes"]
        ).to(self.device)
        self.model.apply(self.init_model_hps)

    def init_optim(self):
        lr = 10 ** self.hp["learning_rate"]
        self.optim: t.optim.Optimizer = self.cfg["optim"](
            self.model.parameters(), lr=lr
        )

    def __init__(self, cfg: LadderConfig, hp: LadderHP, name=None):
        self.name = name if name is not None else "Ladder"
        # init hyperparameters
        self.hp = hp
        self.cfg = cfg

        self.init_device()
        # initialize model
        self.init_model()
        # initialize optimizer
        self.init_optim()
        # initialize dataloader
        self.init_dataloaders()

    def go(self):
        num_train_examples = len(cast(Sized, self.train_dataloader.dataset))
        num_val_examples = len(cast(Sized, self.validation_dataloader.dataset))

        train_losses = t.zeros((self.cfg["epochs"]), dtype=t.float64, device=self.device)
        train_accuracies = t.zeros((self.cfg["epochs"],), dtype=t.float64, device=self.device)
        train_metrics_epoch_wide = t.zeros((self.cfg["epochs"], num_train_examples, 3), dtype=t.float64, device=self.device)
        train_confusion_matrices = t.zeros((self.cfg["epochs"], self.cfg["output_classes"], self.cfg["output_classes"]), dtype=t.int64, device=self.device)

        val_losses = t.zeros((self.cfg["epochs"]), dtype=t.float64, device=self.device)
        val_accuracies = t.zeros((self.cfg["epochs"],), dtype=t.float64, device=self.device)
        val_metrics_epoch_wide = t.zeros((self.cfg["epochs"], num_val_examples, 3), dtype=t.float64, device=self.device)
        val_confusion_matrices = t.zeros((self.cfg["epochs"], self.cfg["output_classes"], self.cfg["output_classes"]), dtype=t.int64, device=self.device)

        for epoch in range(self.cfg["epochs"]):
            start_time = time.perf_counter()
            train_loss, train_accuracy, train_metrics, train_confusion_matrix = self.one_epoch(False, epoch)

            val_loss, val_accuracy, val_metrics, val_confusion_matrix = self.one_epoch(True, epoch)

            train_losses[epoch] = train_loss
            train_accuracies[epoch] = train_accuracy
            train_metrics_epoch_wide[epoch] = train_metrics
            train_confusion_matrices[epoch] = train_confusion_matrix

            val_losses[epoch] = val_loss
            val_accuracies[epoch] = val_accuracy
            val_metrics_epoch_wide[epoch] = val_metrics
            val_confusion_matrices[epoch] = val_confusion_matrix

            end_time = time.perf_counter()
            print("process took {:.2f} secs to complete".format(end_time - start_time))

            os.makedirs("./models/{}".format(self.name), exist_ok=True)

            self.model.save("./models/{}/epoch_{}.model".format(self.name, epoch))

            theMetrics = Metrics(val_metrics_epoch_wide=val_metrics_epoch_wide, val_losses=val_losses, val_accuracies=val_accuracies, val_confusion_matrices=val_confusion_matrices,
                                 train_metrics_epoch_wide=train_metrics_epoch_wide, train_losses=train_losses, train_accuracies=train_accuracies, train_confusion_matrices=train_confusion_matrices)
            pickle.dump(theMetrics, open("./models/{}/all_epochs.metrics".format(self.name, epoch), "wb"))

        return val_losses, val_accuracies, val_metrics_epoch_wide, val_confusion_matrices

    def ladder_weight(self, epoch: int):
        return (10 ** self.hp["ladder_loss_saturation"]) * t.sigmoid(
            t.tensor(
                [
                    (10 ** self.hp["ladder_loss_acceleration"])
                    * (epoch - self.hp["ladder_loss_delay"])
                ]
            )
        ).to(self.device)

    def one_epoch(self, val: bool, epoch):
        if val:
            self.model.eval()
            dl = self.validation_dataloader
        else:
            self.model.train()
            dl = self.train_dataloader

        num_examples = len(cast(Sized, dl.dataset))
        num_batches = num_examples // cast(int, dl.batch_size)

        totalLoss = t.tensor([0.0], device=self.device)
        totalAccuracy = t.tensor([0.0], device=self.device)

        metrics = t.zeros((num_examples, 3))
        confusion_matrix = t.zeros((self.cfg["output_classes"], self.cfg["output_classes"]), dtype=t.int64)

        for index, batch in enumerate(dl):
            if not val:
                self.optim.zero_grad()

            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            logit_loss_fn = nn.CrossEntropyLoss(reduction="none")

            ladder_up, ladder_down = self.model(x)
            logits = ladder_up[-1]

            logit_loss: t.Tensor = logit_loss_fn(logits, y)

            mean_layer_loss = self.ladder_weight(epoch) * sum([t.linalg.matrix_norm(ladder_up[i] - ladder_down[i]).mean() for i in range(len(ladder_up) - 1)])  # type: ignore

            mean_logit_loss = logit_loss.mean()

            if not val:
                mean_logit_loss.backward(retain_graph=True)
                cast(t.Tensor, mean_layer_loss).backward(retain_graph=True)
                self.optim.step()

            with t.no_grad():
                # calculate statistics
                totalLoss += mean_logit_loss.to("cpu") / num_examples
                batch_start = cast(int, dl.batch_size) * index
                batch_end = batch_start + y.size(0)

                predictions = logits.argmax(dim=1)  # tensor of indices of max values

                metrics[batch_start:batch_end, 0] = y
                metrics[batch_start:batch_end, 1] = predictions
                metrics[batch_start:batch_end, 2] = logit_loss

                accuracy = (y == predictions).sum() / y.size(0)
                confusion_matrix[y, predictions] += 1

                totalAccuracy += (y == predictions).sum() / num_examples

                if index % 100 == 0:
                    print(
                        "epoch {}/{}:\t{}\tbatch: {}/{}\tloss: {:.2f}\taccuracy: {:.2f}".format(
                            epoch,
                            self.cfg["epochs"],
                            "val" if val else "train",
                            index,
                            num_batches,
                            mean_logit_loss,
                            accuracy,
                        )
                    )

        return totalLoss, totalAccuracy, metrics, confusion_matrix
