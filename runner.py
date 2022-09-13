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
    metrics_epoch_wide: t.Tensor
    losses: t.Tensor
    accuracies: t.Tensor
    confusion_matrices: t.Tensor


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
        num_examples = len(cast(Sized, self.validation_dataloader.dataset))

        losses = t.zeros((self.cfg["epochs"], 2), dtype=t.float64, device=self.device)
        accuracies = t.zeros((self.cfg["epochs"], 2), dtype=t.float64, device=self.device)
        metrics_epoch_wide = t.zeros((self.cfg["epochs"], 2, num_examples, 3), dtype=t.float64, device=self.device)
        confusion_matrices = t.zeros((self.cfg["epochs"], 2, self.cfg["output_classes"], self.cfg["output_classes"]), dtype=t.int64, device=self.device)

        for epoch in range(self.cfg["epochs"]):
            start_time = time.perf_counter()
            train_loss, train_accuracy, train_metrics, train_confusion_matrix = self.one_epoch(False, epoch)

            val_loss, val_accuracy, val_metrics, val_confusion_matrix = self.one_epoch(True, epoch)

            losses[epoch][0] = train_loss
            accuracies[epoch][0] = train_accuracy
            metrics_epoch_wide[epoch][0] = train_metrics
            confusion_matrices[epoch][0] = train_confusion_matrix

            losses[epoch][1] = val_loss
            accuracies[epoch][1] = val_accuracy
            metrics_epoch_wide[epoch][1] = val_metrics
            confusion_matrices[epoch][1] = val_confusion_matrix

            end_time = time.perf_counter()
            print("process took {:.2f} secs to complete".format(end_time - start_time))

            os.makedirs("./models/{}".format(self.name), exist_ok=True)

            self.model.save("./models/{}/epoch_{}.model".format(self.name, epoch))

            theMetrics = Metrics(metrics_epoch_wide=val_metrics, losses=losses, accuracies=accuracies, confusion_matrices=confusion_matrices)
            pickle.dump(theMetrics, open("./models/{}/all_epochs.metrics".format(self.name, epoch), "wb"))

        return losses, accuracies, metrics_epoch_wide, confusion_matrices

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
