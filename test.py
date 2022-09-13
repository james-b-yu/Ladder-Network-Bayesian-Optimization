
from typing import Dict, cast
from runner import Ladder, LadderConfig, LadderHP
import torch as t
from model import LadderModel
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from hp_tuning import BayesOpt

from matplotlib import pyplot as plt


def model(hp: Dict[str, float]):
    train_subset_size = 47 * 1000
    test_subset_size = 47 * 1000

    cfg: LadderConfig = {
        "input_shape": (28, 28),
        "output_classes": 47,
        "train_dataset": Subset(datasets.EMNIST("data/", train=True, download=True, split="balanced", transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081))
        ])), range(train_subset_size)),
        "test_dataset": Subset(datasets.EMNIST("data/", train=True, download=True, split="balanced", transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081))
        ])), range(train_subset_size, train_subset_size + test_subset_size)),
        "epochs": 30,
        "optim": t.optim.Adam,
        "shuffle": True
    }

    a = Ladder(cfg, cast(LadderHP, hp))
    totalLoss, totalAccuracy, _1, _2 = a.go()
    return totalAccuracy.max()


hp_in = {
    "batch_size": (6, 8),
    "learning_rate": (-4, -2),
    "ladder_loss_acceleration": (-2, 0),
    "ladder_loss_saturation": (-4, -1),
    "ladder_loss_delay": (0, 10),
    "ladder_corruption_std": (-4, -1),
}


b = BayesOpt(hp_in, model, "hp")

b.find_hyperparameters()
