from typing import Dict, cast
from runner import Ladder, LadderConfig, LadderHP
import torch as t
from model import LadderModel
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from bayes_opt import BayesOpt

from matplotlib import pyplot as plt


hp = {
    "batch_size": 7,
    "learning_rate": -2.5,
    "ladder_loss_acceleration": -0.69,
    "ladder_loss_saturation": -1.16,
    "ladder_loss_delay": 3.61,
    "ladder_corruption_std": -4,
}

cfg: LadderConfig = {
    "input_shape": (28, 28),
    "output_classes": 47,
    "train_dataset": Subset(datasets.EMNIST("data/", train=True, download=True, split="bymerge", transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
    ])), indices=range(100)),
    "test_dataset": Subset(datasets.EMNIST("data/", train=False, download=True, split="bymerge", transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
    ])), indices=range(100)),
    "epochs": 30,
    "optim": t.optim.Adam,
    "shuffle": True
}

a = Ladder(cfg, cast(LadderHP, hp))
totalLoss, totalAccuracy, _1, _2 = a.go()
