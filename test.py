
from typing import Dict, cast
from app import AutoEncoder, AutoEncoderConfig, AutoEncoderHyperParameters
import torch as t
from model import AutoEncoderModel
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from hp_tuning import BayesOpt

from matplotlib import pyplot as plt


def model(hp: Dict[str, float]):
    train_subset_size = 4700 * 5
    test_subset_size = 4700

    cfg: AutoEncoderConfig = {
        "input_shape": (28, 28),
        "output_classes": 47,
        "model": AutoEncoderModel,
        "train_dataset": Subset(datasets.MNIST("data/", train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081))
        ])), range(train_subset_size)),
        "test_dataset": Subset(datasets.MNIST("data/", train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081))
        ])), range(train_subset_size, train_subset_size + test_subset_size)),
        "epochs": 15,
        "optim": t.optim.Adam,
        "shuffle": True
    }

    a = AutoEncoder(cfg, cast(AutoEncoderHyperParameters, hp))
    totalLoss, totalAccuracy = a.go()
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
