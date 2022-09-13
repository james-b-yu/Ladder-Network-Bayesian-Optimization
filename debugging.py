
import glob
import os
from typing import Dict, cast
from app import AutoEncoder, AutoEncoderConfig, AutoEncoderHyperParameters
import torch as t
from model import AutoEncoderModel
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from hp_tuning import BayesOpt
import gaussian_regression

from matplotlib import pyplot as plt

train_subset_size = 47000
test_subset_size = 3200

training_dataset = datasets.EMNIST("data/", train=True, split="balanced", download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))
]))
training_subset = Subset(training_dataset, range(train_subset_size))


train_x = training_dataset.data[training_subset.indices]  # type: ignore
train_y = training_dataset.train_labels[training_subset.indices]  # type: ignore


for l in range(47):
    print("count {}: {}".format(training_dataset.classes[l], (train_y == l).sum()))

models = glob.glob("./models/*.p")
lastest_model = max(models, key=os.path.getctime)
theModel = gaussian_regression.model.load(open(lastest_model, "rb"))
print(theModel.y)
