from typing import cast
import pickle
from matplotlib import pyplot as plt

from runner import Metrics

metrics = cast(Metrics, pickle.load(open("./models/MNISTLadder/all_epochs.metrics", "rb")))

print(metrics["val_accuracies"].shape, metrics["train_accuracies"].shape)

f1 = plt.figure(1)
plt.plot(range(metrics["val_losses"].shape[0]), metrics["val_losses"])
plt.plot(range(metrics["train_losses"].shape[0]), metrics["train_losses"])
# f1.show()

f2 = plt.figure(2)
plt.plot(range(metrics["val_accuracies"].shape[0]), metrics["val_accuracies"])
plt.plot(range(metrics["train_accuracies"].shape[0]), metrics["train_accuracies"])

f3 = plt.figure(3)
plt.imshow(metrics["val_confusion_matrices"][29])

print(metrics["val_accuracies"])

plt.show()
