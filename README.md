# Ladder network in Pytorch
A ladder network to perform semi-supervised training.

Edit and run `train.py` to run the training.

Edit and run `hp_tuning.py` to tune hyperparameters with a custom implementation of Bayesian optimization.

This model achieves >98% accuracy on the MNIST database of handwritten digits with only 200 training examples per class. Achieves 89% accuracy on the EMNIST database of handwritten characters and digits.
