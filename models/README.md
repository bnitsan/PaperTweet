# Models

This directory contains the backbone of the generative model that we train based on the dataset.

1) cmd_utils.py - Contains utilities to parse arguments passed on to train.py and inference.py
2) dataset.py - Contains CustomDataset class, a torch Dataset class
3) model.py - Contains the model class CondGenModel, a PyTorch Lightning module
4) utils.py - Contains various utility functions