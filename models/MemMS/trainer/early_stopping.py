import os

import numpy as np
import torch


class EarlyStopping:
    """Save the model whenever the monitored validation score improves."""

    def __init__(self, save_path, idx, patience=20, delta=0.0):
        self.save_path = save_path
        self.idx = idx
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        os.makedirs(save_path, exist_ok=True)

    @property
    def checkpoint_path(self):
        return os.path.join(self.save_path, str(self.idx).zfill(2) + "_best_network.pth")

    def __call__(self, *args):
        if len(args) < 7:
            raise ValueError("EarlyStopping expects validation score and model arguments")
        score = float(args[3])
        model = args[6]
        if np.isnan(score):
            score = -np.inf

        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            return

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
