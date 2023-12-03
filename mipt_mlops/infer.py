import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader

from mipt_mlops.constants import BATCH_SIZE, data_dir, models_dir, predictions_dir
from mipt_mlops.datasets import Data
from mipt_mlops.networks import NeuralNetwork


if __name__ == "__main__":
    with open(data_dir / "test.npy", "rb") as stream:
        X_test = np.load(stream)
        y_test = np.load(stream)
    test_data = Data(X_test, y_test)
    test_dataloader = DataLoader(test_data, BATCH_SIZE, shuffle=True)
    model = NeuralNetwork()
    model.load_state_dict(torch.load(models_dir / "net.pkl"))

    y_pred = []
    total = 0
    correct = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            outputs = model(X)
            predicted = np.where(outputs < 0.5, 0, 1)
            predicted = list(itertools.chain(*predicted))
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y.numpy()).sum().item()
    result = np.array(list(itertools.chain(*y_pred)))
    result.tofile(predictions_dir / "predict.csv", sep=",")

    print(f"Accuracy of the network: {100 * correct // total}%")
