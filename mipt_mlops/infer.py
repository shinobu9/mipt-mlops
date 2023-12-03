import itertools

import mipt_mlops.constants as constants
import numpy as np
import torch
from mipt_mlops.datasets import Data
from mipt_mlops.networks import NeuralNetwork
from torch.utils.data import DataLoader


if __name__ == "__main__":
    with open(constants.data_dir / "test.npy", "rb") as stream:
        X_test = np.load(stream)
        y_test = np.load(stream)
    test_data = Data(X_test, y_test)
    test_dataloader = DataLoader(test_data, constants.BATCH_SIZE, shuffle=True)
    model = NeuralNetwork()
    model.load_state_dict(torch.load(constants.models_dir / "net.pkl"))

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
    result.tofile(constants.predictions_dir / "predict.csv", sep=",")

    print(f"Accuracy of the network: {100 * correct // total}%")
