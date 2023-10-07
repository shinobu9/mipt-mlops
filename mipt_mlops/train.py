import warnings
import numpy as np
import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from .constants import models_dir, data_dir
from .datasets import Data
from .networks import NeuralNetwork


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    X, y = make_circles(n_samples=100, noise=0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    with open(data_dir / 'train.npy', 'wb') as stream:
        np.save(stream, X_train)
        np.save(stream, y_train)
    with open(data_dir / 'train.npy', 'rb') as stream:
        X_train1 = np.load(stream)
        y_train1 = np.load(stream)

    print(X_train == X_train1)
    print(y_train == y_train1)

    # batch_size = 64
    # train_data = Data(X_train, y_train)
    # train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
    # test_data = Data(X_test, y_test)
    # test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

    # input_dim = 2
    # hidden_dim = 10
    # output_dim = 1
    # model = NeuralNetwork(input_dim, hidden_dim, output_dim)

    # learning_rate = 0.1
    # loss_fn = nn.BCELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # num_epochs = 100
    # loss_values = []
    # for epoch in range(num_epochs):
    #     print(f"Training epoch #{epoch}")
    #     for X, y in train_dataloader:
    #         optimizer.zero_grad()
    #         pred = model(X)
    #         loss = loss_fn(pred, y.unsqueeze(-1))
    #         loss_values.append(loss.item())
    #         loss.backward()
    #         optimizer.step()
    # print("Training Complete")
    # torch.save(model.state_dict(), models_dir / "circle_network.pkl")
