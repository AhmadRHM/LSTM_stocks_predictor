import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LSTM_stock_predictor(nn.Module):

    def __init__(self, hidden_dim):
        super(LSTM_stock_predictor, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes prices as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=3, batch_first=True, dropout=0.2)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2price = nn.Linear(hidden_dim, 1)

    def forward(self, prices):
        lstm_out, _ = self.lstm(prices)
        predicted_prices = self.hidden2price(lstm_out)
        return predicted_prices


def get_sequenced(data, seq_size=60):
    N = data.shape[0]
    length = data.shape[1]
    data_seqed = []
    for i in range(N):
        for j in range(seq_size, length):
            data_seqed.append(data[i, j - seq_size:j])
    return np.array(data_seqed)


def make_batch(data, batch_size=256):
    N = len(data)
    batched_data = []
    for i in range(0, N, batch_size):
        batched_data.append(data[i:min(i + batch_size, N), :])
    return batched_data


def prepare_data():
    prices = np.loadtxt("prices.csv", delimiter=',')
    sc = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = sc.fit_transform(prices)

    whole_length = 2406
    train_length = 2000
    validation_length = 300
    train_data = prices_scaled[:, :train_length]
    validation_data = prices_scaled[:, train_length:train_length + validation_length]
    test_data = prices_scaled[:, train_length + validation_length:]

    return make_batch(get_sequenced(train_data)), make_batch(get_sequenced(validation_data)), make_batch(
        get_sequenced(test_data))


def train(train_data, model, loss_function, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    batch_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(1, 50):
        model = model.train()

        end = time.time()
        for i, data in enumerate(train_data):
            model.zero_grad()

            data = torch.tensor(data, requires_grad=False).float()
            data = data.unsqueeze(dim=2)

            predicted_prices = model(data)

            loss = loss_function(data[:, 1:, 0], predicted_prices[:, :-1, 0])
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            batch_time.update(time.time() - end)
            end = time.time()
            if i % 10 == 0:  # print every 2560 sequences
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(epoch, i, len(train_data), batch_time=batch_time,
                                                                      loss=losses))


if __name__ == '__main__':
    train_data, validation_data, test_data = prepare_data()

    model = LSTM_stock_predictor(64).float()
    loss_function = nn.MSELoss()

    train(train_data, model, loss_function, lr=0.01)
