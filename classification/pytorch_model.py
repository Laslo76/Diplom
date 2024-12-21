import time

from torch.nn import Sequential, Linear, Softmax, CrossEntropyLoss
from torch.optim import SGD

import services


class Model_torch:
    def __init__(self, in_param, out_class, x_train, y_train, file_dataset: str, count_epoch=5_000):
        start_time = time.perf_counter()
        self.model = Sequential(
            Linear(in_param, 16),
            Linear(16, 8),
            Linear(8, out_class),
            Softmax(dim=1))
        criterion = CrossEntropyLoss()
        optimizer = SGD(self.model.parameters(), lr=0.2)

        for epoch in range(count_epoch):
            predict = self.model(x_train)
            loss = criterion(predict, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.timer = time.perf_counter() - start_time
        self.library = 'pytorch'
        self.file_dataset = file_dataset
        self.matrix_of_inaccuracies = None
        self.size_matrix = 0
        self.description_total = ''
        self.dataset = None
        self.class_dictionary = {}

    def predict(self, x):
        return self.model(x)
