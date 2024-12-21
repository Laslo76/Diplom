import time

from torch.nn import Sequential, Linear, MSELoss
from torch.optim import SGD


class Line_regress:
    def __init__(self, x_train, y_train, file_dataset: str, count_epoch=1000):
        start_time = time.perf_counter()
        self.model = Sequential(
            Linear(1, 1))
        criterion = MSELoss(reduction='mean')
        optimizer = SGD(self.model.parameters(), lr=0.01)
        for epoch in range(count_epoch):
            predict = self.model(x_train)
            loss = criterion(predict, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.timer = time.perf_counter() - start_time
        self.library = 'pytorch'
        self.file_dataset = file_dataset
        self.description_total = ''

    def description(self):
        return f'Базовая модель линейной регрессии библиотеки PyTorch\nвремя создания и тренировки:{self.timer:.4f}\n\n'

    def get_slope(self):
        return self.model.linear.weight[0][0]

    def get_intercept(self):
        return self.model.linear.bias[0]

    def predict(self, x):
        return self.model(x)
