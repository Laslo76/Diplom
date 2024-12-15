import torch
import time


class Model_torch:
    def __init__(self, in_param, out_class, x_train, y_train, file_dataset: str, count_epoch=10_000):
        start_time = time.perf_counter()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_param, out_class),
            torch.nn.Softmax(dim=1))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.2)

        for epoch in range(count_epoch):
            y_pred = self.model(x_train)
            loss = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.timer = time.perf_counter() - start_time
        self.library = 'pytorch'
        self.file_dataset = file_dataset
        self.matrix_of_inaccuracies = None
        self.size_matrix = 0
        self.description = ''
        self.dataset = None
        self.class_dictionary = {}

    def predict(self, x):
        return self.model(x)
