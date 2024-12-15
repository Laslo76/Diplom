from sklearn import linear_model
import time


class Line_regress:
    def __init__(self, x_train, y_train, file_dataset: str):
        start_time = time.perf_counter()
        self.model = linear_model.LinearRegression()
        self.model.fit(x_train, y_train)
        self.timer = time.perf_counter() - start_time
        self.library = 'scikit'
        self.file_dataset = file_dataset
        self.description_total = ''

    def description(self):
        return f'Базовая модель линейной регрессии библиотеки Scikit\nвремя создания и тренировки:{self.timer:.4f}\n\n'

    def get_slope(self):
        return self.model.coef_[0][0]

    def get_intercept(self):
        return self.model.intercept_[0]

    def predict(self, x):
        return self.model.predict(x)
