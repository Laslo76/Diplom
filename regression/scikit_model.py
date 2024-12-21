from sklearn.linear_model import LinearRegression as sc_LinearRegression
import time


class Line_regress:
    def __init__(self, x_train, y_train, file_dataset: str):
        start_time = time.perf_counter()
        self.model = sc_LinearRegression()
        self.model.fit(x_train, y_train)
        self.timer = time.perf_counter() - start_time
        self.library = 'scikit'
        self.file_dataset = file_dataset
        self.description_total = ''

    def description(self):
        text_ = f'Базовая модель линейной регрессии библиотеки Scikit\nвремя создания и тренировки:'
        return f'{text_}{self.timer:.4f}\n\n'

    def get_slope(self):
        return self.model.coef_[0][0]

    def get_intercept(self):
        return self.model.intercept_[0]

    def predict(self, x):
        return self.model.predict(x)
