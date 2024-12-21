import time
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class Line_regress:
    def __init__(self, x_train, y_train, file_dataset: str, batch_size=10, epochs=80):
        start_time = time.perf_counter()
        self.model = Sequential()
        self.model.add(Dense(units=1, ))
        self.model.compile(optimizer=Adam(0.1),
                           loss='mean_squared_error')
        self.model.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=0.3)
        self.timer = time.perf_counter() - start_time
        self.library = 'tensorflow'
        self.file_dataset = file_dataset
        self.description_total = ''

    def description(self):
        text = f'Базовая модель для линейной регрессии библиотеки Tensorflow\nвремя создания и тренировки:'
        return f'{text}{self.timer:.4f}с\n\n'

    def get_weights(self):
        return self.model.get_layer(index=0).get_weights()

    def get_slope(self):
        weights = self.get_weights()
        return weights[0][0]

    def get_intercept(self):
        weights = self.get_weights()
        return weights[1]

    def predict(self, x):
        return self.model.predict(x)
