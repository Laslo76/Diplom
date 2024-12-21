import time
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input


class Model_tensorflow:
    def __init__(self, x, y, count_types, file_dataset: str, batch_size=8, epochs=80):
        start_time = time.perf_counter()
        self.model = Sequential([
            Input(shape=(x.shape[1],)),  # размер 2
            Dense(8, activation='relu'),
            Dense(16, activation='relu'),
            Dense(count_types, activation='softmax')  # размер 1
        ])
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(x, y, batch_size, epochs)
        self.timer = time.perf_counter() - start_time
        self.library = 'tensorflow'
        self.file_dataset = file_dataset
        self.matrix_of_inaccuracies = None
        self.size_matrix = 0
        self.description_total = ''
        self.dataset = None
        self.class_dictionary = {}

    def predict(self, x):
        return self.model.predict(x)
