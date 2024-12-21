from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import time


class model_SVC:
    def __init__(self, x_train, y_train, file_dataset: str):
        start_time = time.perf_counter()
        self.model = SVC()
        self.model.fit(x_train, y_train)
        self.timer = time.perf_counter() - start_time
        self.library = "scikit_SVC"
        self.file_dataset = file_dataset
        self.matrix_of_inaccuracies = None
        self.size_matrix = 0
        self.description_total = ''
        self.dataset = None
        self.class_dictionary = {}

    def predict(self, x_test):
        return self.model.predict(x_test)


class model_KNN:
    def __init__(self, x_train, y_train, file_dataset: str):
        start_time = time.perf_counter()
        self.model = KNeighborsClassifier()
        self.model.fit(x_train, y_train)
        self.timer = time.perf_counter() - start_time
        self.library = "scikit_KNN"
        self.file_dataset = file_dataset
        self.matrix_of_inaccuracies = None
        self.size_matrix = 0
        self.description = ''
        self.dataset = None
        self.class_dictionary = {}

    def predict(self, x_test):
        return self.model.predict(x_test)
