from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
import time


class model_SVC:
    def __init__(self, X_train, y_train, file_dataset: str):
        start_time = time.perf_counter()
        self.model = SVC()
        self.model.fit(X_train, y_train)
        self.timer = time.perf_counter() - start_time
        self.library = "scikit"
        self.file_dataset = file_dataset
        self.matrix_of_inaccuracies = None
        self.size_matrix = 0
        self.description = ''
        self.dataset = None
        self.class_dictionary = {}

    def predict(self, X_test):
        return self.model.predict(X_test)


class model_KNN:
    def __init__(self, X_train, y_train, file_dataset: str):
        start_time = time.perf_counter()
        self.model = KNN()
        self.model.fit(X_train, y_train)
        self.timer = time.perf_counter() - start_time
        self.library = "scikit"
        self.file_dataset = file_dataset
        self.matrix_of_inaccuracies = None
        self.size_matrix = 0
        self.description = ''
        self.dataset = None
        self.class_dictionary = {}

    def predict(self, X_test):
        return self.model.predict(X_test)
