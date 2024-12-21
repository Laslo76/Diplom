from os import path

import numpy as np
import pandas as pd
import sklearn.utils
import torch
from pandas import read_csv
from numpy import argmax, array as np_array, empty as np_empty, vectorize as np_vectorize, arange as np_arange
from numpy import meshgrid as np_meshgrid, c_ as np_c_, apply_along_axis as np_apply_along_axis

from torch import tensor as torch_tensor, float as torch_float, max as torch_max

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split

from tkinter import messagebox


def converter(y_test) -> np.array:
    """
    Получить индексы максимальных значений в рядах массива
    :param y_test: массив ndarray
    :return: возвращает массив ndarray
    """
    return argmax(y_test, axis=-1)


def to_tensor(in_array: np.array, lite=False, type_int=False) -> torch.Tensor:
    """
    Преобразует массив в тензор
    :param in_array: преобразуемый массив
    :param lite: компактная версия преобразования без решейпинга
    :param type_int: значения в тензоре целые если Истина, иначе вещественные
    :return:
    """
    if lite:
        if type_int:
            return torch_tensor(in_array)
        else:
            return torch_tensor(in_array).float()
    else:
        return torch_tensor(in_array).reshape(len(in_array), 1).to(torch_float)


def from_tensor(in_tensor: torch.Tensor) -> np.array:
    """
    Возвращает массив NuPy из Тензора
    :param in_tensor: Tensor
    :return: возвращает массив NuPy
    """
    return in_tensor.detach().numpy()


def test_file(file_name: str) -> bool:
    """
    Функция проверят корректно ли заполнено поле 'Имя файла датасета'
    :param file_name: строка с именем файла
    :return: возвращает True когда некорректно и False в противном случае
    """
    if not file_name:
        messagebox.showerror('Файловая ошибка', 'Ошибка: Не задано имя файла датасета!')
        return True

    if not path.exists(file_name):
        messagebox.showerror('Файловая ошибка', f'Ошибка: Не найден файл {file_name}!')
        return True

    return False


def load_dataset(file_name) -> pd.DataFrame:
    """
    Функция производит чтение из файла file_name
    :param file_name: имя файла для чтения
    :return: pandas.DataFrame
    """
    data_frame = read_csv(file_name)
    return data_frame


def dataset_for_torch(data_frame: pd.DataFrame, data_file_name: str) -> sklearn.utils.Bunch:
    """
    Подготавливает датасет для работы с библиотеками tensorflow и torch
    :param data_frame: сходный датафрейм
    :param data_file_name: имя файла
    :return: Объект-контейнер Bunch, предоставляющий ключи как атрибуты.
    """
    data = np_array(data_frame[['Products', 'Gas']])
    target = np_array(data_frame['class'])

    frame = np_empty(1)
    target_names = ['class']
    feature_names = ['Products', 'Gas']
    f_descr = 'types azs'

    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        DESCR=f_descr,
        feature_names=feature_names,
        filename=data_file_name,
        data_module="sklearn.datasets.data",
    )


def prepare_dataset(library: str, class_dictianory: dict, dataset):
    """
    Подготавливает - оцифровывает метки в исходном датасете
    :param library: наименование библиотеки для которой подготавливаем наборы
    :param class_dictianory: классификационный словарь для меток
    :param dataset: подготовленный массив данных
    :return: ничего не возвращает
    """

    if library == 'tensorflow':
        app_date = [class_dictianory[x] for x in dataset.target]
        dataset.target = np_array(app_date)
    else:
        app_date = np_vectorize(lambda x: class_dictianory[x])
        dataset.target = app_date(dataset.target)


def get_sets(dataset, name_var_x, name_var_y, library: str, size_train=0.8, size_test=0.2) -> tuple:
    """
    Функция делит исходный датасет на тренировочную и тестовую части.
    :param dataset: Исходный датасет
    :param name_var_x: список признаков объекта
    :param name_var_y: признак присвоенных обучающих меток
    :param library: имя используемой для классификации библиотеки
    :param size_train: доля данных для тренировки в общем датасете (от 0 до 1)
    :param size_test: доля данных для тестирования в общем датасете (от 0 до 1)
    :return: кортеж тренировочные признаки, тестовые признаки, учебные метки, тестовые метки
    """
    if library in ('tensorflow', 'pytorch'):
        x_train, x_test, y_train, y_test = train_test_split(dataset.data,
                                                            dataset.target,
                                                            test_size=size_test,
                                                            random_state=42)
    else:
        if type(name_var_x) is list:
            x = dataset[name_var_x]
            len_x = len(name_var_x)
        else:
            x = dataset[name_var_x]
            len_x = 1
        y = dataset[name_var_y]

        x = x.values.reshape(len(x), len_x)
        y = y.values.reshape(len(y), )

        size_train = int(len(x) * size_train)
        size_test = int(len(x) * size_test)
        x_train = x[:size_train]
        x_test = x[-size_test:]

        y_train = y[:size_train]
        y_test = y[-size_test:]

    return x_train, y_train, x_test, y_test


def predicted(model, x_train, x_test):
    """
    Функция получения прогнозируемых значений для заданных параметров
    :param model: модель линейной регрессии
    :param x_train: тренировочный набор данных
    :param x_test: тестовый набор данных
    :return: картеж результат прогнозирования для тренировочных данных и для тестовых данных
    """
    if model.library == 'pytorch':
        x_train = to_tensor(x_train)
        x_test = to_tensor(x_test)

    y_train, y_test = model.predict(x_train), model.predict(x_test)

    if model.library == 'pytorch':
        return from_tensor(y_train), from_tensor(y_test)
    else:
        return y_train, y_test


def my_converter(name_library: str, predicted_label, label_) -> tuple:
    """
    Подготовка множеств к сравнению
    :param name_library: имя библиотеки выполнившей предсказание меток
    :param predicted_label: коллекция предсказанные метки
    :param label_: коллекция фактические метки
    :return: возвращает картеж коллекций предсказанные метки, фактические метки
    """
    if name_library == 'tensorflow':
        y_pred = converter(predicted_label)
        y_test = converter(label_)
    elif name_library == 'pytorch':
        y_pred = torch_max(predicted_label, dim=1)[1]
        y_test = label_
    else:
        y_pred = predicted_label
        y_test = label_
    return y_pred, y_test


def matrix(class_model, predicted_y, test_y):
    """
    Построение матрицы неточностей и сохранение ее в поле matrix_of_inaccuracies экземпляра модели классификации
    :param class_model: экземпляр объекта модели классификации
    :param test_y: метки для обучения из тестовой части датасета
    :param predicted_y: полученные прогнозируемые метки для объектов тестовой части датасета
    :return:
    """
    name_library = class_model.library
    y_pred, y_test = my_converter(name_library, predicted_y, test_y)
    class_model.matrix_of_inaccuracies = confusion_matrix(y_pred, y_test)


def class_report(class_model, predicted_y, test_y):
    """
    Процедура заполнения реквизита description_total экземпляра объекта классификатора
    :param class_model: экземпляр объекта классификатора
    :param predicted_y: предсказанные значения для тестового набора данных
    :param test_y: обучающие метки для тестового набора данных
    :return: ничего не возвращает
    """
    title_result = "Класcификационная модель создана при\nпомощи библиотеки"

    y_pred, y_test = my_converter(class_model.library, predicted_y, test_y)
    result = f"{title_result} {class_model.library}\n{classification_report(y_pred, y_test)}\n"
    class_model.description_total = f"{result}Время на создание и тренировку модели: {class_model.timer:.3f}с"


def calc_metric(model, y_train, y_train_predict, y_test, y_predict):
    """
    Процедура заполнения реквизита description_total экземпляра объекта построителя линейной регрессии
    :param model: ссылка на объект построитель регрессии
    :param y_train: реальные значения из тренировочно набора данных
    :param y_train_predict: предсказанные значения для набора тренировочных данных
    :param y_test: реальные значения из тестового набора данных
    :param y_predict: предсказанные значения для набора тестовых данных
    :return: ничего не возвращает
    """
    model_desc = model.description()
    mse_train = mean_squared_error(y_train, y_train_predict)
    mse_test = mean_squared_error(y_test, y_predict)

    mae_train = mean_absolute_error(y_train, y_train_predict)
    mae_test = mean_absolute_error(y_test, y_predict)

    text_result = f'{model_desc}MSE\n  train: {mse_train:.4f}, \n  test: {mse_test:.4f}'
    text_mae = f'\n\nMAE\n  train: {mae_train:.4f},\n  test: {mae_test:.4f}'

    model.description_total = f'{text_result}{text_mae}'


def plots(name: str, lib: str, x_label, y_label, x_test, y_test, y_predict):
    """
    Построение графика линейной регрессии на фоне исходных данных.
    Формирует файл и помещает его в директорию ./img.
    :param name: Наименование диаграммы
    :param lib: библиотека при помощи которой получены результаты
    :param x_label: подпись оси Х
    :param y_label: подпись оси Y
    :param x_test: значения точек тестового набора по оси X
    :param y_test: значения точек тестового набора по оси Y
    :param y_predict: Предсказанные значения
    :return: Ничего не возвращает
    """
    plt.scatter(x_test, y_test, color='black')
    plt.title(f'Библиотека {lib}. {name}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(())
    plt.yticks(())
    plt.plot(x_test, y_predict, color='red', linewidth=3)
    plt.savefig(f'./img/{lib}_regression.png')
    plt.close()


def plots_confusion(name: str, confusion_mat, len_matrix: int):
    """
    Построение матрицы неточностей и вывод ее на экран
    :param name: заголовок диаграммы
    :param confusion_mat: матрица неточностей
    :param len_matrix: размерность матрицы
    :return: ничего не возвращает
    """
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.get_cmap('gray'))
    plt.title(name)
    plt.colorbar()
    ticks = np_arange(len_matrix)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.ylabel('Калибровочные метки')
    plt.xlabel('Предсказанные метки')
    plt.show()
    plt.close()


def visualize_classifier(class_model, x, y, x_base=None, mean=None, std=None):
    """
    Процедура формирования итогового графического представления результатов классификации модели
    :param class_model: Модель для проведения классификации
    :param x: массив признаков объектов
    :param y: массив меток объектов
    :param x_base: нормализованный тензор для модели из библиотеки pytorch
    :param mean: среднее значение всех элементов нормализованного тензора для модели из библиотеки pytorch
    :param std: стандартное отклонение в нормализованном тензоре для модели из библиотеки pytorch
    :return: ничего не возвращает
    """
    # Определение для Х и У минимального и максимального
    # значений, которые будут использоваться при построении сетки
    min_x, max_x = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    min_y, max_y = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    mesh_step_size = 0.01
    x_vals, y_vals = np_meshgrid(np_arange(min_x, max_x, mesh_step_size),
                                 np_arange(min_y, max_y, mesh_step_size))
    if class_model.library == "pytorch":
        y_pred = class_model.predict(torch_tensor(np_c_[x_vals.ravel(), y_vals.ravel()]).float())
        _, output = torch_max(y_pred, dim=1)
        prn_x = x_base
        # Восстановить пространство представления (денормализация) x_vals, y_vals (nparray)
        xprn_vals = np_apply_along_axis(lambda _: _ * float(std[0]) + float(mean[0]), axis=0, arr=x_vals)
        yprn_vals = np_apply_along_axis(lambda _: _ * float(std[1]) + float(mean[1]), axis=0, arr=y_vals)
    elif class_model.library == "tensorflow":
        array_for_predict = np_c_[x_vals.ravel(), y_vals.ravel()]
        predicted_ = class_model.predict(array_for_predict)
        output = converter(predicted_)
        xprn_vals = x_vals
        yprn_vals = y_vals
        prn_x = x
    else:
        output = class_model.predict(np_c_[x_vals.ravel(), y_vals.ravel()])
        prn_x = x
        xprn_vals = x_vals
        yprn_vals = y_vals
    output = output.reshape(x_vals.shape)
    plt.figure()
    plt.title(f'Классификатор для тестового набора. Модель - {class_model.library}')
    plt.pcolormesh(xprn_vals, yprn_vals, output, cmap=plt.cm.get_cmap('gray'))
    plt.scatter(prn_x[:, 0], prn_x[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.get_cmap('Paired'))
    plt.xlim(xprn_vals.min(), xprn_vals.max())
    plt.ylim(yprn_vals.min(), yprn_vals.max())
    plt.xticks((np_arange(int(prn_x[:, 0] .min() - 1), int(prn_x[:, 0].max() + 1), 1.0)))
    plt.yticks((np_arange(int(prn_x[:, 1] .min() - 1), int(prn_x[:, 1].max() + 1), 1.0)))
    plt.savefig(f'./img/{class_model.library}_classification.png')
    plt.close()
