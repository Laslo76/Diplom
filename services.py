import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
from sklearn.utils import Bunch
import torch
from sklearn.model_selection import train_test_split
from tkinter import messagebox
from os import path


def convertation(y_test):
    return np.argmax(y_test, axis=-1)


def to_tensor(in_array, lite=False, type_int=False):
    if lite:
        if type_int:
            return torch.tensor(in_array)
        else:
            return torch.tensor(in_array).float()
    else:
        return torch.tensor(in_array).reshape(len(in_array), 1).to(torch.float)


def from_tensor(in_tensor):
    return in_tensor.detach().numpy()


def test_file(file_name):
    if "" == file_name:
        messagebox.showerror('Файловая ошибка', 'Ошибка: Не задано имя файла датасета!')
        return True

    if not path.exists(file_name):
        messagebox.showerror('Файловая ошибка', f'Ошибка: Не найден файл {file_name}!')
        return True

    return False


def load_dataset(file_name):
    data_frame = pd.read_csv(file_name)
    return data_frame


def load_for_torch(file_name, return_X_y=False, as_frame=False):
    data_frame = load_dataset(file_name)

    data_file_name = file_name
    data = np.array(data_frame[['Products', 'Gas']])
    target = np.array(data_frame['class'])
    if as_frame:
        frame = np.array(data_frame[['Products', 'Gas', 'class']])
    else:
        frame = np.empty(1)
    target_names = ['class']
    feature_names = ['Products', 'Gas']
    fdescr = 'types azs'

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file_name,
        data_module="sklearn.datasets.data",
    )


def prepare_dataset(library, class_dictianory, dataset):
    if library == 'tensorflow':
        appdate = np.vectorize(lambda x: class_dictianory[x])
        apdate_target = [class_dictianory[x] for x in dataset.target]
        dataset.target = np.array(apdate_target)
        X_train, X_test, y_train, y_test = train_test_split(dataset.data,
                                                            dataset.target,
                                                            test_size=0.2,
                                                            random_state=42)
    elif library == 'pytorch':
        appdate = np.vectorize(lambda x: class_dictianory[x])
        dataset.target = appdate(dataset.target)

        X_train, X_test, y_train, y_test = train_test_split(dataset.data,
                                                            dataset.target,
                                                            test_size=0.2,
                                                            random_state=42)

    else:
        dataset['class'] = dataset['class'].apply(lambda x: class_dictianory[x])
        X_train, y_train, X_test, y_test = get_sets(dataset,
                                                    ['Products', 'Gas'],
                                                    'class', size_train=1, size_test=0.1)
    return X_train, y_train, X_test, y_test


def get_sets(dataset, name_varX, name_varY, size_train=0.95, size_test=0.1) -> tuple:
    if type(name_varX) is list:
        x = dataset[name_varX]
        len_x = len(name_varX)
    else:
        x = dataset[name_varX]
        len_x = 1
    y = dataset[name_varY]

    x = x.values.reshape(len(x), len_x)
    y = y.values.reshape(len(y), 1)

    size_train = int(len(x) * size_train)
    size_test = int(len(x) * size_test)
    x_train = x[:size_train]
    x_test = x[-size_test:]

    y_train = y[:size_train]
    y_test = y[-size_test:]

    return x_train, y_train, x_test, y_test


def predicted(model, x_train, x_test):
    if model.library == 'pytorch':
        x_train = to_tensor(x_train)
        x_test = to_tensor(x_test)

    y_train, y_test = model.predict(x_train), model.predict(x_test)

    if model.library == 'pytorch':
        return from_tensor(y_train), from_tensor(y_test)
    else:
        return y_train, y_test


def plots(name: str, lib: str, x_label, y_label, x_test, y_test, y_predict):
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
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
    plt.title(name)
    plt.colorbar()
    ticks = np.arange(len_matrix)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.ylabel('Калибровочные метки')
    plt.xlabel('Предсказанные метки')
    plt.show()
    plt.close()


def visualize_classifier(classifier, X, y, name, X_base=None, mean=None, std=None):
    # Определение для Х и У минимального и максимального
    # значений, которые будут использоваться при построении сетки
    min_x, max_x = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    min_y, max_y = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    mesh_step_size = 0.01
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
                                 np.arange(min_y, max_y, mesh_step_size))
    if classifier.library == "pytorch":
        y_pred = classifier.predict(torch.tensor(np.c_[x_vals.ravel(), y_vals.ravel()]).float())
        _, output = torch.max(y_pred, dim=1)
        prnX = X_base
        # Восстановить пространство представления денормализация x_vals, y_vals (nparray)
        xprn_vals = np.apply_along_axis(lambda _: _ * float(std[0]) + float(mean[0]), axis=0, arr=x_vals)
        yprn_vals = np.apply_along_axis(lambda _: _ * float(std[1]) + float(mean[1]), axis=0, arr=y_vals)
    elif classifier.library == "tensorflow":
        array_for_predict = np.c_[x_vals.ravel(), y_vals.ravel()]
        predicted = classifier.predict(array_for_predict)
        output = np.argmax(predicted, axis=-1)
        xprn_vals = x_vals
        yprn_vals = y_vals
        prnX = X
    else:
        output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
        prnX = X
        xprn_vals = x_vals
        yprn_vals = y_vals
    output = output.reshape(x_vals.shape)
    plt.figure()
    plt.title(f'Классификатор для тестового набора. Модель - {name}')
    plt.pcolormesh(xprn_vals, yprn_vals, output, cmap=plt.cm.gray)
    plt.scatter(prnX[:, 0], prnX[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    plt.xlim(xprn_vals.min(), xprn_vals.max())
    plt.ylim(yprn_vals.min(), yprn_vals.max())
    plt.xticks((np.arange(int(prnX[:, 0] .min() - 1), int(prnX[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(prnX[:, 1] .min() - 1), int(prnX[:, 1].max() + 1), 1.0)))
    plt.savefig(f'./img/{name}_classification.png')
    plt.close()


def matrix(class_model, test_y, predicted_y):
    if class_model.library == 'tensorflow':
        y_pred = convertation(predicted_y)
        y_test = convertation(test_y)
    elif class_model.library == 'pytorch':
        y_pred = torch.max(predicted_y, dim=1)[1]
        y_test = test_y
    else:
        y_pred = predicted_y
        y_test = test_y
    class_model.matrix_of_inaccuracies = confusion_matrix(y_pred, y_test)


def class_report(class_model, predicted_y, test_y):
    title_result = "Класcификационная модель создана при\nпомощи библиотеки"
    if class_model.library == 'tensorflow':
        y_pred = convertation(predicted_y)
        y_test = convertation(test_y)
    elif class_model.library == 'pytorch':
        y_pred = torch.max(predicted_y, dim=1)[1]
        y_test = test_y
    else:
        y_pred = predicted_y
        y_test = test_y
    result = f"{title_result} {class_model.library}\n{classification_report(y_pred, y_test)}\n"
    class_model.description = f"{result}Время на создание и тренировку модели: {class_model.timer:.3f}с"


def calc_metric(model, y_train, y_train_predict, y_test, y_predict):
    model_desc = model.description()
    mse_train = mean_squared_error(y_train, y_train_predict)
    mse_test = mean_squared_error(y_test, y_predict)

    mae_train = mean_absolute_error(y_train, y_train_predict)
    mae_test = mean_absolute_error(y_test, y_predict)

    text_result = f'{model_desc}MSE\n  train: {mse_train:.4f}, \n  test: {mse_test:.4f}'
    text_mae = f'\n\nMAE\n  train: {mae_train:.4f},\n  test: {mae_test:.4f}'

    model.description_total = f'{text_result}{text_mae}'
