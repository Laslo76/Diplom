from tkinter import Tk, ttk
from tkinter import LEFT, TOP, END
from tkinter import filedialog
from tkinter import StringVar
from tkinter import Canvas, PhotoImage, Text

import classification.scikit_model
import classification.tensor_model as ctm
import classification.pytorch_model as cpm

from regression import scikit_model as rsm
from regression import tensor_model as rtf
from regression import pytorch_model as rpt


import services


REGRESS_MODELS = [('Scikit', 'scikit'), ('Tensorflow', 'tensorflow'), ('PyTorch', 'pytorch')]
CLASSIFICATION_MODELS = [('Scikit_SVC', 'scikit_SVC'),
                         ('Scikit_KNN', 'scikit_KNN'),
                         ('Tensorflow', 'tensorflow'),
                         ('PyTorch', 'pytorch')]


class Main_Window(Tk):
    def __init__(self, title, geometry):
        super().__init__()
        self.model_regression = {_[1]: None for _ in REGRESS_MODELS}
        self.model_classification = {_[1]: None for _ in CLASSIFICATION_MODELS}
        self.title(title)
        self.geometry(geometry)
        self.use_library_regress = StringVar()
        self.use_library_classification = StringVar()
        self.use_library_regress.set(REGRESS_MODELS[0][1])
        self.use_library_classification.set(CLASSIFICATION_MODELS[0][1])
        data_file_name = ttk.Frame(self, height=30)
        label_file_name = ttk.Label(data_file_name, text='Имя файла датасета:')
        self.file_name = ttk.Entry(data_file_name, width=170)
        file_selection_button = ttk.Button(data_file_name, width=2, text='..', command=self.click_button_file_dataset)

        label_file_name.pack(side=LEFT)
        self.file_name.pack(side=LEFT, padx=6)
        file_selection_button.pack(side=LEFT)
        data_file_name.pack(padx=2, pady=6)

        tab_control = ttk.Notebook(self)
        tab1 = ttk.Frame(self)
        tab2 = ttk.Frame(self)

        tab_control.add(tab1, text='Регрессия')
        top_level = ttk.Frame(tab1)
        top_level.pack(side=TOP, anchor='nw')
        label_use_library = ttk.Label(top_level, text='Использовать библиотеку:')
        self.type_buttons = [self.create_radio(c, top_level, self.use_library_regress) for c in REGRESS_MODELS]
        make_model_regression_button = ttk.Button(top_level, text='Создать модель',
                                                  command=self.click_do_model_regression)

        label_use_library.pack(side=LEFT, anchor='nw', padx=4, pady=6)
        for button in self.type_buttons:
            button.pack(side=LEFT, anchor='nw', padx=10, pady=6)
        make_model_regression_button.pack(side=LEFT, anchor='nw', padx=10, pady=6)

        img0 = PhotoImage(file='')
        img0.img = img0
        next_level = ttk.Frame(tab1)
        next_level.pack(side=TOP, anchor='nw')
        self.canvas_img = Canvas(next_level, width=650, height=484, bg='white')
        self.canvas_img.pack(side=LEFT, padx=2)
        self.img_regress_container = self.canvas_img.create_image(325, 245, image=img0)

        self.model_description = Text(next_level, width=64, height=30)
        self.model_description.pack(side=LEFT, padx=5)

        tab_control.add(tab2, text='Классификация')
        class_top_level = ttk.Frame(tab2)
        class_top_level.pack(side=TOP, anchor='nw')
        class_label_use_library = ttk.Label(class_top_level, text='Использовать библиотеку:')
        self.class_type_buttons = [self.create_radio(c, class_top_level, self.use_library_classification)
                                   for c in CLASSIFICATION_MODELS]
        make_model_classification_button = ttk.Button(class_top_level,
                                                      text='Создать модель',
                                                      command=self.click_do_model_classification)
        self.make_matrix_classification_button = ttk.Button(class_top_level,
                                                      text='Матрица неточности',
                                                      command=self.click_do_matrix)

        class_label_use_library.pack(side=LEFT, anchor='nw', padx=4, pady=6)
        for button in self.class_type_buttons:
            button.pack(side=LEFT, anchor='nw', padx=10, pady=6)
        make_model_classification_button.pack(side=LEFT, anchor='nw', padx=10, pady=6)

        img_c0 = PhotoImage(file='')
        img_c0.img = img0
        class_next_level = ttk.Frame(tab2)
        class_next_level.pack(side=TOP, anchor='nw')
        self.class_canvas_img = Canvas(class_next_level, width=650, height=484, bg='white')
        self.class_canvas_img.pack(side=LEFT, padx=2)
        self.img_classification_container = self.class_canvas_img.create_image(325, 245, image=img_c0)

        self.class_model_description = Text(class_next_level, width=64, height=30)
        self.class_model_description.pack(side=LEFT, padx=5)

        tab_control.pack(expand=1, padx=5, fill="both")

    def click_button_file_dataset(self):
        filepath = filedialog.askopenfilename()
        if filepath != "":
            self.file_name.delete(0, 'end')
            self.file_name.insert(0, filepath)

    def create_radio(self, option, parent, default_value):
        text, value = option
        return ttk.Radiobutton(parent, text=text,
                               value=value,
                               variable=default_value,
                               command=self.change_model)

    def change_model(self):
        self.model_description.delete("1.0", END)
        model_name = self.use_library_regress.get()
        current_model = self.model_regression.get(model_name)
        if current_model is None or current_model.file_dataset != self.file_name.get():
            self.canvas_img.delete('all')
            img_c0 = PhotoImage(file='')
            img_c0.img = img_c0
            self.img_regress_container = self.canvas_img.create_image(325, 245, image=img_c0)
            return

        img_c2 = PhotoImage(file=f'img/{self.use_library_regress.get()}_regression.png')
        img_c2.img = img_c2
        self.canvas_img.itemconfig(self.img_regress_container, image=img_c2)
        self.model_description.insert("1.0", current_model.description_total)

        self.class_model_description.delete("1.0", END)
        model_name = self.use_library_classification.get()
        current_model = self.model_classification.get(model_name)
        self.forgot_matrix(model_name)
        if current_model is None or current_model.file_dataset != self.file_name.get():
            self.class_canvas_img.delete('all')
            img_c0 = PhotoImage(file='')
            img_c0.img = img_c0
            self.img_classification_container = self.class_canvas_img.create_image(325, 245, image=img_c0)
            return

        img_c2 = PhotoImage(file=f'img/{self.use_library_classification.get()}_classification.png')
        img_c2.img = img_c2
        self.class_canvas_img.itemconfig(self.img_classification_container, image=img_c2)
        self.class_model_description.insert("1.0", current_model.description)

    def click_do_model_regression(self):
        img0 = PhotoImage(file='')
        self.canvas_img.itemconfig(self.img_regress_container, image=img0)

        if services.test_file(self.file_name.get()):
            return

        dataset = services.load_dataset(self.file_name.get())
        x_train, y_train, x_test, y_test = services.get_sets(dataset, 'Gas',
                                                             'Products',
                                                             0.8)

        current_model = self.model_regression.get(self.use_library_regress.get())
        if current_model is None or current_model.file_dataset != self.file_name.get():

            if self.model_regression[self.use_library_regress.get()] is None:
                if self.use_library_regress.get() == REGRESS_MODELS[2][1]:
                    Y_tensor = services.to_tensor(y_train)
                    X_tensor = services.to_tensor(x_train)

                    self.model_regression[self.use_library_regress.get()] = rpt.Line_regress(X_tensor,
                                                                                             Y_tensor,
                                                                                             self.file_name.get(),
                                                                                             1500)
                elif self.use_library_regress.get() == REGRESS_MODELS[1][1]:
                    self.model_regression[self.use_library_regress.get()] = rtf.Line_regress(x_train,
                                                                                             y_train,
                                                                                             self.file_name.get())
                else:
                    self.model_regression[self.use_library_regress.get()] = rsm.Line_regress(x_train,
                                                                                             y_train,
                                                                                             self.file_name.get())

        y_train_pred, y_pred = services.predicted(self.model_regression[self.use_library_regress.get()],
                                                  x_train, x_test)

        services.plots('Взаимосвязь продажи сопутствующих товаров\nот объемов реализации топлива',
                       self.use_library_regress.get(), 'Топливо', 'Сопутствующие товары',
                       x_test, y_test, y_pred)
        services.calc_metric(self.model_regression[self.use_library_regress.get()],
                                                           y_train,
                                                           y_train_pred,
                                                           y_test,
                                                           y_pred)

        self.model_description.delete("1.0", END)
        self.model_description.insert("1.0",
                                      self.model_regression[self.use_library_regress.get()].description_total)

        img2 = PhotoImage(file=f'img/{self.use_library_regress.get()}_regression.png')
        img2.img = img2
        self.canvas_img.itemconfig(self.img_regress_container, image=img2)

    def click_do_model_classification(self):
        prediction, y_test = [], []
        img_c0 = PhotoImage(file='')
        self.canvas_img.itemconfig(self.img_classification_container, image=img_c0)

        if services.test_file(self.file_name.get()):
            return
        model_name = self.use_library_classification.get()
        current_model = self.model_classification.get(model_name)
        if current_model is None or current_model.file_dataset != self.file_name.get():
            if self.use_library_classification.get() == 'tensorflow':
                dataset = services.load_for_torch(self.file_name.get())
                class_dictionary = {'A': [1, 0, 0, 0], 'B': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'D': [0, 0, 0, 1]}
            elif self.use_library_classification.get() == 'pytorch':
                dataset = services.load_for_torch(self.file_name.get())
                class_dictionary = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            else:
                dataset = services.load_dataset(self.file_name.get())
                class_dictionary = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

            X_train, y_train, X_test, y_test = services.prepare_dataset(self.use_library_classification.get(),
                                                                       class_dictionary,
                                                                       dataset)
            if self.use_library_classification.get() == 'scikit_SVC':
                self.model_classification[model_name] = classification.scikit_model.model_SVC(X_train, y_train,
                                                                       self.file_name.get())

                prediction = self.model_classification[model_name].predict(X_test)
                services.matrix(self.model_classification[model_name], y_test, prediction)
                services.class_report(self.model_classification[model_name], prediction, y_test)
                services.visualize_classifier(self.model_classification[model_name],
                                              X_test,
                                              y_test,
                                              self.use_library_classification.get())
            elif self.use_library_classification.get() == 'scikit_KNN':
                self.model_classification[model_name] = classification.scikit_model.model_KNN(X_train, y_train,
                                                                                              self.file_name.get())
                prediction = self.model_classification[model_name].predict(X_test)
                services.matrix(self.model_classification[model_name], y_test, prediction)
                services.class_report(self.model_classification[model_name], prediction, y_test)
                services.visualize_classifier(self.model_classification[model_name],
                                              X_test,
                                              y_test,
                                              self.use_library_classification.get())
            elif self.use_library_classification.get() == 'tensorflow':
                test_y = services.convertation(y_test)
                self.model_classification[model_name] = ctm.Model_tensorflow(X_train, y_train,4, self.file_name.get())
                prediction = self.model_classification[model_name].predict(X_test)
                services.matrix(self.model_classification[model_name], y_test, prediction)
                services.class_report(self.model_classification[model_name], prediction, y_test)
                services.visualize_classifier(self.model_classification[model_name],
                                              X_test,
                                              test_y,
                                              self.use_library_classification.get())
            else:
                # Преобразуем в PyTorch tensors
                X_train = services.to_tensor(X_train, True)
                X_test = services.to_tensor(X_test, True)
                y_train = services.to_tensor(y_train, True, True)
                y_test = services.to_tensor(y_test, True, True)

                # Нормализация исходных данных
                mean = X_train.mean(dim=0)
                std = X_train.std(dim=0)
                X_train = (X_train - mean) / std
                X_ = X_test.clone()
                X_test = (X_test - mean) / std

                self.model_classification[model_name] = cpm.Model_torch(2,
                                                                        4,
                                                                        X_train, y_train, self.file_name.get())
                prediction = self.model_classification[model_name].predict(X_test)
                services.matrix(self.model_classification[model_name], y_test, prediction)
                services.class_report(self.model_classification[model_name], prediction, y_test)
                services.visualize_classifier(self.model_classification[model_name],
                                              X_test,
                                              y_test,
                                              self.use_library_classification.get(),
                                              X_, mean, std)

        img_c2 = PhotoImage(file=f'img/{self.use_library_classification.get()}_classification.png')
        img_c2.img = img_c2
        self.class_canvas_img.itemconfig(self.img_classification_container, image=img_c2)
        self.forgot_matrix(model_name)

        self.class_model_description.delete("1.0", END)
        self.class_model_description.insert("1.0", self.model_classification[model_name].description)
        self.model_classification[model_name].size_matrix = 4
        self.model_classification[model_name].dataset = dataset
        self.model_classification[model_name].class_dictionary = class_dictionary

    def forgot_matrix(self, model_name):
        current_model = self.model_classification.get(model_name)
        if current_model is None:
            self.make_matrix_classification_button.pack_forget()
        else:
            self.make_matrix_classification_button.pack(side=LEFT, anchor='nw', padx=4, pady=6)

    def click_do_matrix(self):
        model_reg = self.use_library_regress.get()

        model_name = self.use_library_classification.get()
        current_model = self.model_classification.get(model_name)
        if current_model is not None:
            services.plots_confusion(f"Матрица неточности {current_model.library}",
                                     current_model.matrix_of_inaccuracies, current_model.size_matrix)
            img_c2 = PhotoImage(file=f'img/{self.use_library_classification.get()}_classification.png')
            img_c2.img = img_c2
            self.class_canvas_img.itemconfig(self.img_classification_container, image=img_c2)

            img_c2 = PhotoImage(file=f'img/{model_reg}_regression.png')
            img_c2.img = img_c2
            self.canvas_img.itemconfig(self.img_regress_container, image=img_c2)


if __name__ == '__main__':
    application = Main_Window("Классификация и регрессия", "1200x595")
    application.mainloop()
