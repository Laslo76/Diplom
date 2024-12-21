from tkinter import Tk, ttk
from tkinter import LEFT, TOP, END, filedialog, StringVar
from tkinter import Canvas, PhotoImage, Text

from classification import scikit_model as scmc, tensor_model as ctm, pytorch_model as cpm
from regression import scikit_model as scm, tensor_model as rtf, pytorch_model as rpt
import services

REGRESS_MODELS = [('Scikit', 'scikit'), ('Tensorflow', 'tensorflow'), ('PyTorch', 'pytorch')]
CLASSIFICATION_MODELS = [('Scikit_SVC', 'scikit_SVC'),
                         ('Scikit_KNN', 'scikit_KNN'),
                         ('Tensorflow', 'tensorflow'),
                         ('PyTorch', 'pytorch')]
CLASSIFICATION_DICTIONARY = {
    'tensorflow': {'A': [1, 0, 0, 0], 'B': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'D': [0, 0, 0, 1]},
    'pytorch': {'A': 0, 'B': 1, 'C': 2, 'D': 3},
    'scikit_SVC': {'A': 0, 'B': 1, 'C': 2, 'D': 3},
    'scikit_KNN': {'A': 0, 'B': 1, 'C': 2, 'D': 3}}


class Main_Window(Tk):
    """
    Основное окно программы
    """
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

    @staticmethod
    def visualisation_model(img_container, current_canvas, file_name=''):
        """
        Заполняет поле графического представления результатов работы модели.
        :param img_container: Контейнер для вывода.
        :param current_canvas: Обрабатываемый контур.
        :param file_name: Имя отображаемого файла.
        :return: Ничего не возвращает
        """
        current_canvas.delete('all')
        img_c0 = PhotoImage(file=file_name)
        img_c0.img = img_c0
        img_container = current_canvas.create_image(325, 245, image=img_c0)

    def create_radio(self, option: tuple, parent, default_value):
        """
        Функция создает и возвращает радио-кнопку
        :param option: кортеж текст и значение кнопки
        :param parent: родительский элемент
        :param default_value: значение по умолчанию
        :return: возвращает ссылку на радио-кнопку
        """
        text, value = option
        return ttk.Radiobutton(parent, text=text,
                               value=value,
                               variable=default_value,
                               command=self.change_model)

    def click_button_file_dataset(self):
        """
        Обработка нажатия кнопки выбора файла содержащего набор данных
        заполняет поле с именем файла.
        :return: Ничего не возвращает
        """
        filepath = filedialog.askopenfilename()
        if filepath:
            self.file_name.delete(0, 'end')
            self.file_name.insert(0, filepath)

    def change_model(self):
        """
        Функция описывает поведение основного окна программы при выборе радиокнопки.
        Если модель произведено переключение на готовую модель, то заполняется графическое представление
        результатов работы модели и оценки параметров качества. В противном случае эти поля очищаются.
        :return: Ничего не возвращает
        """
        self.model_description.delete("1.0", END)
        model_name = self.use_library_regress.get()
        current_model = self.model_regression.get(model_name)
        file_out = ''
        if current_model is not None and current_model.file_dataset == self.file_name.get():
            file_out = f'img/{self.use_library_regress.get()}_regression.png'
            self.model_description.insert("1.0", current_model.description_total)
        self.visualisation_model(self.img_regress_container, self.canvas_img, file_out)

        self.class_model_description.delete("1.0", END)
        model_name = self.use_library_classification.get()
        current_model = self.model_classification.get(model_name)
        self.forgot_matrix(model_name)
        file_out = ''
        if current_model is not None and current_model.file_dataset == self.file_name.get():
            file_out = f'img/{self.use_library_classification.get()}_classification.png'
            self.class_model_description.insert("1.0", current_model.description_total)
        self.visualisation_model(self.img_classification_container, self.class_canvas_img, file_out)

    def click_do_model_regression(self):
        """
        Процедура построения и обучения модели для построения линейной регрессии
        :return: Ничего не возвращает
        """
        self.visualisation_model(self.img_regress_container, self.canvas_img)

        if services.test_file(self.file_name.get()):
            # Если не выбран файл с исходными данными прервать процедуру
            return

        dataset = services.load_dataset(self.file_name.get())
        x_train, y_train, x_test, y_test = services.get_sets(dataset,
                                                             'Gas',
                                                             'Products',
                                                             0.8)

        current_library = self.use_library_regress.get()
        current_model = self.model_regression.get(current_library)
        file_dataset = self.file_name.get()
        if current_model is None or current_model.file_dataset != file_dataset:
            # Если модель не сформирована или изменился набор исходных данных перестроить и обучить модель
            if current_library == 'pytorch':
                tensor_y = services.to_tensor(y_train)
                tensor_x = services.to_tensor(x_train)
                current_model = rpt.Line_regress(tensor_x, tensor_y, file_dataset, 1500)

            elif current_library == 'tensorflow':
                current_model = rtf.Line_regress(x_train,
                                                 y_train,
                                                 file_dataset)

            else:
                current_model = scm.Line_regress(x_train, y_train, file_dataset)

            self.model_regression[current_library] = current_model

        # Заполняем поля с графическим представлением регрессии и описанием полученной модели
        y_train_pred, y_pred = services.predicted(current_model, x_train, x_test)

        services.plots('Взаимосвязь продажи сопутствующих товаров\nот объемов реализации топлива',
                       current_library, 'Топливо', 'Сопутствующие товары',
                       x_test, y_test, y_pred)
        services.calc_metric(current_model, y_train, y_train_pred, y_test, y_pred)

        self.model_description.delete("1.0", END)
        self.model_description.insert("1.0", current_model.description_total)

        file_out = f'img/{self.use_library_regress.get()}_regression.png'
        self.visualisation_model(self.img_regress_container, self.canvas_img, file_out)

    def click_do_model_classification(self):
        """
        Процедура построения и обучения модели для классификации объектов
        :return: Ничего не возвращает
        """
        self.visualisation_model(self.img_classification_container, self.class_canvas_img)

        file_data_name = self.file_name.get()
        if services.test_file(file_data_name):
            return
        library_name = self.use_library_classification.get()
        current_model = self.model_classification.get(library_name)

        if current_model is None or current_model.file_dataset != file_data_name:
            class_dictionary = CLASSIFICATION_DICTIONARY[library_name]
            dataset = services.load_dataset(file_data_name)
            if library_name in [x[1] for x in CLASSIFICATION_MODELS][2:]:
                dataset = services.dataset_for_torch(dataset, file_data_name)
                services.prepare_dataset(library_name, class_dictionary, dataset)

            else:
                dataset['class'] = dataset['class'].apply(lambda x: class_dictionary[x])

            x_train, y_train, x_test, y_test = services.get_sets(dataset,
                                                                 ['Products', 'Gas'],
                                                                 'class',
                                                                 library_name,
                                                                 size_train=0.9, size_test=0.1)

            if library_name == 'scikit_KNN':
                current_model = scmc.model_KNN(x_train, y_train, file_data_name)

            elif library_name == 'tensorflow':
                test_y = services.converter(y_test)
                current_model = ctm.Model_tensorflow(x_train, y_train, 4, file_data_name)

            elif library_name == 'pytorch':
                # Преобразуем в PyTorch tensors
                x_train = services.to_tensor(x_train, True)
                x_test = services.to_tensor(x_test, True)
                y_train = services.to_tensor(y_train, True, True)
                y_test = services.to_tensor(y_test, True, True)

                # Нормализация исходных данных
                mean = x_train.mean(dim=0)
                std = x_train.std(dim=0)
                x_train = (x_train - mean) / std
                x_ = x_test.clone()
                x_test = (x_test - mean) / std

                current_model = cpm.Model_torch(2, 4, x_train, y_train, file_data_name)

            else:
                current_model = scmc.model_SVC(x_train, y_train, file_data_name)

            prediction = current_model.predict(x_test)
            services.matrix(current_model, prediction, y_test)
            services.class_report(current_model, prediction, y_test)

            if library_name == 'tensorflow':
                services.visualize_classifier(current_model, x_test, test_y)
            elif library_name == 'pytorch':
                services.visualize_classifier(current_model, x_test, y_test, x_, mean, std)
            else:
                services.visualize_classifier(current_model, x_test, y_test)

            current_model.size_matrix = 4
            current_model.dataset = dataset
            current_model.class_dictionary = class_dictionary
            self.model_classification[library_name] = current_model

        file_out = f'img/{current_model.library}_classification.png'
        self.visualisation_model(self.img_classification_container, self.class_canvas_img, file_out)
        self.forgot_matrix(library_name)

        self.class_model_description.delete("1.0", END)
        self.class_model_description.insert("1.0", current_model.description_total)

    def forgot_matrix(self, model_name):
        """
        Процедура управления видимостью кнопки получения матрицы ошибок.
        Если модель создана и натренирована кнопка видна, в противном случае скрыта.
        :param model_name: Имя модели
        :return:
        """
        current_model = self.model_classification.get(model_name)
        if current_model is None:
            self.make_matrix_classification_button.pack_forget()
        else:
            self.make_matrix_classification_button.pack(side=LEFT, anchor='nw', padx=4, pady=6)

    def click_do_matrix(self):
        """
        Процедура отображения матрицы неточностей для выбранной модели.
        :return:
        """
        model_reg = self.model_regression.get(self.use_library_regress.get())

        model_name = self.use_library_classification.get()
        current_model = self.model_classification.get(model_name)
        if current_model is not None:
            services.plots_confusion(f"Матрица неточности {current_model.library}",
                                     current_model.matrix_of_inaccuracies, current_model.size_matrix)

            file_out = f'img/{current_model.library}_classification.png'
            self.visualisation_model(self.img_classification_container, self.class_canvas_img, file_out)

            if model_reg is not None:
                file_out = f'img/{model_reg.library}_regression.png'
                self.visualisation_model(self.img_regress_container, self.canvas_img, file_out)


if __name__ == '__main__':
    application = Main_Window("Классификация и регрессия", "1200x595")
    application.mainloop()
