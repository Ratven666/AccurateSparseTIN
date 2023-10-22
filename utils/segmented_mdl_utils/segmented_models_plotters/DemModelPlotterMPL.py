import matplotlib.pyplot as plt
import numpy as np


class DemModelPlotterMPL:

    def __init__(self, alpha=1.0):
        self.dem_model = None
        self.__alpha = alpha
        self.__fig = None
        self.__ax = None

    def __calk_plot_limits(self):
        """
        Расчитывает область построения графика
        :return: Словарь с пределами построения графика по осям
        """
        min_x, min_y, min_z = (self.dem_model.voxel_model.min_X,
                               self.dem_model.voxel_model.min_Y,
                               self.dem_model.voxel_model.min_Z)
        max_x, max_y, max_z = (self.dem_model.voxel_model.max_X,
                               self.dem_model.voxel_model.max_Y,
                               self.dem_model.voxel_model.max_Z)
        limits = [max_x - min_x,
                  max_y - min_y,
                  max_z - min_z]
        length = max(limits) / 2
        x_lim = [((min_x + max_x) / 2) - length, ((min_x + max_x) / 2) + length]
        y_lim = [((min_y + max_y) / 2) - length, ((min_y + max_y) / 2) + length]
        z_lim = [((min_z + max_z) / 2) - length, ((min_z + max_z) / 2) + length]
        return {"X_lim": x_lim, "Y_lim": y_lim, "Z_lim": z_lim}

    def __set_plot_limits(self, plot_limits):
        """
        Устанавливает в графике области построения в соответствии с переданными значениями
        :param plot_limits: словарь с границами области построения
        :return: None
        """
        self.__ax.set_xlim(*plot_limits["X_lim"])
        self.__ax.set_ylim(*plot_limits["Y_lim"])
        self.__ax.set_zlim(*plot_limits["Z_lim"])

    @staticmethod
    def __calk_mpl_colors(cell):
        """
        Пересчитывает данные цвета точек в формат для библиотеки matplotlib
        :param cell: ячейка модели для которой нужно пересчитать цвет
        :return: Список с цветами в формате для библиотеки matplotlib
        """
        color = [cell.voxel.R / 255.0,
                 cell.voxel.G / 255.0,
                 cell.voxel.B / 255.0]
        return color

    def plot(self, dem_model):
        """
        Отрисовка скана в 3D
        :param dem_model: dem модель которая будет отрисовываться
        :return: None
        """
        self.dem_model = dem_model
        self.__fig = plt.figure()
        self.__ax = self.__fig.add_subplot(111, projection="3d")

        for dem_cell in self.dem_model:
            if dem_cell is None:
                continue
            x_min, y_min = dem_cell.voxel.X, dem_cell.voxel.Y
            x_max, y_max = dem_cell.voxel.X + 2*dem_cell.voxel.step, dem_cell.voxel.Y + 2*dem_cell.voxel.step
            X = np.arange(x_min, x_max, dem_cell.voxel.step)
            Y = np.arange(y_min, y_max, dem_cell.voxel.step)
            X, Y = np.meshgrid(X, Y)
            Z = 0 * X + 0 * Y + dem_cell.avr_z
            color = self.__calk_mpl_colors(dem_cell)
            self.__ax.plot_surface(X, Y, Z, color=color, alpha=self.__alpha)
        self.__set_plot_limits(self.__calk_plot_limits())
        plt.show()
