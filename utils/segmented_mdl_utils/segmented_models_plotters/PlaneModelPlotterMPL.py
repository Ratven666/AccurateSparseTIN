import matplotlib.pyplot as plt
import numpy as np


class PlaneModelPlotterMPL:

    def __init__(self, alpha=1.0):
        self.model = None
        self.__alpha = alpha
        self.__fig = None
        self.__ax = None

    def __calk_plot_limits(self):
        min_x, min_y, min_z = (self.model.voxel_model.min_X,
                               self.model.voxel_model.min_Y,
                               self.model.voxel_model.min_Z)
        max_x, max_y, max_z = (self.model.voxel_model.max_X,
                               self.model.voxel_model.max_Y,
                               self.model.voxel_model.max_Z)
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
        пересчитывает данные цвета точек в формат для библиотеки matplotlib
        :param plot_data: данные для которых нужно пересчитать цвета
        :return: Список с цветами в формате для библиотеки matplotlib
        """
        color = [cell.voxel.R / 255.0,
                 cell.voxel.G / 255.0,
                 cell.voxel.B / 255.0]
        return color

    def plot(self, plane_model):
        """
        Отрисовка скана в 3D
        :param scan: скан который будет отрисовываться
        :return: None
        """
        self.model = plane_model
        self.__fig = plt.figure()
        self.__ax = self.__fig.add_subplot(111, projection="3d")

        for cell in self.model:
            if cell is None:
                continue
            x_min, y_min = cell.voxel.X, cell.voxel.Y
            x_max, y_max = cell.voxel.X + 2*cell.voxel.step, cell.voxel.Y + 2*cell.voxel.step
            X = np.arange(x_min, x_max, cell.voxel.step)
            Y = np.arange(y_min, y_max, cell.voxel.step)
            X, Y = np.meshgrid(X, Y)
            try:
                Z = cell.a * X + cell.b * Y + cell.d
            except TypeError:
                continue
            color = self.__calk_mpl_colors(cell)
            self.__ax.plot_surface(X, Y, Z, color=color, alpha=self.__alpha)
        self.__set_plot_limits(self.__calk_plot_limits())
        plt.show()
