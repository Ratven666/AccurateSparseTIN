import matplotlib.pyplot as plt
import numpy as np


class VoxelModelPlotter:
    """
    Визуализирует воксельную модель через библиотеку matplotlib
    """
    def __init__(self):
        self.voxel_model = None

    def plot(self, voxel_model):
        """
        Визуализирует воксельную модель средствами бибилиотеки matplotlib
        :param voxel_model: визуализируемая модель
        :return: None
        """
        self.voxel_model = voxel_model
        area = np.full((self.voxel_model.X_count,
                        self.voxel_model.Y_count,
                        self.voxel_model.Z_count), False)

        colors = np.zeros(area.shape + (3,))

        for voxel in self.voxel_model:
            i, j, k = self.__calk_indexes(voxel)
            area[i][j][k] = True
            colors[i][j][k][0] = voxel.R / 255.0
            colors[i][j][k][1] = voxel.G / 255.0
            colors[i][j][k][2] = voxel.B / 255.0

        ax = plt.figure().add_subplot(projection='3d')
        self.__set_plot_limits(ax, self.__calk_plot_limits(self.voxel_model))
        ax.voxels(area, facecolors=colors)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        self.__set_ax_ticks(ax, self.__calk_ax_ticks())
        plt.show()

    def __calk_ax_ticks(self):
        """
        Рассчитывает корректные подписи по осям модели на основе координат вокселя
        :return: словарь с полдписями для осей
        """
        x_ticks = [self.voxel_model.min_X + idx*self.voxel_model.step for idx in range(self.voxel_model.X_count + 1)]
        y_ticks = [self.voxel_model.min_Y + idx*self.voxel_model.step for idx in range(self.voxel_model.Y_count + 1)]
        z_ticks = [self.voxel_model.min_Z + idx*self.voxel_model.step for idx in range(self.voxel_model.Z_count + 1)]
        return {"x_ticks": x_ticks, "y_ticks": y_ticks, "z_ticks": z_ticks}

    @staticmethod
    def __set_ax_ticks(ax, ax_ticks: dict):
        """
        Применяет подписи для осей в графике для осей ax
        :param ax: график в котором строится модель
        :param ax_ticks: новые подписи осей
        :return: None
        """
        ax.set_xticks(np.arange(len(ax_ticks["x_ticks"])), labels=ax_ticks["x_ticks"])
        ax.set_yticks(np.arange(len(ax_ticks["y_ticks"])), labels=ax_ticks["y_ticks"])
        ax.set_zticks(np.arange(len(ax_ticks["z_ticks"])), labels=ax_ticks["z_ticks"])

    def __calk_indexes(self, voxel):
        """
        Рассчитывает индексы вокселя внутри модели по трем осям
        на основании координат вокселя, его размера и области модели
        :param voxel: воксель для которого выполняется расчет индекса
        :return: кортеж с индексами вокселя
        """
        x0 = self.voxel_model.min_X
        y0 = self.voxel_model.min_Y
        z0 = self.voxel_model.min_Z
        i = int((voxel.X - x0) / self.voxel_model.step)
        j = int((voxel.Y - y0) / self.voxel_model.step)
        k = int((voxel.Z - z0) / self.voxel_model.step)
        return i, j, k

    @staticmethod
    def __calk_plot_limits(vxl_mdl):
        """
        Рассчитывает область построения воксельной модели для сохранения пропорций вдоль осей
        :param vxl_mdl: вокселььная модель которая будет отрисовываться
        :return: Словарь с пределами построения модели вдоль трех осей
        """
        limits = [vxl_mdl.X_count,
                  vxl_mdl.Y_count,
                  vxl_mdl.Z_count]
        length = max(limits)
        x_lim = [0, length]
        y_lim = [0, length]
        z_lim = [0, length]
        return {"X_lim": x_lim, "Y_lim": y_lim, "Z_lim": z_lim}

    @staticmethod
    def __set_plot_limits(ax, plot_limits):
        """
        Устанавливает в графике области построения в соответствии с переданными значениями
        :param plot_limits: словарь с границами области построения
        :return: None
        """
        ax.set_xlim(*plot_limits["X_lim"])
        ax.set_ylim(*plot_limits["Y_lim"])
        ax.set_zlim(*plot_limits["Z_lim"])

