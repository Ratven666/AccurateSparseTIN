import matplotlib.pyplot as plt

from CONFIG import MAX_POINT_SCAN_PLOT
from utils.scan_utils.scan_plotters.ScanPlotterABC import ScanPlotterABC
from utils.scan_utils.scan_samplers.TotalPointCountScanSampler import TotalPointCountScanSampler


class ScanScanPlotterMPL(ScanPlotterABC):
    """
    Отрисовка скана в виде облака точек через библиотеку matplotlib
    """
    def __init__(self, sampler=TotalPointCountScanSampler(MAX_POINT_SCAN_PLOT), point_size=5):
        super().__init__()
        self.__fig = None
        self.__ax = None
        self.__sampler = sampler
        self.__point_size = point_size

    def __set_plot_limits(self, plot_limits):
        """
        Устанавливает в графике области построения в соответствии с переданными значениями
        :param plot_limits: словарь с границами области построения
        :return: None
        """
        try:
            self.__ax.set_xlim(*plot_limits["X_lim"])
            self.__ax.set_ylim(*plot_limits["Y_lim"])
            self.__ax.set_zlim(*plot_limits["Z_lim"])
        except TypeError:
            self.logger.critical(f"Переданны некорректные лимиты области модели.")

    @staticmethod
    def __calk_mpl_colors(plot_data):
        """
        пересчитывает данные цвета точек в формат для библиотеки matplotlib
        :param plot_data: данные для которых нужно пересчитать цвета
        :return: Список с цветами в формате для библиотеки matplotlib
        """
        c_lst = []
        for point_color in plot_data["color"]:
            color = [color / 255.0 for color in point_color]
            c_lst.append(color)
        return c_lst

    def plot(self, scan):
        """
        Отрисовка скана в 3D
        :param scan: скан который будет отрисовываться
        :return: None
        """
        self.__fig = plt.figure()
        self.__ax = self.__fig.add_subplot(projection="3d")

        plot_data = self.get_sample_data(scan)
        c_lst = self.__calk_mpl_colors(plot_data)
        self.__set_plot_limits(self.calk_plot_limits(scan))
        self.__ax.scatter(plot_data["x"], plot_data["y"], plot_data["z"],
                          c=c_lst, marker="+", s=self.__point_size)
        plt.show()
