from abc import abstractmethod

import plotly.graph_objects as go
import numpy as np
from scipy.spatial import Delaunay

from CONFIG import MAX_POINT_SCAN_PLOT
from utils.scan_utils.scan_plotters.ScanPlotterABC import ScanPlotterABC
from utils.scan_utils.scan_samplers.TotalPointCountScanSampler import TotalPointCountScanSampler


class ScanPlotterPlotly(ScanPlotterABC):
    """
    Абстрактный класс для плоттера сканов в библиотеке plotly
    """
    def __init__(self, sampler=TotalPointCountScanSampler(MAX_POINT_SCAN_PLOT)):
        super().__init__(sampler)

    def set_plot_limits(self, scan, fig):
        """
        Устанавливает в графике области построения для сохранения пропорций вдоль осей
        :param scan: скан, который будет визуализироваться
        :param fig: объект фигуры для которого будет задаваться настройка осей
        :return: None
        """
        pl = self.calk_plot_limits(scan)
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=pl["X_lim"]),
                yaxis=dict(range=pl["Y_lim"]),
                zaxis=dict(range=pl["Z_lim"])),
            margin=dict(r=10, l=10, b=10, t=10))

    @abstractmethod
    def plot(self, scan):
        pass


class ScanPlotterPointsPlotly(ScanPlotterPlotly):
    """
    Отрисовка скана в виде облака точек через библиотеку plotly
    """
    def __init__(self, sampler=TotalPointCountScanSampler(MAX_POINT_SCAN_PLOT), point_size=5):
        super().__init__(sampler)
        self.__point_size = point_size

    def plot(self, scan):
        """
        Запускает процедуру визуализации облака точек
        1. Разряжает исходное облако
        2. Загружает оставшиеся точки в область построения fig
        3. Рассчитывает область построения графика и применяет их к области fig
        4. Запускает отображение построенных данных
        :param scan: скан, который визуализируем
        :return: None
        """
        plot_data = self.get_sample_data(scan)
        fig = go.Figure(data=[go.Scatter3d(x=plot_data["x"],
                                           y=plot_data["y"],
                                           z=plot_data["z"],
                                           mode="markers",
                                           marker=dict(
                                               size=self.__point_size,
                                               color=plot_data["z"],
                                               opacity=1,
                                               colorscale="Rainbow"
                                           )
                                           )])
        self.set_plot_limits(scan, fig)
        fig.show()


class ScanPlotterMeshPlotly(ScanPlotterPlotly):
    """
    Отрисовка скана в виде триангуляции Делоне через библиотеку plotly
    """
    def __init__(self, sampler=TotalPointCountScanSampler(MAX_POINT_SCAN_PLOT)):
        super().__init__(sampler)

    @staticmethod
    def __calk_delone_triangulation(plot_data):
        """
        Рассчитываает треугольники между точками
        :param plot_data: Данные для которых рассчитывается триангуляция
        :return: славарь с указанием вершин треугольников
        """
        points2D = np.vstack([plot_data["x"], plot_data["y"]]).T
        tri = Delaunay(points2D)
        i_lst, j_lst, k_lst = ([triplet[c] for triplet in tri.simplices] for c in range(3))
        return {"i_lst": i_lst, "j_lst": j_lst, "k_lst": k_lst}

    @staticmethod
    def __calk_faces_colors(ijk_dict, plot_data):
        """
        Рассчитывает цвета треугольников на основании усреднения цветов точек, образующих
        треугольник
        :param ijk_dict: словарь с вершинами треугольников
        :param plot_data: словарь с данными о точках отриовываемой модели
        :return: список цветов треугольников в формате библиотеки plotly
        """
        c_lst = []
        for idx in range(len(ijk_dict["i_lst"])):
            c_i = plot_data["color"][ijk_dict["i_lst"][idx]]
            c_j = plot_data["color"][ijk_dict["j_lst"][idx]]
            c_k = plot_data["color"][ijk_dict["k_lst"][idx]]
            r = round((c_i[0] + c_j[0] + c_k[0]) / 3)
            g = round((c_i[1] + c_j[1] + c_k[1]) / 3)
            b = round((c_i[2] + c_j[2] + c_k[2]) / 3)
            c_lst.append(f"rgb({r}, {g}, {b})")
        return c_lst

    def plot(self, scan):
        """
        Запускает процедуру визуализации триангуляционной поверхности
        1. Разряжает исходное облако
        2. Рассчитываем связи между точками в треугольники
        3. Рассчитываем цвета треугольников
        4. Загружает отреугольники в область построения fig
        5. Рассчитывает область построения графика и применяет их к области fig
        6. Запускает отображение построенных данных
        :param scan: скан, который визуализируем
        :return: None
        """
        plot_data = self.get_sample_data(scan)

        ijk_dict = self.__calk_delone_triangulation(plot_data)
        c_lst = self.__calk_faces_colors(ijk_dict, plot_data)

        fig = go.Figure(data=[go.Mesh3d(x=plot_data["x"],
                                        y=plot_data["y"],
                                        z=plot_data["z"],
                                        i=ijk_dict["i_lst"],
                                        j=ijk_dict["j_lst"],
                                        k=ijk_dict["k_lst"],
                                        opacity=1,
                                        facecolor=c_lst,
                                        )])
        self.set_plot_limits(scan, fig)
        fig.show()
