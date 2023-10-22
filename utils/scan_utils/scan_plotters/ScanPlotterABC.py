import logging
from abc import ABC, abstractmethod

from CONFIG import MAX_POINT_SCAN_PLOT, LOGGER
from utils.scan_utils.scan_samplers.TotalPointCountScanSampler import TotalPointCountScanSampler


class ScanPlotterABC(ABC):
    """
    Абстрактный класс плоттера скана
    """
    logger = logging.getLogger(LOGGER)

    def __init__(self, sampler=TotalPointCountScanSampler(MAX_POINT_SCAN_PLOT)):
        self.__sampler = sampler

    def __str__(self):
        return f"Плоттер типа: {self.__class__.__name__}"

    @abstractmethod
    def plot(self, scan):
        pass

    def calk_plot_limits(self, scan):
        """
        Рассчитывает область построения скана для сохранения пропорций вдоль осей
        :param scan: скан который будет отрисовываться
        :return: Словарь с пределами построения модель вдоль трех осей
        """
        min_x, min_y, min_z = scan.min_X, scan.min_Y, scan.min_Z
        max_x, max_y, max_z = scan.max_X, scan.max_Y, scan.max_Z
        try:
            limits = [max_x - min_x,
                      max_y - min_y,
                      max_z - min_z]
        except TypeError:
            self.logger.critical(f"Не рассчитаны значения границ в скане \"{scan.scan_name}\". "
                                   f"Невозможно рассчитать область построения модели.")
            return
        length = max(limits) / 2
        x_lim = [((min_x + max_x) / 2) - length, ((min_x + max_x) / 2) + length]
        y_lim = [((min_y + max_y) / 2) - length, ((min_y + max_y) / 2) + length]
        z_lim = [((min_z + max_z) / 2) - length, ((min_z + max_z) / 2) + length]
        return {"X_lim": x_lim, "Y_lim": y_lim, "Z_lim": z_lim}

    def get_sample_data(self, scan):
        """
        Запускает процедуру разряжения скана если задан атрибут __sampler
        возвращает словарь с данными для визуализации разреженных данных
        :param scan: скан, который требуется разрядить и подготовить к визуализации
        :return: словарь с данными для визуализации
        """
        if self.__sampler is not None:
            scan = self.__sampler.do_sampling(scan)
        x_lst, y_lst, z_lst, c_lst = [], [], [], []
        for point in scan:
            x_lst.append(point.X)
            y_lst.append(point.Y)
            z_lst.append(point.Z)
            c_lst.append([point.R, point.G, point.B])
        return {"x": x_lst, "y": y_lst, "z": z_lst, "color": c_lst, "scan": scan}
