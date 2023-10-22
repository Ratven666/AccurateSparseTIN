import logging
from abc import ABC, abstractmethod

from CONFIG import MAX_POINT_SCAN_PLOT, LOGGER
from utils.scan_utils.scan_plotters.ScanPlotterMPL import ScanScanPlotterMPL
from utils.scan_utils.scan_samplers.TotalPointCountScanSampler import TotalPointCountScanSampler
from utils.scan_utils.scan_savers.ScanTXTSaver import ScanTXTSaver


class ScanABC(ABC):
    """
    Абстрактный класс скана
    """
    logger = logging.getLogger(LOGGER)

    def __init__(self, scan_name):
        self.id = None
        self.scan_name: str = scan_name
        self.len: int = 0
        self.min_X, self.max_X = None, None
        self.min_Y, self.max_Y = None, None
        self.min_Z, self.max_Z = None, None

    def __str__(self):
        return f"{self.__class__.__name__} "\
               f"[id: {self.id},\tName: {self.scan_name}\tLEN: {self.len}]"

    def __repr__(self):
        return f"{self.__class__.__name__} [ID: {self.id}]"

    def __len__(self):
        return self.len

    @abstractmethod
    def __iter__(self):
        pass

    def save_scan_in_file(self, file_name=None, scan_saver=ScanTXTSaver()):
        scan_saver.save_scan(self, file_name)

    def plot(self, plotter=ScanScanPlotterMPL(sampler=TotalPointCountScanSampler(MAX_POINT_SCAN_PLOT))):
        """
        Вывод отображения скана
        :param plotter: объект определяющий логику отображения скана
        :return: None
        """
        plotter.plot(self)
