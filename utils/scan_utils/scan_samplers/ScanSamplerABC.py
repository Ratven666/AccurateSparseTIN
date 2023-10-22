from abc import ABC, abstractmethod


class ScanSamplerABC(ABC):
    """
    Абстрактный разрядитель плотности точек в скане
    """
    @abstractmethod
    def __init__(self):
        pass

    def __str__(self):
        return f"Разредитель точек типа: {self.__class__.__name__}"

    @abstractmethod
    def do_sampling(self, scan):
        """
        Запускает процедуру разрежения плотности облака точек в скане
        :param scan: скан который требуется разредить
        :return: разреженный скан типа ScanLite (в оперативной памяти)
        """
        pass
