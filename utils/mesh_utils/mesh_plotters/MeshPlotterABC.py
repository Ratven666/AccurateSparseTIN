import logging
from abc import ABC, abstractmethod

from CONFIG import LOGGER


class MeshPlotterABC(ABC):
    """
    Абстрактный класс плоттера скана
    """
    logger = logging.getLogger(LOGGER)

    def __str__(self):
        return f"Плоттер типа: {self.__class__.__name__}"

    @abstractmethod
    def plot(self, mesh):
        pass
