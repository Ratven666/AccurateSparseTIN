import logging
from abc import ABC, abstractmethod

from CONFIG import LOGGER


class SerializerABC(ABC):
    logger = logging.getLogger(LOGGER)

    def __init__(self, data):
        self.data = data

    @abstractmethod
    def dump(self):
        pass

    @staticmethod
    @abstractmethod
    def load():
        pass
