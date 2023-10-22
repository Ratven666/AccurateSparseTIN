import logging
from abc import ABC, abstractmethod

from CONFIG import LOGGER


class ScanSaverABC(ABC):
    logger = logging.getLogger(LOGGER)

    @abstractmethod
    def save_scan(self, scan, file_name):
        pass
