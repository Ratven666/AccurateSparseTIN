from abc import ABC, abstractmethod


class MeshFilterABC(ABC):

    def __init__(self):
        self.triangulator = None

    def filtrate(self, triangulator):
        self.triangulator = triangulator
        self._filter_logic()

    @abstractmethod
    def _filter_logic(self):
        pass
