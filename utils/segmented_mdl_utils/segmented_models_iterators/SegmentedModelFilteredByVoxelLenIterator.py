import sys
import threading


class SegmentedModelFilteredByVoxelLenIterator:
    """
    Итератор сегментированных моделей,
    возвращающий только те ячейки количество точек в которых преывшает заданный
    в min_voxel_len порог
    """

    def __init__(self, segmented_model):
        self.__model = segmented_model
        self.data = [cell for cell in self.__model._model_structure.values()]
        self.idx = 0

    def __iter__(self):
        return self

    # def __next__(self):
    #     if self.idx >= len(self.data):
    #         raise StopIteration
    #     cell = self.data[self.idx]
    #     self.idx += 1
    #     if cell.voxel.len <= self.__model.min_voxel_len:
    #         self.__next__()
    #     return cell

    def __next__(self):
        if self.idx >= len(self.data):
            raise StopIteration
        for cell in self.data[self.idx::]:
            self.idx += 1
            if cell.voxel.len <= self.__model.min_voxel_len:
                continue
            return cell
