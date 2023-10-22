import logging
from abc import ABC, abstractmethod

from CONFIG import LOGGER
from utils.voxel_utils.voxel_model_plotters.Voxel_model_plotter import VoxelModelPlotter


class VoxelModelABC(ABC):
    """
    Абстрактный класс воксельной модели
    """
    logger = logging.getLogger(LOGGER)

    def __init__(self, scan, step, dx, dy, dz, is_2d_vxl_mdl=True):
        self.id = None
        self.is_2d_vxl_mdl = is_2d_vxl_mdl
        self.step = float(step)
        self.dx, self.dy, self.dz = self.__dx_dy_dz_formatter(dx, dy, dz)
        self.vm_name: str = self.__name_generator(scan)
        self.len: int = 0
        self.X_count, self.Y_count, self.Z_count = None, None, None
        self.min_X, self.max_X = None, None
        self.min_Y, self.max_Y = None, None
        self.min_Z, self.max_Z = None, None
        self.base_scan_id = None
        self.base_scan = scan

    @staticmethod
    def __dx_dy_dz_formatter(dx, dy, dz):
        """
        Приводит значения смещения воксельной модели в пределы от 0 до 1
        """
        return dx % 1, dy % 1, dz % 1

    def __name_generator(self, scan):
        """
        Конструктор имени воксельной модели
        :param scan: базовый скан, по которому создается модель
        :return: None
        """
        vm_type = "2D" if self.is_2d_vxl_mdl else "3D"
        return f"VM_{vm_type}_Sc:{scan.scan_name}_st:{self.step}_dx:{self.dx:.2f}_dy:{self.dy:.2f}_dz:{self.dz:.2f}"

    def _calc_vxl_md_metric(self, scan):
        """
        Рассчитывает границы воксельной модели и максимальное количество вокселей
        исходя из размера вокселя и границ скана
        :param scan: скан на основе которого рассчитываются границы модели
        :return: None
        """
        if len(scan) == 0:
            return None
        self.min_X = (scan.min_X // self.step * self.step) - ((1 - self.dx) % 1 * self.step)
        self.min_Y = (scan.min_Y // self.step * self.step) - ((1 - self.dy) % 1 * self.step)
        self.min_Z = (scan.min_Z // self.step * self.step) - ((1 - self.dz) % 1 * self.step)

        self.max_X = (scan.max_X // self.step + 1) * self.step + ((self.dx % 1) * self.step)
        self.max_Y = (scan.max_Y // self.step + 1) * self.step + ((self.dy % 1) * self.step)
        self.max_Z = (scan.max_Z // self.step + 1) * self.step + ((self.dz % 1) * self.step)

        self.X_count = round((self.max_X - self.min_X) / self.step)
        self.Y_count = round((self.max_Y - self.min_Y) / self.step)
        if self.is_2d_vxl_mdl:
            self.Z_count = 1
        else:
            self.Z_count = round((self.max_Z - self.min_Z) / self.step)
        self.len = self.X_count * self.Y_count * self.Z_count

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"[id: {self.id},\tName: {self.vm_name}\tLEN: (x:{self.X_count} * y:{self.Y_count} *" \
               f" z:{self.Z_count})={self.len}]"

    def __repr__(self):
        return f"{self.__class__.__name__} [ID: {self.id}]"

    def __len__(self):
        return self.len

    @abstractmethod
    def __iter__(self):
        pass

    def plot(self, plotter=VoxelModelPlotter()):
        """
        Вывод отображения воксельной модели
        :param plotter: объект определяющий логику отображения модели
        :return: None
        """
        plotter.plot(self)

    @classmethod
    def get_step_by_voxel_count(cls, scan, voxel_count, is_2d_vxl_mdl=True, round_n=2):
        x_len = scan.max_X - scan.min_X
        y_len = scan.max_Y - scan.min_Y
        z_len = scan.max_Z - scan.min_Z
        if is_2d_vxl_mdl:
            area = x_len * y_len
            cell_area = area / voxel_count
            step = round(cell_area ** 0.5, round_n)
        else:
            volume = x_len * y_len * z_len
            cell_volume = volume / voxel_count
            step = round(cell_volume ** (1 / 3), round_n)
        return step

