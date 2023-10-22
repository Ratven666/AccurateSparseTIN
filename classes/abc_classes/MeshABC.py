import logging
from abc import abstractmethod

from CONFIG import LOGGER, VOXEL_IN_VM
from classes.MeshSegmentModelDB import MeshSegmentModelDB
from classes.ScanDB import ScanDB
from classes.VoxelModelDB import VoxelModelDB
from utils.mesh_utils.mesh_plotters.MeshPlotterPlotly import MeshPlotterPlotly
from utils.mesh_utils.mesh_triangulators.ScipyTriangulator import ScipyTriangulator


class MeshABC:
    logger = logging.getLogger(LOGGER)

    def __init__(self, scan, scan_triangulator=ScipyTriangulator):
        self.id = None
        self.scan = scan
        self.scan_triangulator = scan_triangulator
        self.mesh_name = self.__name_generator()
        self.len = 0
        self.r = None
        self.mse = None

    @abstractmethod
    def __iter__(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"[mesh_name: {self.mesh_name},\tlen: {self.len} r: {self.r} mse: {self.mse}]"

    def __len__(self):
        return self.len

    def __name_generator(self):
        return f"MESH_{self.scan.scan_name}"

    @abstractmethod
    def clear_mesh_mse(self):
        """
        Удаляет данные о СКП и степенях свободы из поверхности
        """
        pass

    def calk_mesh_mse(self, base_scan, voxel_size=None,
                      clear_previous_mse=False,
                      delete_temp_models=False):
        """
        Рассчитывает СКП поверхности и ее треугольников относительно базового скана
        :param base_scan: базовый плотный  скан относительно которого считаются ошибки
        :param voxel_size: размер ячейки вспомогательной воксельной модели
        :param clear_previous_mse: предварительно удаление ранее рассчитаныых значений СКП и r
        :param delete_temp_models: удаление промежуточных моделей
        :return: список треугоьлников поверхности
        """
        if clear_previous_mse:
            self.clear_mesh_mse()
            self.r = None
            self.mse = None
        if self.mse is not None:
            return None
        if voxel_size is None:
            # area = (base_scan.max_X - base_scan.min_X) * (base_scan.max_Y - base_scan.min_Y)
            # voxel_size = area / self.scan.len
            # voxel_size = round((voxel_size // 0.05 + 1) * 0.05, 2)
            voxel_size = VoxelModelDB.get_step_by_voxel_count(base_scan, VOXEL_IN_VM,
                                                              is_2d_vxl_mdl=True,
                                                              round_n=2)
        vm = VoxelModelDB(base_scan, voxel_size, is_2d_vxl_mdl=True)
        mesh_segment_model = MeshSegmentModelDB(vm, self)
        triangles = {}
        for point in base_scan:
            point.id = None
            cell = mesh_segment_model.get_model_element_for_point(point)
            if cell is None or len(cell.triangles) == 0:
                continue
            for triangle in cell.triangles:
                if triangle.is_point_in_triangle(point):
                    if point not in [triangle.point_0, triangle.point_1, triangle.point_2]:
                        try:
                            triangle.vv += (triangle.get_z_from_xy(point.X, point.Y) - point.Z) ** 2
                            triangle.r += 1
                        except (AttributeError, TypeError):
                            z = triangle.get_z_from_xy(point.X, point.Y)
                            if z is not None:
                                triangle.vv = (z - point.Z) ** 2
                                triangle.r = 1
                    triangles[triangle.id] = triangle
                    break
        for triangle in triangles.values():
            if triangle.r is not None:
                try:
                    triangle.mse = (triangle.vv / triangle.r) ** 0.5
                except AttributeError:
                    triangle.mse = None
                    triangle.r = None
        svv, sr = 0, 0
        for triangle in triangles.values():
            try:
                svv += triangle.r * triangle.mse ** 2
                sr += triangle.r
            except TypeError:
                continue
        self.mse = (svv / sr) ** 0.5
        self.r = sr
        if delete_temp_models:
            vm.delete_model()
            mesh_segment_model.delete_model()
        return triangles.values()

    def plot(self, plotter=MeshPlotterPlotly()):
        """
        Вывод отображения скана
        :param plotter: объект определяющий логику отображения поверхности
        :return: None
        """
        plotter.plot(self)
