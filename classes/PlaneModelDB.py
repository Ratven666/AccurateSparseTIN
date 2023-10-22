import numpy as np

from classes.PlaneCellDB import PlaneCellDB
from classes.ScanDB import ScanDB
from classes.abc_classes.SegmentedModelABC import SegmentedModelABC
from db_models.dem_models_table import DemTypeEnum


class PlaneModelDB(SegmentedModelABC):
    """
    PLANE  модель связанная с базой данных
    """

    def __init__(self, voxel_model):
        self.model_type = DemTypeEnum.PLANE
        self.model_name = f"{self.model_type.name}_from_{voxel_model.vm_name}"
        self.mse_data = None
        self.cell_type = PlaneCellDB
        super().__init__(voxel_model, self.cell_type)

    def _calk_segment_model(self):
        """
        Метод определяющий логику создания плоскостной модели
        :return: None
        """
        self.logger.info(f"Начат расчет модели {self.model_name}")
        base_scan = ScanDB.get_scan_from_id(self.voxel_model.base_scan_id)
        self.__fit_planes(base_scan)
        self.logger.info(f"Рассчитаны плоскости в модели {self.model_name}")
        self._calk_cell_mse(base_scan)
        self.__calk_abd_mse()
        self.logger.info(f"Рассчитаны СКП модели {self.model_name}")

    def __fit_planes(self, base_scan):
        """
        Расчитывает параметры аппроксимирующих плоскостей по МНК
        :param base_scan: базовй кан воксельной модели
        :return: None
        """
        self.__calk_matrix_params(base_scan)
        for cell in self:
            try:
                m_n = np.array([[cell.a1, cell.b1, cell.c1],
                                [cell.b1, cell.b2, cell.c2],
                                [cell.c1, cell.c2, cell.c3]])
                m_d = np.array([cell.d1, cell.d2, cell.d3])
            except AttributeError:
                continue
            try:
                abc = np.linalg.solve(m_n, m_d)
            except np.linalg.LinAlgError:
                cell.r = -1
                continue
            cell.a = float(abc[0])
            cell.b = float(abc[1])
            cell.d = float(abc[2])
        self.logger.info(f"Расчет параметров вписываемых плоскостей модели {self.model_name} завершен")

    def __calk_abd_mse(self):
        for cell in self:
            if cell.r <= 0:
                continue
            try:
                m_n = np.array([[cell.a1, cell.b1, cell.c1],
                                [cell.b1, cell.b2, cell.c2],
                                [cell.c1, cell.c2, cell.c3]])
            except AttributeError:
                continue
            try:
                Q = np.linalg.inv(m_n)
                D = np.diagonal(Q)
                ABD_mse = np.sqrt(D) * cell.mse
                cell.m_a, cell.m_b, cell.m_d = float(ABD_mse[0]), float(ABD_mse[1]), float(ABD_mse[2])
            except np.linalg.LinAlgError:
                continue
            except TypeError:
                continue

    def __calk_matrix_params(self, base_scan):
        """
        Расчитывает коэфициенты матрицы нормальных уравнений
        :param base_scan: базовый скан воксельной модели
        :return: None
        """
        for point in base_scan:
            cell = self.get_model_element_for_point(point)
            try:
                cell.a1 += point.X ** 2
                cell.b1 += point.X * point.Y
                cell.c1 += point.X
                cell.b2 += point.Y ** 2
                cell.c2 += point.Y
                cell.c3 += 1
                cell.d1 += point.X * point.Z
                cell.d2 += point.Y * point.Z
                cell.d3 += point.Z
            except AttributeError:
                cell.a1 = point.X ** 2
                cell.b1 = point.X * point.Y
                cell.c1 = point.X
                cell.b2 = point.Y ** 2
                cell.c2 = point.Y
                cell.c3 = 1
                cell.d1 = point.X * point.Z
                cell.d2 = point.Y * point.Z
                cell.d3 = point.Z
        self.logger.info(f"Рассчет элементов матрицы нормальных коэфициентов модели {self.model_name} завершен")
