from classes.Line import Line
from classes.MeshCellDB import MeshCellDB
from classes.Point import Point
from classes.ScanDB import ScanDB
from classes.abc_classes.SegmentedModelABC import SegmentedModelABC


class MeshSegmentModelDB(SegmentedModelABC):

    def __init__(self, voxel_model, mesh):
        self.model_type = "MESH"
        self.model_name = f"{self.model_type}_from_{voxel_model.vm_name}"
        self.mse_data = None
        self.cell_type = MeshCellDB
        self.mesh = mesh
        super().__init__(voxel_model, self.cell_type)

    def _calk_segment_model(self):
        """
        Метод определяющий логику создания конкретной модели
        :return: None
        """
        pass

    def _calk_cell_mse(self, base_scan):
        pass

    def _create_model_structure(self, element_class):
        """
        Создание структуры сегментированной модели
        :param element_class: Класс ячейки конкретной модели
        :return: None
        """
        for voxel in self.voxel_model:
            model_key = f"{voxel.X:.5f}_{voxel.Y:.5f}_{voxel.Z:.5f}"
            self._model_structure[model_key] = element_class(voxel, self)
        self._sort_points_and_triangles_to_cells()

    def _sort_points_and_triangles_to_cells(self):
        for triangle in self.mesh:
            lines = [Line(triangle.point_0, triangle.point_1),
                     Line(triangle.point_1, triangle.point_2),
                     Line(triangle.point_2, triangle.point_0)]
            for line in lines:
                cross_points = line.get_grid_cross_points_list(self.voxel_model.step)
                mid_points = []
                for idx in range(len(cross_points) - 1):
                    mid_x = (cross_points[idx].X + cross_points[idx + 1].X) / 2
                    mid_y = (cross_points[idx].Y + cross_points[idx + 1].Y) / 2
                    mid_points.append(Point(X=mid_x, Y=mid_y, Z=0, R=0, G=0, B=0))
                for point in mid_points:
                    cell = self.get_model_element_for_point(point)
                    if cell is None:
                        continue
                    if triangle not in cell.triangles:
                        cell.triangles.append(triangle)
                        cell.count_of_triangles += 1
            for point in triangle:
                cell = self.get_model_element_for_point(point)
                if cell is not None:
                    if point not in cell.points:
                        cell.points.append(point)
                        cell.count_of_mesh_points += 1
                    if triangle not in cell.triangles:
                        cell.triangles.append(triangle)
                        cell.count_of_triangles += 1
