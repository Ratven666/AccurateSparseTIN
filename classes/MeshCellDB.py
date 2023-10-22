from sqlalchemy import insert

from classes.Point import Point
from classes.abc_classes.CellABC import CellABC
from utils.start_db import Tables


class MeshCellDB(CellABC):

    db_table = Tables.mesh_cell_db_table

    def __init__(self, voxel, dem_model):
        self.voxel = voxel
        self.dem_model = dem_model
        self.voxel_id = None
        self.count_of_mesh_points = 0
        self.count_of_triangles = 0
        self.r = None
        self.mse = None
        self.points = []
        self.triangles = []

    def get_z_from_xy(self, x, y):
        """
        Рассчитывает отметку точки (x, y) в ячейке
        :param x: координата x
        :param y: координата y
        :return: координата z для точки (x, y)
        """
        point = Point(x, y, 0, 0, 0, 0)
        for triangle in self.triangles:
            if triangle.is_point_in_triangle(point):
                return triangle.get_z_from_xy(x, y)
        return None

    def _save_cell_data_in_db(self, db_connection):
        """
        Сохраняет данные ячейки из модели в БД
        :param db_connection: открытое соединение с БД
        :return: None
        """
        stmt = insert(self.db_table).values(voxel_id=self.voxel.id,
                                            base_model_id=self.dem_model.id,
                                            count_of_mesh_points=self.count_of_mesh_points,
                                            count_of_triangles=self.count_of_triangles,
                                            r=self.r,
                                            mse=self.mse,
                                            )
        db_connection.execute(stmt)
