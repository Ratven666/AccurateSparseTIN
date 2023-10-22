import os
import sqlite3

from CONFIG import DATABASE_NAME
from classes.Triangle import Triangle


class SqlLiteMeshIterator:
    """
        Иттератор поверхности из БД SQLite
        Реализован через стандартную библиотеку sqlite3
        """

    def __init__(self, mesh):
        self.__path = os.path.join("data_bases", DATABASE_NAME)
        self.mesh_id = mesh.id
        self.cursor = None
        self.generator = None

    def __iter__(self):
        connection = sqlite3.connect(self.__path)
        self.cursor = connection.cursor()
        stmt = """SELECT t.id, t.r, t.mse,
                         p0.id, p0.X, p0.Y, p0.Z, p0.R, p0.G, p0.B,
                         p1.id, p1.X, p1.Y, p1.Z, p1.R, p1.G, p1.B,
                         p2.id, p2.X, p2.Y, p2.Z, p2.R, p2.G, p2.B
                  FROM (SELECT t.id, t.r, t.mse,
                               t.point_0_id,
                               t.point_1_id,
                               t.point_2_id
                        FROM triangles t
                        WHERE t.mesh_id = (?)) t
                  JOIN points p0 ON p0.id = t.point_0_id
                  JOIN points p1 ON p1.id = t.point_1_id
                  JOIN points p2 ON p2.id = t.point_2_id
                  """
        self.generator = (Triangle.parse_triangle_from_db_row(data)
                          for data in self.cursor.execute(stmt, (self.mesh_id,)))
        return self.generator

    def __next__(self):
        try:
            return next(self.generator)
        except StopIteration:
            self.cursor.close()
            raise StopIteration
        finally:
            self.cursor.close()
