from sqlalchemy import select, desc
from sqlalchemy.exc import IntegrityError

from classes.MeshDB import MeshDB
from classes.Point import Point
from classes.Triangle import Triangle
from classes.abc_classes.MeshABC import MeshABC
from utils.mesh_utils.mesh_triangulators.ScipyTriangulator import ScipyTriangulator
from utils.start_db import engine, Tables


class MeshLite(MeshABC):

    def __init__(self, scan, scan_triangulator=ScipyTriangulator):
        super().__init__(scan, scan_triangulator)
        self.triangles = []
        self.__init_mesh()

    def __iter__(self):
        return iter(self.triangles)

    def __len__(self):
        return len(self.triangles)

    def clear_mesh_mse(self):
        """
        Удаляет данные о СКП и степенях свободы треугольников в поверхности
        """
        self.mse = None
        self.r = None
        for triangle in self:
            triangle.mse = None
            triangle.r = None

    def calk_mesh_mse(self, base_scan, voxel_size=None,
                      clear_previous_mse=False,
                      delete_temp_models=False):
        triangles = super().calk_mesh_mse(base_scan=base_scan, voxel_size=voxel_size,
                                          clear_previous_mse=clear_previous_mse,
                                          delete_temp_models=delete_temp_models)
        if triangles is None:
            self.logger.warning(f"СКП модели {self.mesh_name} уже рассчитано!")
            return
        self.triangles = list(triangles)

    def __init_mesh(self):
        triangulation = self.scan_triangulator(self.scan).triangulate()
        self.len = len(triangulation.faces)
        fake_point_id = -1
        fake_triangle_id = -1
        for face in triangulation.faces:
            points = []
            for point_idx in face:
                id_ = triangulation.points_id[point_idx]
                if id_ is None:
                    id_ = fake_point_id
                    fake_point_id -= 1
                point = Point(id_=id_,
                              X=triangulation.vertices[point_idx][0],
                              Y=triangulation.vertices[point_idx][1],
                              Z=triangulation.vertices[point_idx][2],
                              R=triangulation.vertices_colors[point_idx][0],
                              G=triangulation.vertices_colors[point_idx][1],
                              B=triangulation.vertices_colors[point_idx][2])
                points.append(point)
            triangle = Triangle(*points)
            triangle.id = fake_triangle_id
            fake_triangle_id -= 1
            self.triangles.append(triangle)

    def save_to_db(self):
        """
        Сохраняет объект ScanLite в базе данных вместе с точками
        """
        with engine.connect() as db_connection:
            if self.id is None:
                mesh_id_stmt = select(Tables.meshes_db_table.c.id).order_by(desc("id"))
                last_mesh_id = db_connection.execute(mesh_id_stmt).first()
                if last_mesh_id:
                    self.id = last_mesh_id[0] + 1
                else:
                    self.id = 1
            point_id_stmt = select(Tables.points_db_table.c.id).order_by(desc("id"))
            last_point_id = db_connection.execute(point_id_stmt).first()
            last_point_id = last_point_id[0] if last_point_id else 0
            points_id_dict = {}
            triangles = []
            points = []
            for triangle in self:
                points_id = []
                for point in triangle:
                    if point.id in points_id_dict:
                        point_id = points_id_dict[point.id]
                    elif point.id < 0:
                        last_point_id += 1
                        point_id = last_point_id
                        points.append({"id": point_id,
                                       "X": point.X, "Y": point.Y, "Z": point.Z,
                                       "R": point.R, "G": point.G, "B": point.B
                                       })
                        points_id_dict[point.id] = point_id
                    else:
                        point_id = point.id
                        points_id_dict[point.id] = point_id
                    points_id.append(point_id)
                triangles.append({"point_0_id": points_id[0],
                                  "point_1_id": points_id[1],
                                  "point_2_id": points_id[2],
                                  "r": triangle.r,
                                  "mse": triangle.mse,
                                  "mesh_id": self.id})
            try:
                if len(points) > 0:
                    db_connection.execute(Tables.points_db_table.insert(), points)
                db_connection.execute(Tables.triangles_db_table.insert(), triangles)
                db_connection.execute(Tables.meshes_db_table.insert(),
                                      [{"id": self.id,
                                        "mesh_name": self.mesh_name,
                                        "len": self.len,
                                        "r": self.r, "mse": self.mse,
                                        "base_scan_id": self.scan.id}])
                db_connection.commit()
                return self
            except IntegrityError:
                self.logger.warning(f"Такие объекты уже присутствуют в Базе Данных!!")
                return MeshDB(scan=self.scan)
