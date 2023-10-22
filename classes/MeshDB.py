from sqlalchemy import select, insert, update

from classes.ScanDB import ScanDB
from classes.abc_classes.MeshABC import MeshABC
from utils.mesh_utils.mesh_iterators.SqlLiteMeshIterator import SqlLiteMeshIterator
from utils.mesh_utils.mesh_triangulators.ScipyTriangulator import ScipyTriangulator
from utils.start_db import Tables, engine


class MeshDB(MeshABC):
    """
    Поверхность связанная с базой данных
    Треугольники при переборе поверхности берутся напрямую из БД
    """
    db_table = Tables.meshes_db_table

    def __init__(self, scan, scan_triangulator=ScipyTriangulator, db_connection=None):
        super().__init__(scan, scan_triangulator)
        self.base_scan_id = None
        self.__init_mesh(db_connection)

    def __iter__(self):
        return iter(SqlLiteMeshIterator(self))

    def calk_mesh_mse(self, base_scan, voxel_size=None,
                      clear_previous_mse=False,
                      delete_temp_models=False):
        triangles = super().calk_mesh_mse(base_scan=base_scan, voxel_size=voxel_size,
                                          clear_previous_mse=clear_previous_mse,
                                          delete_temp_models=delete_temp_models)
        if triangles is None:
            self.logger.warning(f"СКП модели {self.mesh_name} уже рассчитано!")
            return
        with engine.connect() as db_connection:
            for triangle in triangles:
                stmt = update(Tables.triangles_db_table)\
                    .where(Tables.triangles_db_table.c.id == triangle.id)\
                    .values(r=triangle.r,
                            mse=triangle.mse)
                db_connection.execute(stmt)
            stmt = update(self.db_table) \
                .where(self.db_table.c.id == self.id) \
                .values(r=self.r,
                        mse=self.mse)
            db_connection.execute(stmt)
            db_connection.commit()

    def clear_mesh_mse(self):
        """
        Удаляет записи о СКП и степенях свободы поверхности и ее треугольников из БД
        """
        with engine.connect() as db_connection:
            for triangle in self:
                stmt = update(Tables.triangles_db_table)\
                    .where(Tables.triangles_db_table.c.id == triangle.id)\
                    .values(r=None,
                            mse=None)
                db_connection.execute(stmt)
            stmt = update(self.db_table) \
                .where(self.db_table.c.id == self.id) \
                .values(r=None,
                        mse=None)
            db_connection.execute(stmt)
            db_connection.commit()

    @classmethod
    def get_mesh_by_id(cls, mesh_id):
        select_ = select(cls.db_table).where(cls.db_table.c.id == mesh_id)
        with engine.connect() as db_connection:
            db_mesh_data = db_connection.execute(select_).mappings().first()
            if db_mesh_data is not None:
                scan_id = db_mesh_data["base_scan_id"]
                scan = ScanDB.get_scan_from_id(scan_id)
                return cls(scan)
            else:
                raise ValueError("Нет поверхности с таким id!!!")

    def __load_triangle_data_to_db(self, db_conn, triangulation):
        """
        Загружает рассчитаные треугольники в БД
        """
        triangle_data = []
        for triangle in triangulation.faces:
            triangle_data.append({"point_0_id": triangulation.points_id[triangle[0]],
                                  "point_1_id": triangulation.points_id[triangle[1]],
                                  "point_2_id": triangulation.points_id[triangle[2]],
                                  "mesh_id": self.id})
        db_conn.execute(Tables.triangles_db_table.insert(), triangle_data)
        db_conn.commit()

    def __init_mesh(self, db_connection=None, triangulation=None):
        """
        Инициализирует поверхность при запуске
        Если поверхность с таким именем уже есть в БД - запускает копирование данных из БД в атрибуты поверхности
        Если такой поверхности нет - создает новую запись в БД
        :param db_connection: Открытое соединение с БД
        :return: None
        """
        def init_logic(db_conn, triangulation):
            select_ = select(self.db_table).where(self.db_table.c.mesh_name == self.mesh_name)
            db_mesh_data = db_conn.execute(select_).mappings().first()
            if db_mesh_data is not None:
                self.__copy_mesh_data(db_mesh_data)
                if triangulation is not None:
                    self.__load_triangle_data_to_db(db_conn, triangulation)
            else:
                triangulation = self.scan_triangulator(self.scan).triangulate()
                self.len = len(triangulation.faces)
                stmt = insert(self.db_table).values(mesh_name=self.mesh_name,
                                                    len=self.len,
                                                    r=self.r,
                                                    mse=self.mse,
                                                    base_scan_id=self.scan.id)
                db_conn.execute(stmt)
                db_conn.commit()
                self.__init_mesh(db_conn, triangulation)

        if db_connection is None:
            with engine.connect() as db_connection:
                init_logic(db_connection, triangulation)
        else:
            init_logic(db_connection, triangulation)

    def __copy_mesh_data(self, db_mesh_data: dict):
        """
        Копирует данные записи из БД в атрибуты поверхности
        :param db_mesh_data: Результат запроса к БД
        :return: None
        """
        self.id = db_mesh_data["id"]
        self.scan_name = db_mesh_data["mesh_name"]
        self.len = db_mesh_data["len"]
        self.r = db_mesh_data["r"]
        self.mse = db_mesh_data["mse"]
        self.base_scan_id = db_mesh_data["base_scan_id"]
