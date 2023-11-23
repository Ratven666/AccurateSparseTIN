"""
requirements.txt

ezdxf==1.1.1
numpy==1.26.1
pandas==2.1.1
PyQt6==6.5.3
scipy==1.11.3
seaborn==0.13.0
SQLAlchemy==2.0.22
"""

import csv
import logging
import os
import sqlite3
import sys
from abc import ABC, abstractmethod
from copy import copy
from pathlib import Path
from threading import Lock

import ezdxf
import numpy as np
import pandas as pd
import seaborn as sns

from PyQt6 import QtCore, QtGui
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QFrame, QSlider, QHBoxLayout, QLabel, QSpacerItem, \
    QSizePolicy, QSpinBox, QDoubleSpinBox, QTextEdit, QToolButton, QCheckBox, QProgressBar, QPushButton, QTableWidget, \
    QTableWidgetItem, QFileDialog, QMessageBox, QApplication
from scipy.spatial import Delaunay
from sqlalchemy import delete, and_, update, select, desc, MetaData, create_engine, Table, Column, ForeignKey, Integer, \
    Float, String, insert, func
from sqlalchemy.exc import IntegrityError

DATABASE_NAME = "TEMP.sqlite"
LOGGER = "console"
POINTS_CHUNK_COUNT = 100_000
VOXEL_IN_VM = 100_000

db_path = os.path.join("", DATABASE_NAME)
engine = create_engine(f'sqlite:///{db_path}')
db_metadata = MetaData()


class SingletonMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class TableInitializer(metaclass=SingletonMeta):
    def __init__(self, metadata):
        self.metadata = metadata
        self.points_db_table = self.create_points_db_table()
        self.scans_db_table = self.create_scans_db_table()
        self.points_scans_db_table = self.create_points_scans_db_table()
        self.imported_files_db_table = self.create_imported_files_table()
        self.voxel_models_db_table = self.create_voxel_models_db_table()
        self.voxels_db_table = self.create_voxels_db_table()
        self.dem_models_db_table = self.create_dem_models_db_table()
        self.meshes_db_table = self.create_meshes_db_table()
        self.mesh_cell_db_table = self.create_mesh_cell_db_table()
        self.triangles_db_table = self.create_triangles_db_table()

    def create_points_db_table(self):
        points_db_table = Table("points", self.metadata,
                                Column("id", Integer, primary_key=True),
                                Column("X", Float, nullable=False),
                                Column("Y", Float, nullable=False),
                                Column("Z", Float, nullable=False),
                                Column("R", Integer, nullable=False),
                                Column("G", Integer, nullable=False),
                                Column("B", Integer, nullable=False),
                                )
        return points_db_table

    def create_scans_db_table(self):
        scans_db_table = Table("scans", self.metadata,
                               Column("id", Integer, primary_key=True),
                               Column("scan_name", String, nullable=False, unique=True, index=True),
                               Column("len", Integer, default=0),
                               Column("min_X", Float),
                               Column("max_X", Float),
                               Column("min_Y", Float),
                               Column("max_Y", Float),
                               Column("min_Z", Float),
                               Column("max_Z", Float),
                               )
        return scans_db_table

    def create_points_scans_db_table(self):
        points_scans_db_table = Table("points_scans", self.metadata,
                                      Column("point_id", Integer, ForeignKey("points.id"), primary_key=True),
                                      Column("scan_id", Integer, ForeignKey("scans.id"), primary_key=True)
                                      )
        return points_scans_db_table

    def create_triangles_db_table(self):
        triangles_db_table = Table("triangles", self.metadata,
                                   Column("id", Integer, primary_key=True),
                                   Column("point_0_id", Integer, ForeignKey("points.id")),
                                   Column("point_1_id", Integer, ForeignKey("points.id")),
                                   Column("point_2_id", Integer, ForeignKey("points.id")),
                                   Column("r", Integer, default=None),
                                   Column("mse", Float, default=None),
                                   Column("mesh_id", Integer, ForeignKey("meshes.id"))
                                   )
        return triangles_db_table

    def create_meshes_db_table(self):
        meshes_db_table = Table("meshes", self.metadata,
                                Column("id", Integer, primary_key=True),
                                Column("mesh_name", String, nullable=False, unique=True, index=True),
                                Column("len", Integer, default=0),
                                Column("r", Integer, default=None),
                                Column("mse", Float, default=None),
                                Column("base_scan_id", Integer, ForeignKey("scans.id"))
                                )
        return meshes_db_table

    def create_imported_files_table(self):
        imported_files_table = Table("imported_files", self.metadata,
                                     Column("id", Integer, primary_key=True),
                                     Column("file_name", String, nullable=False),
                                     Column("scan_id", Integer, ForeignKey("scans.id"))
                                     )
        return imported_files_table

    def create_voxels_db_table(self):
        voxels_db_table = Table("voxels", self.metadata,
                                Column("id", Integer, primary_key=True),
                                Column("vxl_name", String, nullable=False, unique=True, index=True),
                                Column("X", Float),
                                Column("Y", Float),
                                Column("Z", Float),
                                Column("step", Float, nullable=False),
                                Column("len", Integer, default=0),
                                Column("R", Integer, default=0),
                                Column("G", Integer, default=0),
                                Column("B", Integer, default=0),
                                Column("scan_id", Integer, ForeignKey("scans.id", ondelete="CASCADE")),
                                Column("vxl_mdl_id", Integer, ForeignKey("voxel_models.id"))
                                )
        return voxels_db_table

    def create_voxel_models_db_table(self):
        voxel_models_db_table = Table("voxel_models", self.metadata,
                                      Column("id", Integer, primary_key=True),
                                      Column("vm_name", String, nullable=False, unique=True, index=True),
                                      Column("step", Float, nullable=False),
                                      Column("dx", Float, nullable=False),
                                      Column("dy", Float, nullable=False),
                                      Column("dz", Float, nullable=False),
                                      Column("len", Integer, default=0),
                                      Column("X_count", Integer, default=0),
                                      Column("Y_count", Integer, default=0),
                                      Column("Z_count", Integer, default=0),
                                      Column("min_X", Float),
                                      Column("max_X", Float),
                                      Column("min_Y", Float),
                                      Column("max_Y", Float),
                                      Column("min_Z", Float),
                                      Column("max_Z", Float),
                                      Column("base_scan_id", Integer, ForeignKey("scans.id"))
                                      )
        return voxel_models_db_table

    def create_dem_models_db_table(self):
        dem_models_db_table = Table("dem_models", self.metadata,
                                    Column("id", Integer, primary_key=True),
                                    Column("base_voxel_model_id", Integer,
                                           ForeignKey("voxel_models.id")),
                                    Column("model_type", String, nullable=False),
                                    Column("model_name", String, nullable=False, unique=True),
                                    Column("MSE_data", Float, default=None)
                                    )
        return dem_models_db_table

    def create_mesh_cell_db_table(self):
        mesh_cell_db_table = Table("mesh_cells", self.metadata,
                                   Column("voxel_id", Integer,
                                          ForeignKey("voxels.id", ondelete="CASCADE"),
                                          primary_key=True),
                                   Column("base_model_id", Integer,
                                          ForeignKey("dem_models.id", ondelete="CASCADE"),
                                          primary_key=True),
                                   Column("count_of_mesh_points", Integer),
                                   Column("count_of_triangles", Integer),
                                   Column("r", Integer),
                                   Column("mse", Float, default=None)
                                   )
        return mesh_cell_db_table


Tables = TableInitializer(db_metadata)
logger = logging.getLogger("console")


def create_db():
    db_is_created = os.path.exists(db_path)
    if not db_is_created:
        db_metadata.create_all(engine)
    else:
        logger.info("Такая БД уже есть!")


class PointABC(ABC):
    __slots__ = ["id", "X", "Y", "Z", "R", "G", "B"]

    def __init__(self, X, Y, Z, R, G, B, id_=None):
        self.id = id_
        self.X, self.Y, self.Z = X, Y, Z
        self.R, self.G, self.B = R, G, B

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"[id: {self.id},\tx: {self.X} y: {self.Y} z: {self.Z},\t" \
               f"RGB: ({self.R},{self.G},{self.B})]"

    def __repr__(self):
        return f"{self.__class__.__name__} [id: {self.id}]"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, PointABC):
            raise TypeError("Операнд справа должен иметь тип производный от PointABC")
        if hash(self) == hash(other) or self.id is None or other.id is None:
            return (self.X == other.X) and \
                (self.Y == other.Y) and \
                (self.Z == other.Z) and \
                (self.R == other.R) and \
                (self.G == other.G) and \
                (self.B == other.B)
        return False


class Point(PointABC):
    __slots__ = []

    @classmethod
    def parse_point_from_db_row(cls, row: tuple):
        return cls(id_=row[0], X=row[1], Y=row[2], Z=row[3], R=row[4], G=row[5], B=row[6])


class ScanTXTSaver:
    @staticmethod
    def save_scan(scan, file_path, scan_name):
        if scan_name is None:
            file_path = os.path.join(file_path, f"{scan.scan_name}.txt")
        else:
            file_path = os.path.join(file_path, f"{scan_name}.txt")
        with open(file_path, "w", encoding="UTF-8") as file:
            for point in scan:
                point_line = f"{point.X} {point.Y} {point.Z} {point.R} {point.G} {point.B}\n"
                file.write(point_line)
        logger.info(f"Сохранение скана {scan} в файл {file_path} завершено")


class ScanABC(ABC):
    logger = logging.getLogger(LOGGER)

    def __init__(self, scan_name):
        self.id = None
        self.scan_name: str = scan_name
        self.len: int = 0
        self.min_X, self.max_X = None, None
        self.min_Y, self.max_Y = None, None
        self.min_Z, self.max_Z = None, None

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"[id: {self.id},\tName: {self.scan_name}\tLEN: {self.len}]"

    def __repr__(self):
        return f"{self.__class__.__name__} [ID: {self.id}]"

    def __len__(self):
        return self.len

    @abstractmethod
    def __iter__(self):
        pass

    def save_scan_in_file(self, file_path=".", scan_name=None, scan_saver=ScanTXTSaver()):
        scan_saver.save_scan(self, file_path, scan_name)

    @staticmethod
    def update_scan_borders(scan, point):
        if scan.min_X is None:
            scan.min_X, scan.max_X = point.X, point.X
            scan.min_Y, scan.max_Y = point.Y, point.Y
            scan.min_Z, scan.max_Z = point.Z, point.Z
        if point.X < scan.min_X:
            scan.min_X = point.X
        if point.X > scan.max_X:
            scan.max_X = point.X
        if point.Y < scan.min_Y:
            scan.min_Y = point.Y
        if point.Y > scan.max_Y:
            scan.max_Y = point.Y
        if point.Z < scan.min_Z:
            scan.min_Z = point.Z
        if point.Z > scan.max_Z:
            scan.max_Z = point.Z


class ScanLite(ScanABC):
    def __init__(self, scan_name):
        super().__init__(scan_name)
        self._points = []

    def __iter__(self):
        return iter(self._points)

    def __len__(self):
        return len(self._points)

    def add_point(self, point):
        if isinstance(point, PointABC):
            self._points.append(point)
            self.len += 1
            self.update_scan_borders(self, point)
        else:
            raise TypeError(f"Можно добавить только объект точки. "
                            f"Переданно - {type(point)}, {point}")

    @staticmethod
    def insert_scan_in_db_from_scan(updated_scan, db_connection=None):
        stmt = insert(Tables.scans_db_table).values(id=updated_scan.id,
                                                    scan_name=updated_scan.scan_name,
                                                    len=updated_scan.len,
                                                    min_X=updated_scan.min_X,
                                                    max_X=updated_scan.max_X,
                                                    min_Y=updated_scan.min_Y,
                                                    max_Y=updated_scan.max_Y,
                                                    min_Z=updated_scan.min_Z,
                                                    max_Z=updated_scan.max_Z)
        if db_connection is None:
            with engine.connect() as db_connection:
                db_connection.execute(stmt)
                db_connection.commit()
        else:
            db_connection.execute(stmt)
            db_connection.commit()

    def save_to_db(self):
        with engine.connect() as db_connection:
            stmt = (select(Tables.points_db_table.c.id).order_by(desc("id")))
            last_point_id = db_connection.execute(stmt).first()
            if last_point_id:
                last_point_id = last_point_id[0]
            else:
                last_point_id = 0
            if self.id is None:
                stmt = (select(Tables.scans_db_table.c.id).order_by(desc("id")))
                last_scan_id = db_connection.execute(stmt).first()
                if last_scan_id:
                    self.id = last_scan_id[0] + 1
                else:
                    self.id = 1
            points = []
            points_scans = []
            for point in self:
                last_point_id += 1
                point.id = last_point_id
                points.append({"id": point.id,
                               "X": point.X, "Y": point.Y, "Z": point.Z,
                               "R": point.R, "G": point.G, "B": point.B
                               })
                points_scans.append({"point_id": point.id, "scan_id": self.id})
            try:
                self.insert_scan_in_db_from_scan(self, db_connection)
                if len(points) > 0:
                    db_connection.execute(Tables.points_db_table.insert(), points)
                    db_connection.execute(Tables.points_scans_db_table.insert(), points_scans)
                db_connection.commit()
                return ScanDB(self.scan_name)
            except IntegrityError:
                self.logger.warning("Скан с таким именем уже есть в БД!")
                return ScanDB(self.scan_name)


class SqlLiteScanIterator:
    def __init__(self, scan):
        self.__path = os.path.join("", DATABASE_NAME)
        self.scan_id = scan.id
        self.cursor = None
        self.generator = None

    def __iter__(self):
        connection = sqlite3.connect(self.__path)
        self.cursor = connection.cursor()
        stmt = """SELECT p.id, p.X, p.Y, p.Z,
                         p.R, p.G, p.B
                  FROM points p
                  JOIN (SELECT ps.point_id
                        FROM points_scans ps
                        WHERE ps.scan_id = (?)) ps
                  ON ps.point_id = p.id
                          """
        self.generator = (Point.parse_point_from_db_row(data) for data in
                          self.cursor.execute(stmt, (self.scan_id,)))
        return self.generator

    def __next__(self):
        try:
            return next(self.generator)
        except StopIteration:
            self.cursor.close()
            raise StopIteration
        finally:
            self.cursor.close()


class ScanParserABC(ABC):
    logger = logging.getLogger(LOGGER)

    def __str__(self):
        return f"Парсер типа: {self.__class__.__name__}"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _check_file_extension(file_name, __supported_file_extensions__):
        file_extension = f".{file_name.split('.')[-1]}"
        if file_extension not in __supported_file_extensions__:
            raise TypeError(f"Неправильный для парсера тип файла. "
                            f"Ожидаются файлы типа: {__supported_file_extensions__}")

    @staticmethod
    def _get_last_point_id():
        with engine.connect() as db_connection:
            stmt = (select(Tables.points_db_table.c.id).order_by(desc("id")))
            last_point_id = db_connection.execute(stmt).first()
            if last_point_id:
                return last_point_id[0]
            else:
                return 0

    @abstractmethod
    def parse(self, file_name: str):
        pass


class ScanTxtParser(ScanParserABC):
    __supported_file_extension__ = [".txt", ".ascii", ".xyz"]

    def __init__(self, chunk_count=POINTS_CHUNK_COUNT):
        self.__chunk_count = chunk_count
        self.__last_point_id = None

    def parse(self, file_name):
        self._check_file_extension(file_name, self.__supported_file_extension__)
        self.__last_point_id = self._get_last_point_id()
        with open(file_name, "rt", encoding="utf-8") as file:
            points = []
            for line in file:
                line = line.strip().split()
                self.__last_point_id += 1
                try:
                    point = {"id": self.__last_point_id,
                             "X": line[0], "Y": line[1], "Z": line[2],
                             "R": line[3], "G": line[4], "B": line[5]
                             }
                except IndexError:
                    self.logger.critical(f"Структура \"{file_name}\" некорректна - \"{line}\"")
                    return
                points.append(point)
                if len(points) == self.__chunk_count:
                    yield points
                    points = []
            yield points


class ImportedFileDB:
    def __init__(self, file_name):
        self.__file_name = file_name
        self.__hash = None

    def is_file_already_imported_into_scan(self, scan):
        select_ = select(Tables.imported_files_db_table).where(
            and_(Tables.imported_files_db_table.c.file_name == self.__file_name,
                 Tables.imported_files_db_table.c.scan_id == scan.id))
        with engine.connect() as db_connection:
            imp_file = db_connection.execute(select_).first()
        if imp_file is None:
            return False
        return True

    def insert_in_db(self, scan):
        with engine.connect() as db_connection:
            stmt = insert(Tables.imported_files_db_table).values(file_name=self.__file_name,
                                                                 scan_id=scan.id)
            db_connection.execute(stmt)
            db_connection.commit()


class ScanLoader:
    __logger = logging.getLogger(LOGGER)

    def __init__(self, scan_parser=ScanTxtParser()):
        self.__scan_parser = scan_parser

    @staticmethod
    def calk_scan_metrics(scan_id):
        with engine.connect() as db_connection:
            stmt = select(func.count(Tables.points_db_table.c.id).label("len"),
                          func.min(Tables.points_db_table.c.X).label("min_X"),
                          func.max(Tables.points_db_table.c.X).label("max_X"),
                          func.min(Tables.points_db_table.c.Y).label("min_Y"),
                          func.max(Tables.points_db_table.c.Y).label("max_Y"),
                          func.min(Tables.points_db_table.c.Z).label("min_Z"),
                          func.max(Tables.points_db_table.c.Z).label("max_Z")).where(and_(
                Tables.points_scans_db_table.c.point_id == Tables.points_db_table.c.id,
                Tables.points_scans_db_table.c.scan_id == Tables.scans_db_table.c.id,
                Tables.scans_db_table.c.id == scan_id
            ))
            scan_metrics = dict(db_connection.execute(stmt).mappings().first())
            scan_metrics["id"] = scan_id
            return scan_metrics

    def update_scan_metrics(self, scan):
        scan_metrics = self.calk_scan_metrics(scan_id=scan.id)
        scan.len = scan_metrics["len"]
        scan.min_X, scan.max_X = scan_metrics["min_X"], scan_metrics["max_X"]
        scan.min_Y, scan.max_Y = scan_metrics["min_Y"], scan_metrics["max_Y"]
        scan.min_Z, scan.max_Z = scan_metrics["min_Z"], scan_metrics["max_Z"]
        return scan

    @staticmethod
    def update_scan_in_db_from_scan(updated_scan, db_connection=None):
        stmt = update(Tables.scans_db_table) \
            .where(Tables.scans_db_table.c.id == updated_scan.id) \
            .values(scan_name=updated_scan.scan_name,
                    len=updated_scan.len,
                    min_X=updated_scan.min_X,
                    max_X=updated_scan.max_X,
                    min_Y=updated_scan.min_Y,
                    max_Y=updated_scan.max_Y,
                    min_Z=updated_scan.min_Z,
                    max_Z=updated_scan.max_Z)
        if db_connection is None:
            with engine.connect() as db_connection:
                db_connection.execute(stmt)
                db_connection.commit()
        else:
            db_connection.execute(stmt)
            db_connection.commit()

    def load_data(self, scan, file_name: str):
        imp_file = ImportedFileDB(file_name)

        if imp_file.is_file_already_imported_into_scan(scan):
            self.__logger.info(f"Файл \"{file_name}\" уже загружен в скан \"{scan.scan_name}\"")
            return

        with engine.connect() as db_connection:
            for points in self.__scan_parser.parse(file_name):
                points_scans = self.__get_points_scans_list(scan, points)
                self.__insert_to_db(points, points_scans, db_connection)
                self.__logger.info(f"Пакет точек загружен в БД")
            db_connection.commit()
        scan = self.update_scan_metrics(scan)
        self.update_scan_in_db_from_scan(scan)
        imp_file.insert_in_db(scan)
        self.__logger.info(f"Точки из файла \"{file_name}\" успешно"
                           f" загружены в скан \"{scan.scan_name}\"")

    @staticmethod
    def __get_points_scans_list(scan, points):
        points_scans = []
        for point in points:
            points_scans.append({"point_id": point["id"], "scan_id": scan.id})
        return points_scans

    @staticmethod
    def __insert_to_db(points, points_scans, db_engine_connection):
        db_engine_connection.execute(Tables.points_db_table.insert(), points)
        db_engine_connection.execute(Tables.points_scans_db_table.insert(), points_scans)

    @property
    def scan_parser(self):
        return self.__scan_parser

    @scan_parser.setter
    def scan_parser(self, parser: ScanParserABC):
        if isinstance(parser, ScanParserABC):
            self.__scan_parser = parser
        else:
            raise TypeError(f"Нужно передать объект парсера! "
                            f"Переданно - {type(parser)}, {parser}")


class ScanDB(ScanABC):
    def __init__(self, scan_name, db_connection=None):
        super().__init__(scan_name)
        self.__init_scan(db_connection)

    def __iter__(self):
        return iter(SqlLiteScanIterator(self))

    def load_scan_from_file(self, file_name,
                            scan_loader=ScanLoader(scan_parser=ScanTxtParser(chunk_count=POINTS_CHUNK_COUNT))):
        scan_loader.load_data(self, file_name)

    @classmethod
    def get_scan_from_id(cls, scan_id: int):
        select_ = select(Tables.scans_db_table).where(Tables.scans_db_table.c.id == scan_id)
        with engine.connect() as db_connection:
            db_scan_data = db_connection.execute(select_).mappings().first()
            if db_scan_data is not None:
                return cls(db_scan_data["scan_name"])
            else:
                raise ValueError("Нет скана с таким id!!!")

    def __init_scan(self, db_connection=None):
        def init_logic(db_conn):
            select_ = select(Tables.scans_db_table).where(Tables.scans_db_table.c.scan_name == self.scan_name)
            db_scan_data = db_conn.execute(select_).mappings().first()
            if db_scan_data is not None:
                self.__copy_scan_data(db_scan_data)
            else:
                stmt = insert(Tables.scans_db_table).values(scan_name=self.scan_name)
                db_conn.execute(stmt)
                db_conn.commit()
                self.__init_scan(db_conn)
        if db_connection is None:
            with engine.connect() as db_connection:
                init_logic(db_connection)
        else:
            init_logic(db_connection)

    def __copy_scan_data(self, db_scan_data: dict):
        self.id = db_scan_data["id"]
        self.scan_name = db_scan_data["scan_name"]
        self.len = db_scan_data["len"]
        self.min_X, self.max_X = db_scan_data["min_X"], db_scan_data["max_X"]
        self.min_Y, self.max_Y = db_scan_data["min_Y"], db_scan_data["max_Y"]
        self.min_Z, self.max_Z = db_scan_data["min_Z"], db_scan_data["max_Z"]


class CsvMeshDataExporter:
    def __init__(self, mesh):
        self.mesh = mesh
        self.file_name = f"{self.mesh.mesh_name}.csv"

    def export_mesh_data(self, file_path="."):
        file_path = os.path.join(file_path, f"{self.file_name}")
        with open(file_path, "w", newline="") as csvfile:
            fieldnames = ["point_0", "point_1", "point_2", "area", "r", "rmse"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for triangle in self.mesh:
                r = 0 if triangle.r is None else triangle.r
                data = {"point_0": self.__point_data_exporter(triangle.point_0),
                        "point_1": self.__point_data_exporter(triangle.point_1),
                        "point_2": self.__point_data_exporter(triangle.point_2),
                        "area": triangle.get_area(),
                        "r": r,
                        "rmse": triangle.mse}
                writer.writerow(data)
        return self.file_name

    @staticmethod
    def __point_data_exporter(point):
        return {"XYZ": [point.X, point.Y, point.Z],
                "RGB": [point.R, point.G, point.B]}


class VoxelABC(ABC):
    logger = logging.getLogger(LOGGER)

    def __init__(self, x, y, z, step, vxl_mdl_id, id_=None):
        self.id = id_
        self.X = x
        self.Y = y
        self.Z = z
        self.step = step
        self.vxl_mdl_id = vxl_mdl_id
        self.vxl_name = self.__name_generator()
        self.scan_id = None
        self.len = 0
        self.R, self.G, self.B = 0, 0, 0
        self.container_dict = {}

    def __name_generator(self):
        return (f"VXL_VM:{self.vxl_mdl_id}_s{self.step}_"
                f"X:{round(self.X, 5)}_"
                f"Y:{round(self.Y, 5)}_"
                f"Z:{round(self.Z, 5)}"
                )

    def __str__(self):
        return (f"{self.__class__.__name__} "
                f"[id: {self.id},\tName: {self.vxl_name}\t\t"
                f"X: {round(self.X, 5)}\tY: {round(self.Y, 5)}\tZ: {round(self.Z, 5)}]"
                )

    def __repr__(self):
        return f"{self.__class__.__name__} [ID: {self.id}]"

    def __len__(self):
        return self.len


class VoxelLite(VoxelABC):
    __slots__ = ["id", "X", "Y", "Z", "step", "vxl_mdl_id", "vxl_name",
                 "scan_id", "len", "R", "G", "B", "container_dict"]

    def __init__(self, X, Y, Z, step, vxl_mdl_id, id_=None):
        super().__init__(X, Y, Z, step, vxl_mdl_id, id_)
        self.scan = ScanLite(f"SC_{self.vxl_name}")


class ScipyTriangulator:
    def __init__(self, scan):
        self.scan = scan
        self.points_id = None
        self.vertices = None
        self.vertices_colors = None
        self.faces = None
        self.face_colors = None

    def __str__(self):
        return (f"{self.__class__.__name__} "
                f"[Name: {self.scan.scan_name}\t\t"
                f"Count_of_triangles: {len(self.faces)}]"
                )

    def __get_data_dict(self):
        point_id_lst, x_lst, y_lst, z_lst, c_lst = [], [], [], [], []
        for point in self.scan:
            point_id_lst.append(point.id)
            x_lst.append(point.X)
            y_lst.append(point.Y)
            z_lst.append(point.Z)
            c_lst.append([point.R, point.G, point.B])
        return {"id": point_id_lst, "x": x_lst, "y": y_lst, "z": z_lst, "color": c_lst}

    def __calk_delone_triangulation(self):
        points2D = self.vertices[:, :2]
        tri = Delaunay(points2D)
        i_lst, j_lst, k_lst = ([triplet[c] for triplet in tri.simplices] for c in range(3))
        return {"i_lst": i_lst, "j_lst": j_lst, "k_lst": k_lst, "ijk": tri.simplices}

    @staticmethod
    def __calk_faces_colors(ijk_dict, scan_data):
        c_lst = []
        for idx in range(len(ijk_dict["i_lst"])):
            c_i = scan_data["color"][ijk_dict["i_lst"][idx]]
            c_j = scan_data["color"][ijk_dict["j_lst"][idx]]
            c_k = scan_data["color"][ijk_dict["k_lst"][idx]]
            r = round((c_i[0] + c_j[0] + c_k[0]) / 3)
            g = round((c_i[1] + c_j[1] + c_k[1]) / 3)
            b = round((c_i[2] + c_j[2] + c_k[2]) / 3)
            c_lst.append([r, g, b])
        return c_lst

    def triangulate(self):
        scan_data = self.__get_data_dict()
        self.vertices = np.vstack([scan_data["x"], scan_data["y"], scan_data["z"]]).T
        self.vertices_colors = scan_data["color"]
        self.points_id = scan_data["id"]
        tri_data_dict = self.__calk_delone_triangulation()
        self.faces = tri_data_dict["ijk"]
        self.face_colors = self.__calk_faces_colors(tri_data_dict, scan_data)
        return self


class SegmentedModelABC(ABC):
    logger = logging.getLogger(LOGGER)
    db_table = Tables.dem_models_db_table

    def __init__(self, voxel_model, element_class):
        self.base_voxel_model_id = voxel_model.id
        self.voxel_model = voxel_model
        self._model_structure = {}
        self._create_model_structure(element_class)
        self.__init_model()

    def __iter__(self):
        return iter(self._model_structure.values())

    def __str__(self):
        return f"{self.__class__.__name__} [ID: {self.id},\tmodel_name: {self.model_name}]"

    def __repr__(self):
        return f"{self.__class__.__name__} [ID: {self.id}]"

    def get_z_from_point(self, point):
        cell = self.get_model_element_for_point(point)
        try:
            z = cell.get_z_from_xy(point.X, point.Y)
        except AttributeError:
            z = None
        return z

    @abstractmethod
    def _calk_segment_model(self):
        pass

    def _create_model_structure(self, element_class):
        for voxel in self.voxel_model:
            model_key = f"{voxel.X:.5f}_{voxel.Y:.5f}_{voxel.Z:.5f}"
            self._model_structure[model_key] = element_class(voxel, self)

    def get_model_element_for_point(self, point):
        X = point.X // self.voxel_model.step * self.voxel_model.step
        Y = point.Y // self.voxel_model.step * self.voxel_model.step
        if self.voxel_model.is_2d_vxl_mdl is False:
            Z = point.Z // self.voxel_model.step * self.voxel_model.step
        else:
            Z = self.voxel_model.min_Z
        model_key = f"{X:.5f}_{Y:.5f}_{Z:.5f}"
        return self._model_structure.get(model_key, None)

    def _load_cell_data_from_db(self, db_connection):
        for cell in self._model_structure.values():
            cell._load_cell_data_from_db(db_connection)

    def _save_cell_data_in_db(self, db_connection):
        for cell in self._model_structure.values():
            cell._save_cell_data_in_db(db_connection)

    def _get_last_model_id(self):
        with engine.connect() as db_connection:
            stmt = (select(self.db_table.c.id).order_by(desc("id")))
            last_model_id = db_connection.execute(stmt).first()
            if last_model_id:
                return last_model_id[0]
            else:
                return 0

    def _copy_model_data(self, db_model_data: dict):
        self.id = db_model_data["id"]
        self.base_voxel_model_id = db_model_data["base_voxel_model_id"]
        self.model_type = db_model_data["model_type"]
        self.model_name = db_model_data["model_name"]
        self.mse_data = db_model_data["MSE_data"]

    def __init_model(self):
        select_ = select(self.db_table) \
            .where(and_(self.db_table.c.base_voxel_model_id == self.voxel_model.id,
                        self.db_table.c.model_type == self.model_type))
        with engine.connect() as db_connection:
            db_model_data = db_connection.execute(select_).mappings().first()
            if db_model_data is not None:
                self._copy_model_data(db_model_data)
                self._load_cell_data_from_db(db_connection)
                self.logger.info(f"Загрузка {self.model_name} модели завершена")
            else:
                stmt = insert(self.db_table).values(base_voxel_model_id=self.voxel_model.id,
                                                    model_type=self.model_type,
                                                    model_name=self.model_name,
                                                    MSE_data=self.mse_data
                                                    )
                db_connection.execute(stmt)
                db_connection.commit()
                self.id = self._get_last_model_id()
                self._calk_segment_model()
                self._save_cell_data_in_db(db_connection)
                db_connection.commit()
                self.logger.info(f"Расчет модели {self.model_name} завершен и загружен в БД\n")

    def delete_model(self, db_connection=None):
        stmt_1 = delete(self.db_table).where(self.db_table.c.id == self.id)
        stmt_2 = delete(self.cell_type.db_table).where(self.cell_type.db_table.c.base_model_id == self.id)
        if db_connection is None:
            with engine.connect() as db_connection:
                db_connection.execute(stmt_1)
                db_connection.commit()
                db_connection.execute(stmt_2)
                db_connection.commit()
        else:
            db_connection.execute(stmt_1)
            db_connection.commit()
            db_connection.execute(stmt_2)
            db_connection.commit()
        self.logger.info(f"Удаление модели {self.model_name} из БД завершено\n")


class CellABC(ABC):
    def __str__(self):
        return f"{self.__class__.__name__} [id: {self.voxel_id}]"

    def __repr__(self):
        return f"{self.__class__.__name__} [id: {self.voxel_id}]"

    @abstractmethod
    def get_z_from_xy(self, x, y):
        pass

    @abstractmethod
    def _save_cell_data_in_db(self, db_connection):
        pass

    def _load_cell_data_from_db(self, db_connection):
        select_ = select(self.db_table) \
            .where(and_(self.db_table.c.voxel_id == self.voxel.id,
                        self.db_table.c.base_model_id == self.dem_model.id))
        db_cell_data = db_connection.execute(select_).mappings().first()
        if db_cell_data is not None:
            self._copy_cell_data(db_cell_data)


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
        point = Point(x, y, 0, 0, 0, 0)
        for triangle in self.triangles:
            if triangle.is_point_in_triangle(point):
                return triangle.get_z_from_xy(x, y)
        return None

    def _save_cell_data_in_db(self, db_connection):
        stmt = insert(self.db_table).values(voxel_id=self.voxel.id,
                                            base_model_id=self.dem_model.id,
                                            count_of_mesh_points=self.count_of_mesh_points,
                                            count_of_triangles=self.count_of_triangles,
                                            r=self.r,
                                            mse=self.mse,
                                            )
        db_connection.execute(stmt)

    def _copy_cell_data(self, db_cell_data):
        self.voxel_id = db_cell_data["voxel_id"]
        self.base_model_id = db_cell_data["base_model_id"]
        self.count_of_mesh_points = db_cell_data["count_of_mesh_points"]
        self.count_of_triangles = db_cell_data["count_of_triangles"]
        self.r = db_cell_data["r"]
        self.mse = db_cell_data["mse"]


class MeshSegmentModelDB(SegmentedModelABC):
    def __init__(self, voxel_model, mesh):
        self.model_type = "MESH"
        self.model_name = f"{self.model_type}_from_{voxel_model.vm_name}"
        self.mse_data = None
        self.cell_type = MeshCellDB
        self.mesh = mesh
        super().__init__(voxel_model, self.cell_type)

    def _calk_segment_model(self):
        pass

    def _calk_cell_mse(self, base_scan):
        pass

    def _create_model_structure(self, element_class):
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


class SqlLiteMeshIterator:
    def __init__(self, mesh):
        self.__path = os.path.join("", DATABASE_NAME)
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


class Triangle:
    __slots__ = ["id", "point_0", "point_1", "point_2", "r", "mse", "vv", "container_dict"]

    def __init__(self, point_0: Point, point_1: Point, point_2: Point, r=None, mse=None, id_=None):
        self.id = id_
        self.point_0 = point_0
        self.point_1 = point_1
        self.point_2 = point_2
        self.r = r
        self.mse = mse
        self.vv = 0
        self.container_dict = {}

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"[id: {self.id}\t[[Point_0: [id: {self.point_0.id},\t" \
               f"x: {self.point_0.X} y: {self.point_0.Y} z: {self.point_0.Z}]\t" \
               f"\t\t [Point_1: [id: {self.point_1.id},\t" \
               f"x: {self.point_1.X} y: {self.point_1.Y} z: {self.point_1.Z}]\t" \
               f"\t\t [Point_2: [id: {self.point_2.id},\t" \
               f"x: {self.point_2.X} y: {self.point_2.Y} z: {self.point_2.Z}]\t" \
               f"r: {self.r},\tmse: {self.mse}"

    def __repr__(self):
        return f"{self.__class__.__name__} [id={self.id}, points=[{self.point_0.id}-" \
               f"{self.point_1.id}-{self.point_2.id}]]"

    def __iter__(self):
        point_lst = [self.point_0, self.point_1, self.point_2]
        return iter(point_lst)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Triangle):
            raise TypeError("Операнд справа должен иметь тип Triangle")
        if hash(self) == hash(other) or self.id is None or other.id is None:
            return (self.point_0 == other.point_0) and \
                (self.point_1 == other.point_1) and \
                (self.point_2 == other.point_2)
        return False

    def get_z_from_xy(self, x, y):
        a = -((self.point_1.Y - self.point_0.Y) * (self.point_2.Z - self.point_0.Z) -
              (self.point_2.Y - self.point_0.Y) * (self.point_1.Z - self.point_0.Z))
        b = ((self.point_1.X - self.point_0.X) * (self.point_2.Z - self.point_0.Z) -
             (self.point_2.X - self.point_0.X) * (self.point_1.Z - self.point_0.Z))
        c = -((self.point_1.X - self.point_0.X) * (self.point_2.Y - self.point_0.Y) -
              (self.point_2.X - self.point_0.X) * (self.point_1.Y - self.point_0.Y))
        d = -(self.point_0.X * a + self.point_0.Y * b + self.point_0.Z * c)
        try:
            z = (a * x + b * y + d) / -c
        except ZeroDivisionError:
            return None
        return z

    def get_area(self):
        a = ((self.point_1.X - self.point_0.X) ** 2 + (self.point_1.Y - self.point_0.Y) ** 2) ** 0.5
        b = ((self.point_2.X - self.point_1.X) ** 2 + (self.point_2.Y - self.point_1.Y) ** 2) ** 0.5
        c = ((self.point_0.X - self.point_2.X) ** 2 + (self.point_0.Y - self.point_2.Y) ** 2) ** 0.5
        p = (a + b + c) / 2
        geron = (p * (p - a) * (p - b) * (p - c))
        s = geron ** 0.5 if geron > 0 else 0
        return s

    def is_point_in_triangle(self, point: Point):
        s_abc = self.get_area()
        if s_abc == 0:
            return False
        s_ab_p = Triangle(self.point_0, self.point_1, point).get_area()
        s_bc_p = Triangle(self.point_1, self.point_2, point).get_area()
        s_ca_p = Triangle(self.point_2, self.point_0, point).get_area()
        delta_s = abs(s_abc - (s_ab_p + s_bc_p + s_ca_p))
        if delta_s < 1e-6:
            return True
        return False

    @classmethod
    def parse_triangle_from_db_row(cls, row: tuple):
        id_ = row[0]
        r = row[1]
        mse = row[2]
        point_0 = Point.parse_point_from_db_row(row[3:10])
        point_1 = Point.parse_point_from_db_row(row[10:17])
        point_2 = Point.parse_point_from_db_row(row[17:])
        return cls(id_=id_, r=r, mse=mse, point_0=point_0, point_1=point_1, point_2=point_2)


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
        pass

    def calk_mesh_mse(self, base_scan, voxel_size=None,
                      clear_previous_mse=False,
                      delete_temp_models=False):
        if clear_previous_mse:
            self.clear_mesh_mse()
            self.r = None
            self.mse = None
        if self.mse is not None:
            return None
        if voxel_size is None:
            voxel_size = VoxelModelLite.get_step_by_voxel_count(base_scan, VOXEL_IN_VM,
                                                                is_2d_vxl_mdl=True,
                                                                round_n=2)
        vm = VoxelModelLite(base_scan, voxel_size, is_2d_vxl_mdl=True)
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
        try:
            self.mse = (svv / sr) ** 0.5
            self.r = sr
        except ZeroDivisionError:
            self.mse = None
            self.r = 0
        if delete_temp_models:
            mesh_segment_model.delete_model()
        return triangles.values()


class MeshDB(MeshABC):
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
                stmt = update(Tables.triangles_db_table) \
                    .where(Tables.triangles_db_table.c.id == triangle.id) \
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
        with engine.connect() as db_connection:
            for triangle in self:
                stmt = update(Tables.triangles_db_table) \
                    .where(Tables.triangles_db_table.c.id == triangle.id) \
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
        triangle_data = []
        for triangle in triangulation.faces:
            triangle_data.append({"point_0_id": triangulation.points_id[triangle[0]],
                                  "point_1_id": triangulation.points_id[triangle[1]],
                                  "point_2_id": triangulation.points_id[triangle[2]],
                                  "mesh_id": self.id})
        db_conn.execute(Tables.triangles_db_table.insert(), triangle_data)
        db_conn.commit()

    def __init_mesh(self, db_connection=None, triangulation=None):
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
        self.id = db_mesh_data["id"]
        self.scan_name = db_mesh_data["mesh_name"]
        self.len = db_mesh_data["len"]
        self.r = db_mesh_data["r"]
        self.mse = db_mesh_data["mse"]
        self.base_scan_id = db_mesh_data["base_scan_id"]


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


class DeleterBadTriangleInMesh:
    def __init__(self, mesh):
        self._mesh = mesh
        self._deleter = self.__chose_deleter()

    def __chose_deleter(self):
        if isinstance(self._mesh, MeshDB):
            return MeshDBBadTriangleDeleter
        if isinstance(self._mesh, MeshLite):
            return MeshLiteBadTriangleDeleter

    def __recalculate_mesh_metrics(self):
        svv, sr = 0, 0
        len_ = 0
        for triangle in self._mesh:
            len_ += 1
            try:
                svv += triangle.r * triangle.mse ** 2
                sr += triangle.r
            except TypeError:
                continue
        try:
            mse = (svv / sr) ** 0.5
            r = sr
        except ZeroDivisionError:
            mse = None
            r = 0
        return {"len": len_, "mse": mse, "r": r}

    def delete_triangles_in_mesh(self, bad_triangles):
        self._deleter.deleting_logic(self._mesh, bad_triangles)
        metrics_dict = self.__recalculate_mesh_metrics()
        self._deleter.update_mesh_metrics(self._mesh, metrics_dict)


class MeshDBBadTriangleDeleter:
    @staticmethod
    def deleting_logic(mesh, bad_triangles):
        with engine.connect() as db_connection:
            for triangle_id in bad_triangles.values():
                stmt = delete(Tables.triangles_db_table) \
                    .where(and_(Tables.triangles_db_table.c.id == triangle_id,
                                Tables.triangles_db_table.c.mesh_id == mesh.id))
                db_connection.execute(stmt)
            db_connection.commit()

    @staticmethod
    def update_mesh_metrics(mesh, metrics_dict):
        with engine.connect() as db_connection:
            stmt = update(Tables.meshes_db_table) \
                .where(Tables.meshes_db_table.c.id == mesh.id) \
                .values(len=metrics_dict["len"],
                        r=metrics_dict["r"],
                        mse=metrics_dict["mse"])
            db_connection.execute(stmt)
            db_connection.commit()
        mesh.len = metrics_dict["len"]
        mesh.r = metrics_dict["r"]
        mesh.mse = metrics_dict["mse"]


class MeshLiteBadTriangleDeleter:
    @staticmethod
    def deleting_logic(mesh, bad_triangles):
        good_triangles = []
        for triangle in mesh:
            if triangle in bad_triangles:
                continue
            good_triangles.append(triangle)
        mesh.triangles = good_triangles

    @staticmethod
    def update_mesh_metrics(mesh, metrics_dict):
        mesh.len = metrics_dict["len"]
        mesh.r = metrics_dict["r"]
        mesh.mse = metrics_dict["mse"]


class MeshFilterABC(ABC):
    def __init__(self, mesh):
        self._mesh = mesh
        self._bad_triangles_deleter = DeleterBadTriangleInMesh(self._mesh)
        self._bad_triangles = {}

    @abstractmethod
    def _filter_logic(self, triangle):
        pass

    def filter_mesh(self):
        for triangle in self._mesh:
            if self._filter_logic(triangle) is False:
                self._bad_triangles[triangle] = triangle.id
        self._bad_triangles_deleter.delete_triangles_in_mesh(self._bad_triangles)


class Line:
    def __init__(self, point_0: Point, point_1: Point, id_=None):
        self.id = id_
        self.point_0 = point_0
        self.point_1 = point_1

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"[id: {self.id},\tp0: {self.point_0} p1: {self.point_1}]"

    def __repr__(self):
        return f"{self.__class__.__name__} [id: {self.id}]"

    def get_distance(self):
        return ((self.point_0.X - self.point_1.X) ** 2 +
                (self.point_0.Y - self.point_1.Y) ** 2 +
                (self.point_0.Z - self.point_1.Z) ** 2) ** 0.5

    def __get_y_by_x(self, x):
        x1, x2 = self.point_0.X, self.point_1.X
        y1, y2 = self.point_0.Y, self.point_1.Y
        y = ((x - x1) * (y2 - y1)) / (x2 - x1) + y1
        return y

    def __get_x_by_y(self, y):
        x1, x2 = self.point_0.X, self.point_1.X
        y1, y2 = self.point_0.Y, self.point_1.Y
        x = ((y - y1) * (x2 - x1)) / (y2 - y1) + x1
        return x

    def get_grid_cross_points_list(self, grid_step):
        points = set()
        x1, x2 = self.point_0.X, self.point_1.X
        y1, y2 = self.point_0.Y, self.point_1.Y
        points.add((x1, y1))
        points.add((x2, y2))
        x, y = min(x1, x2), min(y1, y2)
        x_max, y_max = max(x1, x2), max(y1, y2)
        while True:
            x += grid_step
            grid_x = x // grid_step * grid_step
            if grid_x < x_max:
                grid_y = self.__get_y_by_x(grid_x)
                points.add((grid_x, grid_y))
            else:
                break
        while True:
            y += grid_step
            grid_y = y // grid_step * grid_step
            if grid_y < y_max:
                grid_x = self.__get_x_by_y(grid_y)
                points.add((grid_x, grid_y))
            else:
                break
        points = sorted(list(points), key=lambda x: (x[0], x[1]))
        points = [Point(X=point[0], Y=point[1], Z=0,
                        R=0, G=0, B=0) for point in points]
        return points


class MaxEdgeLengthMeshFilter(MeshFilterABC):
    def __init__(self, mesh, max_edge_length):
        super().__init__(mesh)
        self.max_edge_length = max_edge_length

    def get_distance(self, line):
        return ((line.point_0.X - line.point_1.X) ** 2 +
                (line.point_0.Y - line.point_1.Y) ** 2) ** 0.5

    def _filter_logic(self, triangle):
        tr_edges = [Line(triangle.point_0, triangle.point_1),
                    Line(triangle.point_1, triangle.point_2),
                    Line(triangle.point_2, triangle.point_0)]
        for edge in tr_edges:
            if self.get_distance(edge) > self.max_edge_length:
                return False
        return True


class VoxelModelABC(ABC):
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
        return dx % 1, dy % 1, dz % 1

    def __name_generator(self, scan):
        vm_type = "2D" if self.is_2d_vxl_mdl else "3D"
        return f"VM_{vm_type}_Sc:{scan.scan_name}_st:{self.step}_dx:{self.dx:.2f}_dy:{self.dy:.2f}_dz:{self.dz:.2f}"

    def _calc_vxl_md_metric(self, scan):
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


class VMLiteSeparator:
    def __init__(self):
        self.voxel_model = None
        self.voxel_structure = None

    def separate_voxel_model(self, voxel_model, scan):
        voxel_model.logger.info(f"Начато создание структуры {voxel_model.vm_name}")
        self.__create_full_vxl_struct(voxel_model)
        voxel_model.logger.info(f"Структура {voxel_model.vm_name} создана")
        voxel_model.logger.info(f"Начат расчет метрик сканов и вокселей")
        self.__update_scan_and_voxel_data(scan)
        voxel_model.logger.info(f"Расчет метрик сканов и вокселей завершен")
        self.voxel_model.voxel_structure = self.voxel_structure

    def __create_full_vxl_struct(self, voxel_model):
        self.voxel_model = voxel_model
        id_generator = (id_ for id_ in range(1, len(self.voxel_model) + 1))
        self.voxel_structure = [[[VoxelLite(voxel_model.min_X + x * voxel_model.step,
                                            voxel_model.min_Y + y * voxel_model.step,
                                            voxel_model.min_Z + z * voxel_model.step,
                                            voxel_model.step, voxel_model.id,
                                            id_=-next(id_generator))
                                  for x in range(voxel_model.X_count)]
                                 for y in range(voxel_model.Y_count)]
                                for z in range(voxel_model.Z_count)]
        self.voxel_model.voxel_structure = self.voxel_structure

    def __update_scan_and_voxel_data(self, scan):
        for point in scan:
            vxl_md_X = int((point.X - self.voxel_model.min_X) // self.voxel_model.step)
            vxl_md_Y = int((point.Y - self.voxel_model.min_Y) // self.voxel_model.step)
            if self.voxel_model.is_2d_vxl_mdl:
                vxl_md_Z = 0
            else:
                vxl_md_Z = int((point.Z - self.voxel_model.min_Z) // self.voxel_model.step)
            self.__update_scan_data(self.voxel_structure[vxl_md_Z][vxl_md_Y][vxl_md_X].scan,
                                    point)
            self.__update_voxel_data(self.voxel_structure[vxl_md_Z][vxl_md_Y][vxl_md_X], point)

    @staticmethod
    def __update_scan_data(scan, point):
        scan.len += 1
        scan.update_scan_borders(scan, point)

    @staticmethod
    def __update_voxel_data(voxel, point):
        voxel.R = (voxel.R * voxel.len + point.R) / (voxel.len + 1)
        voxel.G = (voxel.G * voxel.len + point.G) / (voxel.len + 1)
        voxel.B = (voxel.B * voxel.len + point.B) / (voxel.len + 1)
        voxel.len += 1


class VMLiteFullBaseIterator:
    def __init__(self, vxl_mdl):
        self.vxl_mdl = vxl_mdl
        self.x = 0
        self.y = 0
        self.z = 0
        self.X_count, self.Y_count, self.Z_count = vxl_mdl.X_count, vxl_mdl.Y_count, vxl_mdl.Z_count

    def __iter__(self):
        return self

    def __next__(self):
        for vxl_z in range(self.z, self.Z_count):
            for vxl_y in range(self.y, self.Y_count):
                for vxl_x in range(self.x, self.X_count):
                    self.x += 1
                    voxel = self.vxl_mdl.voxel_structure[vxl_z][vxl_y][vxl_x]
                    if len(voxel) > 0:
                        return self.vxl_mdl.voxel_structure[vxl_z][vxl_y][vxl_x]
                self.y += 1
                self.x = 0
            self.z += 1
            self.y = 0
        raise StopIteration


class VoxelModelLite(VoxelModelABC):
    def __init__(self, scan, step, dx=0.0, dy=0.0, dz=0.0, is_2d_vxl_mdl=True,
                 voxel_model_separator=VMLiteSeparator()):
        super().__init__(scan, step, dx, dy, dz, is_2d_vxl_mdl)
        self.voxel_model_separator = voxel_model_separator
        self.voxel_structure = []
        self.__init_vxl_mdl(scan)

    def __iter__(self):
        return iter(VMLiteFullBaseIterator(self))

    def __init_vxl_mdl(self, scan):
        self.base_scan_id = scan.id
        self._calc_vxl_md_metric(scan)
        self.voxel_model_separator.separate_voxel_model(self, self.base_scan)


class VoxelDownsamplingScanSampler:
    def __init__(self, grid_step, is_2d_sampling=False, average_the_data=False):
        self.average_the_data = average_the_data
        self.is_2d_sampling = is_2d_sampling
        self.grid_step = grid_step
        self._voxel_model = None

    def do_sampling(self, scan):
        self.create_vm(scan)
        sample_func = self.average_sampling if self.average_the_data else self.central_point_sampling
        for point in scan:
            voxel = self.get_voxel_by_point(point)
            sample_func(voxel, point)
        sampled_scan = ScanLite(self.scan_name_generator(scan))
        for voxel in self._voxel_model:
            voxel_point = voxel.container_dict["voxel_point"]
            voxel_point.X = round(voxel_point.X, 3)
            voxel_point.Y = round(voxel_point.Y, 3)
            voxel_point.Z = round(voxel_point.Z, 3)
            voxel_point.R = round(voxel_point.R)
            voxel_point.G = round(voxel_point.G)
            voxel_point.B = round(voxel_point.B)
            sampled_scan.add_point(voxel_point)
            sampled_scan.update_scan_borders(sampled_scan, voxel_point)
        return sampled_scan

    def create_vm(self, scan):
        vm = VoxelModelLite(scan=scan, step=self.grid_step, is_2d_vxl_mdl=self.is_2d_sampling)
        self._voxel_model = vm

    def get_voxel_by_point(self, point):
        vxl_md_X = int((point.X - self._voxel_model.min_X) // self._voxel_model.step)
        vxl_md_Y = int((point.Y - self._voxel_model.min_Y) // self._voxel_model.step)
        if self._voxel_model.is_2d_vxl_mdl:
            vxl_md_Z = 0
        else:
            vxl_md_Z = int((point.Z - self._voxel_model.min_Z) // self._voxel_model.step)
        return self._voxel_model.voxel_structure[vxl_md_Z][vxl_md_Y][vxl_md_X]

    def scan_name_generator(self, scan):
        sampler_type = "Voxel_Average" if self.average_the_data else "Voxel_Center"
        sample_dimension = "2D" if self.is_2d_sampling else "3D"
        scan_name = f"Sampled_{scan.scan_name}_by_{sampler_type}_{sample_dimension}_step_{self.grid_step}m"
        return scan_name

    @staticmethod
    def average_sampling(voxel, point):
        if len(voxel.container_dict) == 0:
            voxel.container_dict["voxel_point"] = point
            voxel.container_dict["point_count"] = 1
        else:
            v_point = voxel.container_dict["voxel_point"]
            p_count = voxel.container_dict["point_count"]
            v_point.X = (v_point.X * p_count + point.X) / (p_count + 1)
            v_point.Y = (v_point.Y * p_count + point.Y) / (p_count + 1)
            v_point.Z = (v_point.Z * p_count + point.Z) / (p_count + 1)
            v_point.R = (v_point.R * p_count + point.R) / (p_count + 1)
            v_point.G = (v_point.G * p_count + point.G) / (p_count + 1)
            v_point.B = (v_point.B * p_count + point.B) / (p_count + 1)
            voxel.container_dict["voxel_point"] = v_point
            voxel.container_dict["point_count"] += 1

    def central_point_sampling(self, voxel, point):
        def calc_dist_to_voxel_center(vm, voxel, point):
            if vm.is_2d_vxl_mdl is True:
                dist = ((point.X - voxel.container_dict["c_point"].X) ** 2 +
                        (point.Y - voxel.container_dict["c_point"].Y) ** 2) ** 0.5
            else:
                dist = ((point.X - voxel.container_dict["c_point"].X) ** 2 +
                        (point.Y - voxel.container_dict["c_point"].Y) ** 2 +
                        (point.Z - voxel.container_dict["c_point"].Z) ** 2) ** 0.5
            return dist
        if len(voxel.container_dict) == 0:
            voxel.container_dict["voxel_point"] = point
            voxel.container_dict["c_point"] = Point(X=voxel.X + voxel.step / 2,
                                                    Y=voxel.Y + voxel.step / 2,
                                                    Z=voxel.Z + voxel.step / 2,
                                                    R=voxel.R, G=voxel.G, B=voxel.B)
            voxel.container_dict["min_dist"] = calc_dist_to_voxel_center(self._voxel_model, voxel, point)
        else:
            new_dist = calc_dist_to_voxel_center(self._voxel_model, voxel, point)
            if new_dist < voxel.container_dict["min_dist"]:
                voxel.container_dict["voxel_point"] = point
                voxel.container_dict["min_dist"] = new_dist


class MeshMSEConstDB:
    logger = logging.getLogger(LOGGER)
    db_table = Tables.meshes_db_table
    BORDER_LEN_COEF = 0.7

    def __init__(self, scan, max_border_length_m, max_triangle_mse_m, n=5, is_2d=True, calk_with_brute_force=False):
        self.scan = scan
        self.max_border_length = max_border_length_m
        self.max_triangle_mse = max_triangle_mse_m
        self.calk_with_brute_force = calk_with_brute_force
        self.n = n
        self.is_2d = is_2d
        self.mesh_name = self.__name_generator()
        self.mesh = None
        self.sampled_scan = None
        self.vm = None
        self.voxel_size = None
        self.temp_mesh = None
        self.loop_counter = 0
        self.count_of_bad_tr = 0

    def __iter__(self):
        return iter(self.mesh)

    def __len__(self):
        return len(self.mesh)

    def __name_generator(self):
        d_type = "2D" if self.is_2d else "3D"
        return (f"MESH_{self.scan.scan_name}"
                f"_max_brd_{round(self.max_border_length, 3)}"
                f"_max_mse_{self.max_triangle_mse}_n_{self.n}_{d_type}")

    def __init_mesh(self):
        select_ = select(self.db_table).where(self.db_table.c.mesh_name == self.mesh_name)
        with engine.connect() as db_connection:
            db_mesh_data = db_connection.execute(select_).mappings().first()
            if db_mesh_data is not None:
                mesh_id = db_mesh_data["id"]
                self.mesh = MeshDB.get_mesh_by_id(mesh_id)
                return True

    def calculate_mesh(self):
        if self.__init_mesh():
            return
        self.__do_prepare_calc()
        yield self.loop_counter
        for iteration in self.__do_basic_logic():
            yield self.loop_counter
        if self.calk_with_brute_force:
            self.__do_brute_force_calk()
        self.__end_logic()
        self.vm = None
        for tr in self.mesh:
            if tr.mse is not None and tr.mse > self.max_triangle_mse:
                self.count_of_bad_tr += 1

    def __do_prepare_calc(self):
        border_length = self.BORDER_LEN_COEF * self.max_border_length / 2 ** 0.5
        self.sampled_scan = VoxelDownsamplingScanSampler(grid_step=border_length,
                                                         is_2d_sampling=self.is_2d,
                                                         average_the_data=False).do_sampling(self.scan)
        self.voxel_size = self.__get_voxel_size_for_vm()
        self.vm = VoxelModelLite(self.scan, self.voxel_size, is_2d_vxl_mdl=True)
        self.mesh = MeshLite(self.sampled_scan)
        self.mesh.calk_mesh_mse(self.scan, voxel_size=self.voxel_size, delete_temp_models=True)
        self.temp_mesh = copy(self.mesh)

    def __do_basic_logic(self):
        while self.loop_counter < self.n:
            bad_triangles = self.__find_and_prepare_bad_triangles(self.temp_mesh)
            if len(bad_triangles) == 0:
                break
            self.temp_mesh.triangles = bad_triangles
            bad_triangles = self.__calc_mesh_triangles_centroids(self.temp_mesh)
            for triangle in bad_triangles:
                self.sampled_scan.add_point(triangle.container_dict["c_point"])
            self.temp_mesh.triangles = self.mesh.triangles
            self.mesh = MeshLite(self.sampled_scan)
            self.temp_mesh.triangles = self.__get_new_triangles(self.temp_mesh,
                                                                self.mesh)
            if len(self.temp_mesh.triangles) == 0:
                break
            self.temp_mesh.calk_mesh_mse(self.scan, voxel_size=self.voxel_size, clear_previous_mse=True,
                                         delete_temp_models=True)
            self.loop_counter += 1
            yield

    def __do_brute_force_calk(self):
        while self.loop_counter < self.n:
            self.temp_mesh = MeshLite(self.sampled_scan)
            self.temp_mesh.calk_mesh_mse(self.scan, voxel_size=self.voxel_size, clear_previous_mse=True,
                                         delete_temp_models=True)
            bad_triangles = self.__find_and_prepare_bad_triangles(self.temp_mesh)
            if len(bad_triangles) == 0:
                break
            self.temp_mesh.triangles = bad_triangles
            bad_triangles = self.__calc_mesh_triangles_centroids(self.temp_mesh)
            for triangle in bad_triangles:
                self.sampled_scan.add_point(triangle.container_dict["c_point"])
            self.loop_counter += 1
            print(self.loop_counter)

    def __end_logic(self):
        points = set()
        for point in self.sampled_scan:
            points.add(point)
        self.sampled_scan.scan_name = self.mesh_name[5:]
        self.sampled_scan._points = list(points)
        self.sampled_scan.save_to_db()
        self.mesh = MeshDB(self.sampled_scan)
        self.mesh.calk_mesh_mse(self.scan, delete_temp_models=True)
        MaxEdgeLengthMeshFilter(self.mesh, self.max_border_length).filter_mesh()

    @staticmethod
    def __get_new_triangles(prior_mesh, new_mesh):
        def get_key_for_triangle(tr):
            id_list = [point.id for point in tr]
            id_list.sort()
            return tuple(id_list)
        prior_triangles_keys = {get_key_for_triangle(tr) for tr in prior_mesh}
        new_triangles = []
        for triangle in new_mesh:
            tr_id = get_key_for_triangle(triangle)
            if tr_id not in prior_triangles_keys:
                new_triangles.append(triangle)
        return new_triangles

    def __get_voxel_size_for_vm(self):
        voxel_size = VoxelModelLite.get_step_by_voxel_count(self.scan, VOXEL_IN_VM,
                                                            is_2d_vxl_mdl=True,
                                                            round_n=2)
        return voxel_size

    @staticmethod
    def __calc_centroid_point_in_triangle(triangle):
        s_x, s_y, s_z = 0, 0, 0
        for point in triangle:
            s_x += point.X
            s_y += point.Y
            s_z += point.Z
        return Point(X=s_x / 3, Y=s_y / 3, Z=s_z / 3,
                     R=0, G=0, B=0)

    def __find_and_prepare_bad_triangles(self, mesh):
        bad_triangles = []
        for triangle in mesh:
            if triangle.mse is None:
                continue
            if triangle.mse > self.max_triangle_mse:
                triangle.container_dict["centroid_point"] = self.__calc_centroid_point_in_triangle(triangle)
                triangle.container_dict["dist"] = float("inf")
                triangle.container_dict["c_point"] = None
                bad_triangles.append(triangle)
        return bad_triangles

    def __calc_mesh_triangles_centroids(self, mesh):
        triangles = {}
        mesh_segment_model = MeshSegmentModelDB(self.vm, mesh)
        for point in self.scan:
            cell = mesh_segment_model.get_model_element_for_point(point)
            if cell is None or len(cell.triangles) == 0:
                continue
            for triangle in cell.triangles:
                if triangle.is_point_in_triangle(point):
                    if triangle.id not in triangles:
                        triangles[triangle.id] = triangle
                    dist = ((point.X - triangle.container_dict["centroid_point"].X) ** 2 +
                            (point.Y - triangle.container_dict["centroid_point"].Y) ** 2) ** 0.5
                    if dist < triangle.container_dict["dist"]:
                        triangle.container_dict["dist"] = dist
                        triangle.container_dict["c_point"] = point
                    break
        mesh_segment_model.delete_model()
        return triangles.values()


class MeshStatisticCalculator:
    def __init__(self, mesh, file_path="."):
        sns.set_style("darkgrid")
        self.mesh = mesh
        self.file_path = file_path
        try:
            self.df = pd.read_csv(os.path.join(self.file_path, f"{self.mesh.mesh_name}.csv"),
                                  delimiter=",")[["area", "r", "rmse"]]
        except FileNotFoundError:
            file_name = CsvMeshDataExporter(mesh).export_mesh_data(file_path=self.file_path)
            self.df = pd.read_csv(os.path.join(self.file_path, file_name), delimiter=",")[["area", "r", "rmse"]]
            os.remove(os.path.join(self.file_path, file_name))

    def get_statistic(self):
        statistic = self.df.describe()
        total_area = self.df["area"].sum()
        statistic.loc["TOTAL_MESH"] = [total_area, self.mesh.r, self.mesh.mse]
        return {"Total_area": total_area,
                "Count_of_r": self.mesh.r,
                "Cloud_MSE": self.mesh.mse,
                "Min_MSE": statistic["rmse"]["min"],
                "Max_MSE": statistic["rmse"]["max"],
                "Median_MSE": statistic["rmse"]["50%"]}

    def save_statistic(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        statistic = self.df.describe()
        total_area = self.df["area"].sum()
        statistic.loc["TOTAL_MESH"] = [total_area, self.mesh.r, self.mesh.mse]
        file_path = os.path.join(file_path, f"{self.mesh.mesh_name}_statistics.csv")
        statistic.to_csv(file_path)

    def save_area_distributions_histograms(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        sns_plot = sns.displot(self.df, x="area", kde=True)
        file_path = os.path.join(file_path, f"{self.mesh.mesh_name}_area_distribution.png")
        sns_plot.savefig(file_path)

    def save_r_distributions_histograms(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        sns_plot = sns.displot(self.df, x="r", kde=True)
        file_path = os.path.join(file_path, f"{self.mesh.mesh_name}_r_distribution.png")
        sns_plot.savefig(file_path)

    def save_rmse_distributions_histograms(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        sns_plot = sns.displot(self.df, x="rmse", kde=True)
        file_path = os.path.join(file_path, f"{self.mesh.mesh_name}_rmse_distribution.png")
        sns_plot.savefig(file_path)

    def save_pair_plot_distributions_histograms(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        pair_grid = sns.PairGrid(self.df)
        pair_grid.map_upper(sns.histplot)
        pair_grid.map_lower(sns.kdeplot, fill=True)
        pair_grid.map_diag(sns.histplot, kde=True)
        file_path = os.path.join(file_path, f"{self.mesh.mesh_name}_pair_grid.png")
        pair_grid.savefig(file_path)

    def save_distributions_histograms(self, graf_dict, file_path=None):
        if file_path is None:
            file_path = self.file_path
        if graf_dict["rmse"]:
            self.save_rmse_distributions_histograms(file_path=file_path)
        if graf_dict["r"]:
            self.save_r_distributions_histograms(file_path=file_path)
        if graf_dict["area"]:
            self.save_area_distributions_histograms(file_path=file_path)
        if graf_dict["pair_plot"]:
            self.save_pair_plot_distributions_histograms(file_path=file_path)


class MeshExporterABC(ABC):
    def __init__(self, mesh):
        self.mesh = mesh
        self.vertices = []
        self.vertices_colors = []
        self.faces = []
        self._init_base_data()

    def _init_base_data(self):
        points = {}
        triangles = []
        fake_id = -1
        for triangle in self.mesh:
            face_indexes = []
            triangles.append(triangle)
            for point in triangle:
                if point.id is None:
                    point.id = fake_id
                    fake_id -= 1
                if point in points:
                    face_indexes.append(points[point])
                else:
                    new_idx = len(points)
                    points[point] = new_idx
                    face_indexes.append(new_idx)
                    self.vertices.append([point.X, point.Y, point.Z])
                    self.vertices_colors.append([point.R, point.G, point.B])
            self.faces.append(face_indexes)

    @abstractmethod
    def export(self):
        pass


class DxfMeshExporter(MeshExporterABC):
    def __init__(self, mesh):
        super().__init__(mesh)

    def __save_dxf(self, file_path):
        doc = ezdxf.new("R2000")
        msp = doc.modelspace()
        mesh = msp.add_mesh()
        mesh.dxf.subdivision_levels = 0
        with mesh.edit_data() as mesh_data:
            mesh_data.vertices = self.vertices
            mesh_data.faces = self.faces
        doc.saveas(file_path)

    def export(self, file_path="."):
        file_path = os.path.join(file_path, f"{self.mesh.mesh_name.replace(':', '=')}.dxf")
        self.__save_dxf(file_path)


class PlyMeshExporter(MeshExporterABC):
    def __init__(self, mesh):
        super().__init__(mesh)

    def __create_header(self):
        return f"ply\n" \
               f"format ascii 1.0\n" \
               f"comment author: Mikhail Vystrchil\n" \
               f"comment object: {self.mesh.mesh_name}\n" \
               f"element vertex {len(self.vertices)}\n" \
               f"property float x\n" \
               f"property float y\n" \
               f"property float z\n" \
               f"property uchar red\n" \
               f"property uchar green\n" \
               f"property uchar blue\n" \
               f"element face {len(self.faces)}\n" \
               f"property list uchar int vertex_index\n" \
               f"end_header\n"

    def __create_vertices_str(self):
        vertices_str = ""
        for idx in range(len(self.vertices)):
            vertices_str += f"{self.vertices[idx][0]} {self.vertices[idx][1]} {self.vertices[idx][2]} " \
                            f"{self.vertices_colors[idx][0]} " \
                            f"{self.vertices_colors[idx][1]} " \
                            f"{self.vertices_colors[idx][2]}\n"
        return vertices_str

    def __create_faces_str(self):
        faces_str = ""
        for face in self.faces:
            faces_str += f"3 {face[0]} {face[1]} {face[2]}\n"
        return faces_str

    def _save_ply(self, file_path):
        with open(file_path, "wb") as file:
            file.write(self.__create_header().encode("ascii"))
            file.write(self.__create_vertices_str().encode("ascii"))
            file.write(self.__create_faces_str().encode("ascii"))

    def export(self, file_path="."):
        file_path = os.path.join(file_path, f"{self.mesh.mesh_name.replace(':', '=')}.ply")
        self._save_ply(file_path)


class PlyMseMeshExporter(PlyMeshExporter):
    def __init__(self, mesh, min_mse=None, max_mse=None):
        self.min_mse, self.max_mse = self.__get_mse_limits(mesh, min_mse, max_mse)
        super().__init__(mesh)

    @staticmethod
    def __get_mse_limits(mesh, min_mse, max_mse):
        if min_mse is not None and max_mse is not None:
            return min_mse, max_mse
        min_mesh_mse = float("inf")
        max_mesh_mse = 0
        for triangle in mesh:
            mse = triangle.mse
            if mse is None:
                continue
            if mse < min_mesh_mse:
                min_mesh_mse = mse
            if mse > max_mesh_mse:
                max_mesh_mse = mse
        if min_mesh_mse - max_mesh_mse == float("inf"):
            raise ValueError("В поверхности не расчитаны СКП!")
        if min_mse is not None:
            return min_mse, max_mesh_mse
        if max_mse is not None:
            return min_mesh_mse, max_mse
        return min_mesh_mse, max_mesh_mse

    def __get_color_for_mse(self, mse):
        if mse is None or mse == 0:
            return [0, 0, 255]
        if mse > self.max_mse:
            return [255, 0, 0]
        if mse < self.min_mse:
            return [0, 255, 0]
        half_mse_delta = (self.max_mse - self.min_mse) / 2
        mse = mse - half_mse_delta - self.min_mse
        gradient_color = 255 - round((255 * abs(mse)) / half_mse_delta)
        if mse > 0:
            return [255, gradient_color, 0]
        elif mse < 0:
            return [gradient_color, 255, 0]
        else:
            return [255, 255, 0]

    def _init_base_data(self):
        for triangle in self.mesh:
            face_indexes = []
            color_lst = self.__get_color_for_mse(triangle.mse)
            for point in triangle:
                self.vertices.append([point.X, point.Y, point.Z])
                self.vertices_colors.append(color_lst)
                face_indexes.append(len(self.vertices))
            self.faces.append(face_indexes)

    def export(self, file_path="."):
        file_path = os.path.join(file_path, f"MSE_{self.mesh.mesh_name.replace(':', '=')}"
                                            f"_MseLimits=[{self.min_mse:.3f}-{self.max_mse:.3f}].ply")
        self._save_ply(file_path)


class ASTinUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.base_filepath = None
        self.max_border_length_m = self.slider_max_border_lenght.value()
        self.target_mse = self.slider_target_mse.value()
        self.max_iteration_count = self.slider_iteration_counter.value()
        self.file_path_button.clicked.connect(self.open_file_dialog_base_filepath)
        self.file_path_text.textChanged.connect(self.base_filepath_from_text_line)
        self.slider_max_border_lenght.valueChanged.connect(self.sliders_update)
        self.slider_target_mse.valueChanged.connect(self.sliders_update)
        self.slider_iteration_counter.valueChanged.connect(self.sliders_update)
        self.sb_max_border_lenght.valueChanged.connect(self.sb_update)
        self.sb_target_mse.valueChanged.connect(self.sb_update)
        self.sb_iteration_counter.valueChanged.connect(self.sb_update)
        self.progressBar.setProperty("value", 0)
        self.start_button.clicked.connect(self.start_calculation)

    def start_calculation(self):
        self.start_button.setEnabled(False)
        dir_path = os.path.dirname(self.base_filepath)
        create_db()
        scan_name = os.path.basename(self.base_filepath).split(".")[0]
        scan = ScanDB(scan_name=scan_name)
        scan.load_scan_from_file(file_name=self.base_filepath)
        self.progressBar.setProperty("value", 10)
        self.target_mse = self.target_mse / 100
        mesh = MeshMSEConstDB(scan, max_border_length_m=self.max_border_length_m,
                              max_triangle_mse_m=self.target_mse,
                              n=self.max_iteration_count, calk_with_brute_force=False)
        progress = self.progressBar.value()
        progress_step = self.calc_progress_bar_step()
        for iteration in mesh.calculate_mesh():
            progress += progress_step
            self.progressBar.setProperty("value", progress)
        self.progressBar.setProperty("value", 80)
        if self.cb_save_full_tin_csv_log.isChecked():
            CsvMeshDataExporter(mesh.mesh).export_mesh_data(file_path=dir_path)
        stat_calculator = MeshStatisticCalculator(mesh.mesh, file_path=dir_path)
        if self.cb_save_base_stat.isChecked():
            MeshStatisticCalculator(mesh.mesh).save_statistic(file_path=dir_path)
        self.progressBar.setProperty("value", 85)
        if self.cb_save_scan_to_txt.isChecked():
            mesh.sampled_scan.save_scan_in_file(file_path=dir_path)
        if self.cb_save_dxf.isChecked():
            DxfMeshExporter(mesh=mesh).export(file_path=dir_path)
        if self.cb_ply_rgb.isChecked():
            PlyMeshExporter(mesh=mesh).export(file_path=dir_path)
        if self.cb_ply_mse.isChecked():
            PlyMseMeshExporter(mesh=mesh, min_mse=0, max_mse=self.target_mse).export(file_path=dir_path)
        self.progressBar.setProperty("value", 90)
        graf_dict = {"rmse": self.cb_mse_graf.isChecked(),
                     "r": self.cb_r_graf.isChecked(),
                     "area": self.cb_area_graf.isChecked(),
                     "pair_plot": self.cb_pair_plot_graf.isChecked()}
        stat_calculator.save_distributions_histograms(graf_dict, file_path=dir_path)
        self.progressBar.setProperty("value", 100)
        stat_dict = stat_calculator.get_statistic()
        self.result_table.setEnabled(True)
        self.result_table.setItem(0, 0, QTableWidgetItem(str(len(scan))))
        self.result_table.setItem(0, 1, QTableWidgetItem(str(len(mesh.sampled_scan))))
        self.result_table.setItem(0, 2, QTableWidgetItem(str(len(mesh.mesh))))
        self.result_table.setItem(0, 3, QTableWidgetItem(str(round(mesh.mesh.mse, 4))))
        self.result_table.setItem(0, 4, QTableWidgetItem(str(round(stat_dict["Max_MSE"], 4))))
        self.result_table.setItem(0, 5, QTableWidgetItem(str(mesh.count_of_bad_tr)))
        self.result_table.setItem(0, 6, QTableWidgetItem(str(mesh.loop_counter)))
        if self.cb_save_db.isChecked() is False:
            engine.dispose()
            os.remove(os.path.join(".", DATABASE_NAME))
        self.start_button.setEnabled(True)
        dig = QMessageBox(self)
        dig.setWindowTitle("Result")
        dig.setText("Расчет завершен!")
        dig.setIcon(QMessageBox.Icon.Information)
        dig.exec()

    def sb_update(self):
        self.max_border_length_m = self.sb_max_border_lenght.value()
        self.target_mse = self.sb_target_mse.value()
        self.max_iteration_count = self.sb_iteration_counter.value()
        self.slider_max_border_lenght.setValue(int(self.max_border_length_m))
        self.slider_target_mse.setValue(int(self.target_mse))
        self.slider_iteration_counter.setValue(int(self.max_iteration_count))

    def sliders_update(self):
        self.max_border_length_m = self.slider_max_border_lenght.value()
        self.target_mse = self.slider_target_mse.value()
        self.max_iteration_count = self.slider_iteration_counter.value()
        self.sb_max_border_lenght.setValue(self.max_border_length_m)
        self.sb_target_mse.setValue(self.target_mse)
        self.sb_iteration_counter.setValue(self.max_iteration_count)

    def open_file_dialog_base_filepath(self):
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Select a File",
            ".",
            "PointCloud (*.txt *.ascii *.xyz)"
        )
        if filename:
            path = Path(filename)
            self.file_path_text.setText(str(filename))
            self.base_filepath = str(path)

    def base_filepath_from_text_line(self):
        self.base_filepath = self.file_path_text.toPlainText()
        if self.base_filepath:
            self.start_button.setEnabled(True)
        else:
            self.start_button.setEnabled(False)

    def calc_progress_bar_step(self):
        return int(70 / (self.max_iteration_count + 1))

    def setupUi(self):
        self.setObjectName("AccurateSparseTIN")
        self.setWindowIcon(QIcon("icon.ico"))
        self.setEnabled(True)
        self.resize(1010, 615)
        self.setMinimumSize(QtCore.QSize(1010, 615))
        self.setMaximumSize(QtCore.QSize(1010, 590))
        self.verticalLayoutWidget = QWidget(parent=self)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 0, 996, 668))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.line_2 = QFrame(parent=self.verticalLayoutWidget)
        self.line_2.setFrameShadow(QFrame.Shadow.Plain)
        self.line_2.setLineWidth(28)
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setObjectName("line_2")
        self.gridLayout_4.addWidget(self.line_2, 2, 1, 1, 1)
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.slider_target_mse = QSlider(parent=self.verticalLayoutWidget)
        self.slider_target_mse.setMinimum(0)
        self.slider_target_mse.setMaximum(100)
        self.slider_target_mse.setProperty("value", 50)
        self.slider_target_mse.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider_target_mse.setObjectName("slider_target_mse")
        self.gridLayout_2.addWidget(self.slider_target_mse, 1, 1, 1, 1)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QLabel(parent=self.verticalLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.label_5 = QLabel(parent=self.verticalLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_2.addWidget(self.label_5)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_2, 4, 1, 1, 1)
        self.label_iteration_counter = QLabel(parent=self.verticalLayoutWidget)
        self.label_iteration_counter.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_iteration_counter.setObjectName("label_iteration_counter")
        self.gridLayout_4.addWidget(self.label_iteration_counter, 5, 0, 1, 1)
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        spacerItem1 = QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_2.addItem(spacerItem1)
        self.sb_iteration_counter = QSpinBox(parent=self.verticalLayoutWidget)
        self.sb_iteration_counter.setMinimum(0)
        self.sb_iteration_counter.setMaximum(50)
        self.sb_iteration_counter.setProperty("value", 10)
        self.sb_iteration_counter.setObjectName("sb_iteration_counter")
        self.verticalLayout_2.addWidget(self.sb_iteration_counter)
        self.gridLayout_4.addLayout(self.verticalLayout_2, 5, 2, 1, 1)
        self.label_target_mse = QLabel(parent=self.verticalLayoutWidget)
        self.label_target_mse.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_target_mse.setObjectName("label_target_mse")
        self.gridLayout_4.addWidget(self.label_target_mse, 4, 0, 1, 1)
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QLabel(parent=self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        spacerItem2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.label_2 = QLabel(parent=self.verticalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 1, 1, 1)
        self.slider_iteration_counter = QSlider(parent=self.verticalLayoutWidget)
        self.slider_iteration_counter.setMinimum(0)
        self.slider_iteration_counter.setMaximum(50)
        self.slider_iteration_counter.setProperty("value", 10)
        self.slider_iteration_counter.setSliderPosition(10)
        self.slider_iteration_counter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider_iteration_counter.setObjectName("slider_iteration_counter")
        self.gridLayout.addWidget(self.slider_iteration_counter, 1, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout, 5, 1, 1, 1)
        self.gridLayout_5 = QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.slider_max_border_lenght = QSlider(parent=self.verticalLayoutWidget)
        self.slider_max_border_lenght.setMinimum(1)
        self.slider_max_border_lenght.setMaximum(100)
        self.slider_max_border_lenght.setProperty("value", 20)
        self.slider_max_border_lenght.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider_max_border_lenght.setObjectName("slider_max_border_lenght")
        self.gridLayout_5.addWidget(self.slider_max_border_lenght, 1, 1, 1, 1)
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_8 = QLabel(parent=self.verticalLayoutWidget)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_4.addWidget(self.label_8)
        spacerItem3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem3)
        self.label_9 = QLabel(parent=self.verticalLayoutWidget)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_4.addWidget(self.label_9)
        self.gridLayout_5.addLayout(self.horizontalLayout_4, 0, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_5, 3, 1, 1, 1)
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        spacerItem4 = QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_3.addItem(spacerItem4)
        self.sb_target_mse = QDoubleSpinBox(parent=self.verticalLayoutWidget)
        self.sb_target_mse.setEnabled(True)
        self.sb_target_mse.setMaximumSize(QtCore.QSize(48, 16777215))
        self.sb_target_mse.setDecimals(1)
        self.sb_target_mse.setMaximum(100.0)
        self.sb_target_mse.setSingleStep(0.1)
        self.sb_target_mse.setProperty("value", 50.0)
        self.sb_target_mse.setObjectName("sb_target_mse")
        self.verticalLayout_3.addWidget(self.sb_target_mse)
        self.gridLayout_4.addLayout(self.verticalLayout_3, 4, 2, 1, 1)
        self.label_vax_border_length = QLabel(parent=self.verticalLayoutWidget)
        self.label_vax_border_length.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_vax_border_length.setObjectName("label_vax_border_length")
        self.gridLayout_4.addWidget(self.label_vax_border_length, 3, 0, 1, 1)
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        spacerItem5 = QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_4.addItem(spacerItem5)
        self.sb_max_border_lenght = QDoubleSpinBox(parent=self.verticalLayoutWidget)
        self.sb_max_border_lenght.setEnabled(True)
        self.sb_max_border_lenght.setMaximumSize(QtCore.QSize(48, 16777215))
        self.sb_max_border_lenght.setSingleStep(0.1)
        self.sb_max_border_lenght.setProperty("value", 20.0)
        self.sb_max_border_lenght.setObjectName("sb_max_border_lenght")
        self.verticalLayout_4.addWidget(self.sb_max_border_lenght)
        self.gridLayout_4.addLayout(self.verticalLayout_4, 3, 2, 1, 1)
        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        spacerItem6 = QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_5.addItem(spacerItem6)
        self.file_path_text = QTextEdit(parent=self.verticalLayoutWidget)
        self.file_path_text.setMinimumSize(QtCore.QSize(0, 20))
        self.file_path_text.setMaximumSize(QtCore.QSize(16777215, 25))
        self.file_path_text.setObjectName("file_path_text")
        self.verticalLayout_5.addWidget(self.file_path_text)
        spacerItem7 = QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_5.addItem(spacerItem7)
        self.gridLayout_4.addLayout(self.verticalLayout_5, 1, 1, 1, 1)
        self.Label_scan_name = QLabel(parent=self.verticalLayoutWidget)
        self.Label_scan_name.setMaximumSize(QtCore.QSize(16777215, 100))
        self.Label_scan_name.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.Label_scan_name.setObjectName("Label_scan_name")
        self.gridLayout_4.addWidget(self.Label_scan_name, 1, 0, 1, 1)
        self.line_3 = QFrame(parent=self.verticalLayoutWidget)
        self.line_3.setFrameShadow(QFrame.Shadow.Plain)
        self.line_3.setLineWidth(28)
        self.line_3.setFrameShape(QFrame.Shape.HLine)
        self.line_3.setObjectName("line_3")
        self.gridLayout_4.addWidget(self.line_3, 2, 2, 1, 1)
        self.line = QFrame(parent=self.verticalLayoutWidget)
        self.line.setFrameShadow(QFrame.Shadow.Plain)
        self.line.setLineWidth(28)
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setObjectName("line")
        self.gridLayout_4.addWidget(self.line, 2, 0, 1, 1)
        self.verticalLayout_8 = QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        spacerItem8 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_8.addItem(spacerItem8)
        self.file_path_button = QToolButton(parent=self.verticalLayoutWidget)
        self.file_path_button.setObjectName("file_path_button")
        self.verticalLayout_8.addWidget(self.file_path_button)
        spacerItem9 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_8.addItem(spacerItem9)
        self.gridLayout_4.addLayout(self.verticalLayout_8, 1, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_4)
        self.gridLayout_6 = QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.line_7 = QFrame(parent=self.verticalLayoutWidget)
        self.line_7.setFrameShadow(QFrame.Shadow.Plain)
        self.line_7.setLineWidth(28)
        self.line_7.setFrameShape(QFrame.Shape.HLine)
        self.line_7.setObjectName("line_7")
        self.gridLayout_6.addWidget(self.line_7, 0, 1, 1, 1)
        self.label_16 = QLabel(parent=self.verticalLayoutWidget)
        self.label_16.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_16.setObjectName("label_16")
        self.gridLayout_6.addWidget(self.label_16, 1, 0, 1, 1)
        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.cb_ply_rgb = QCheckBox(parent=self.verticalLayoutWidget)
        self.cb_ply_rgb.setEnabled(True)
        self.cb_ply_rgb.setObjectName("cb_ply_rgb")
        self.horizontalLayout_5.addWidget(self.cb_ply_rgb)
        self.gridLayout_3.addLayout(self.horizontalLayout_5, 3, 3, 1, 1)
        self.cb_r_graf = QCheckBox(parent=self.verticalLayoutWidget)
        self.cb_r_graf.setEnabled(True)
        self.cb_r_graf.setObjectName("cb_r_graf")
        self.gridLayout_3.addWidget(self.cb_r_graf, 3, 2, 1, 1)
        self.line_5 = QFrame(parent=self.verticalLayoutWidget)
        self.line_5.setFrameShadow(QFrame.Shadow.Plain)
        self.line_5.setLineWidth(28)
        self.line_5.setFrameShape(QFrame.Shape.HLine)
        self.line_5.setObjectName("line_5")
        self.gridLayout_3.addWidget(self.line_5, 1, 3, 1, 1)
        self.gridLayout_8 = QGridLayout()
        self.gridLayout_8.setObjectName("gridLayout_8")
        spacerItem10 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.gridLayout_8.addItem(spacerItem10, 0, 0, 1, 1)
        self.label_13 = QLabel(parent=self.verticalLayoutWidget)
        self.label_13.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.label_13.setObjectName("label_13")
        self.gridLayout_8.addWidget(self.label_13, 0, 1, 1, 1)
        spacerItem11 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.gridLayout_8.addItem(spacerItem11, 0, 2, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_8, 0, 2, 1, 1)
        self.gridLayout_7 = QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        spacerItem12 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.gridLayout_7.addItem(spacerItem12, 0, 2, 1, 1)
        self.label_11 = QLabel(parent=self.verticalLayoutWidget)
        self.label_11.setObjectName("label_11")
        self.gridLayout_7.addWidget(self.label_11, 0, 1, 1, 1)
        spacerItem13 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.gridLayout_7.addItem(spacerItem13, 0, 0, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_7, 0, 0, 1, 1)
        self.cb_mse_graf = QCheckBox(parent=self.verticalLayoutWidget)
        self.cb_mse_graf.setEnabled(True)
        self.cb_mse_graf.setObjectName("cb_mse_graf")
        self.gridLayout_3.addWidget(self.cb_mse_graf, 2, 2, 1, 1)
        self.cb_area_graf = QCheckBox(parent=self.verticalLayoutWidget)
        self.cb_area_graf.setEnabled(True)
        self.cb_area_graf.setObjectName("cb_area_graf")
        self.gridLayout_3.addWidget(self.cb_area_graf, 4, 2, 1, 1)
        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.gridLayout_9 = QGridLayout()
        self.gridLayout_9.setObjectName("gridLayout_9")
        spacerItem14 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.gridLayout_9.addItem(spacerItem14, 0, 0, 1, 1)
        spacerItem15 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.gridLayout_9.addItem(spacerItem15, 0, 2, 1, 1)
        self.label_14 = QLabel(parent=self.verticalLayoutWidget)
        self.label_14.setObjectName("label_14")
        self.gridLayout_9.addWidget(self.label_14, 0, 1, 1, 1)
        self.horizontalLayout_6.addLayout(self.gridLayout_9)
        self.gridLayout_3.addLayout(self.horizontalLayout_6, 0, 3, 1, 1)
        self.line_6 = QFrame(parent=self.verticalLayoutWidget)
        self.line_6.setFrameShadow(QFrame.Shadow.Plain)
        self.line_6.setLineWidth(28)
        self.line_6.setFrameShape(QFrame.Shape.HLine)
        self.line_6.setObjectName("line_6")
        self.gridLayout_3.addWidget(self.line_6, 1, 2, 1, 1)
        self.line_4 = QFrame(parent=self.verticalLayoutWidget)
        self.line_4.setFrameShadow(QFrame.Shadow.Plain)
        self.line_4.setLineWidth(28)
        self.line_4.setFrameShape(QFrame.Shape.HLine)
        self.line_4.setObjectName("line_4")
        self.gridLayout_3.addWidget(self.line_4, 1, 0, 1, 1)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.cb_save_dxf = QCheckBox(parent=self.verticalLayoutWidget)
        self.cb_save_dxf.setEnabled(True)
        self.cb_save_dxf.setChecked(True)
        self.cb_save_dxf.setObjectName("cb_save_dxf")
        self.horizontalLayout_3.addWidget(self.cb_save_dxf)
        self.gridLayout_3.addLayout(self.horizontalLayout_3, 2, 3, 1, 1)
        self.cb_pair_plot_graf = QCheckBox(parent=self.verticalLayoutWidget)
        self.cb_pair_plot_graf.setEnabled(True)
        self.cb_pair_plot_graf.setObjectName("cb_pair_plot_graf")
        self.gridLayout_3.addWidget(self.cb_pair_plot_graf, 5, 2, 1, 1)
        self.cb_save_db = QCheckBox(parent=self.verticalLayoutWidget)
        self.cb_save_db.setEnabled(True)
        self.cb_save_db.setChecked(False)
        self.cb_save_db.setObjectName("cb_save_res_scan_to_txt")
        self.gridLayout_3.addWidget(self.cb_save_db, 5, 0, 1, 1)
        self.cb_save_full_tin_csv_log = QCheckBox(parent=self.verticalLayoutWidget)
        self.cb_save_full_tin_csv_log.setEnabled(True)
        self.cb_save_full_tin_csv_log.setObjectName("cb_save_full_tin_csv_log")
        self.gridLayout_3.addWidget(self.cb_save_full_tin_csv_log, 4, 0, 1, 1)
        self.cb_save_scan_to_txt = QCheckBox(parent=self.verticalLayoutWidget)
        self.cb_save_scan_to_txt.setEnabled(True)
        self.cb_save_scan_to_txt.setChecked(True)
        self.cb_save_scan_to_txt.setObjectName("cb_save_db")
        self.gridLayout_3.addWidget(self.cb_save_scan_to_txt, 2, 0, 1, 1)
        self.cb_save_base_stat = QCheckBox(parent=self.verticalLayoutWidget)
        self.cb_save_base_stat.setEnabled(True)
        self.cb_save_base_stat.setObjectName("cb_save_base_stat")
        self.gridLayout_3.addWidget(self.cb_save_base_stat, 3, 0, 1, 1)
        self.cb_ply_mse = QCheckBox(parent=self.verticalLayoutWidget)
        self.cb_ply_mse.setEnabled(True)
        self.cb_ply_mse.setObjectName("cb_ply_mse")
        self.gridLayout_3.addWidget(self.cb_ply_mse, 4, 3, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_3, 1, 1, 1, 1)
        self.line_8 = QFrame(parent=self.verticalLayoutWidget)
        self.line_8.setFrameShadow(QFrame.Shadow.Plain)
        self.line_8.setLineWidth(28)
        self.line_8.setFrameShape(QFrame.Shape.HLine)
        self.line_8.setObjectName("line_8")
        self.gridLayout_6.addWidget(self.line_8, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_6)
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.progressBar = QProgressBar(parent=self.verticalLayoutWidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_7.addWidget(self.progressBar)
        self.start_button = QPushButton(parent=self.verticalLayoutWidget)
        self.start_button.setEnabled(False)
        self.start_button.setStyleSheet("background-color: rgb(170, 255, 127);")
        self.start_button.setFlat(False)
        self.start_button.setObjectName("start_button")
        self.horizontalLayout_7.addWidget(self.start_button)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.result_table = QTableWidget(parent=self.verticalLayoutWidget)
        self.result_table.setEnabled(False)
        self.result_table.setMinimumSize(QtCore.QSize(994, 109))
        self.result_table.setMaximumSize(QtCore.QSize(994, 109))
        self.result_table.setObjectName("result_table")
        self.result_table.setColumnCount(7)
        self.result_table.setRowCount(1)
        self.result_table.setColumnWidth(0, 130)
        self.result_table.setColumnWidth(1, 130)
        self.result_table.setColumnWidth(2, 120)
        self.result_table.setColumnWidth(3, 100)
        self.result_table.setColumnWidth(4, 60)
        self.result_table.setColumnWidth(5, 180)
        self.result_table.setColumnWidth(6, 90)
        item = QTableWidgetItem()
        self.result_table.setVerticalHeaderItem(0, item)
        item = QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setHorizontalHeaderItem(0, item)
        item = QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setHorizontalHeaderItem(1, item)
        item = QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setHorizontalHeaderItem(2, item)
        item = QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setHorizontalHeaderItem(3, item)
        item = QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setHorizontalHeaderItem(4, item)
        item = QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setHorizontalHeaderItem(5, item)
        item = QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setHorizontalHeaderItem(6, item)
        self.verticalLayout.addWidget(self.result_table)
        self.retranslateUi()
        self.slider_iteration_counter.sliderMoved['int'].connect(self.sb_iteration_counter.setValue)
        self.sb_iteration_counter.valueChanged['int'].connect(self.slider_iteration_counter.setValue)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Form", "AccurateSparseTIN"))
        self.label_4.setText(_translate("Form", "Точнее"))
        self.label_5.setText(_translate("Form", "Грубее"))
        self.label_iteration_counter.setText(_translate("Form", "Количество\nитераций:"))
        self.label_target_mse.setText(_translate("Form", "Целевая\nСКП\nповерхности, см:"))
        self.label.setText(_translate("Form", "Быстро"))
        self.label_2.setText(_translate("Form", "Долго"))
        self.label_8.setText(_translate("Form", "Меньше"))
        self.label_9.setText(_translate("Form", "Больше"))
        self.label_vax_border_length.setText(_translate("Form", "Максимальная\n"
                                                                "сторона\nполигона, м:"))
        self.Label_scan_name.setText(_translate("Form", "Исходный\nскан:"))
        self.file_path_button.setText(_translate("Form", "..."))
        self.label_16.setText(_translate("Form", "Настройки\nвыводимой\nинформации:"))
        self.cb_ply_rgb.setText(_translate("Form", "Сохранить TIN поверхность в PLY формате\n"
                                                   "(цвет RGB  исходного облака)"))
        self.cb_r_graf.setText(_translate("Form", "Распределение количества\nизбыточных данных"))
        self.label_13.setText(_translate("Form", "Графики распределения\n"
                                                 "параметров TIN поверхности"))
        self.label_11.setText(_translate("Form", "Сохраняемая информация"))
        self.cb_mse_graf.setText(_translate("Form", "Распределение СКП\n"
                                                    "в треугольниках поверхности"))
        self.cb_area_graf.setText(_translate("Form", "Распределение площади\n"
                                                     "треугольников в поверхности"))
        self.label_14.setText(_translate("Form", "Создать TIN поверхность"))
        self.cb_save_dxf.setText(_translate("Form", "Сохранить TIN поверхность в DXF формате"))
        self.cb_pair_plot_graf.setText(_translate("Form", "Совмещенный график\nраспределения"))
        self.cb_save_db.setText(_translate("Form", "Сохранить служебную базу данных"))
        self.cb_save_full_tin_csv_log.setText(_translate("Form", "Сохранить полное описание "
                                                                 "результрующей\nповерхности в CSV файле"))
        self.cb_save_scan_to_txt.setText(_translate("Form", "Сохранить результирующий скан "
                                                            "в TXT формате"))
        self.cb_save_base_stat.setText(_translate("Form", "Сохранить общую статистику\n"
                                                          "поверхности в CSV файле"))
        self.cb_ply_mse.setText(_translate("Form", "Сохранить TIN поверхность в PLY формате\n"
                                                   "(цвет пропорционален СКП)"))
        self.start_button.setText(_translate("Form", "Запуск разрежения"))
        item = self.result_table.verticalHeaderItem(0)
        item.setText(_translate("Form", "Рассчитанные значения"))
        item = self.result_table.horizontalHeaderItem(0)
        item.setText(_translate("Form", "К-во точек в исх.облаке"))
        item = self.result_table.horizontalHeaderItem(1)
        item.setText(_translate("Form", "К-во в точек раз.облаке"))
        item = self.result_table.horizontalHeaderItem(2)
        item.setText(_translate("Form", "К-во полигонов в TIN"))
        item = self.result_table.horizontalHeaderItem(3)
        item.setText(_translate("Form", "СКП поверхности"))
        item = self.result_table.horizontalHeaderItem(4)
        item.setText(_translate("Form", "Макс.СКП"))
        item = self.result_table.horizontalHeaderItem(5)
        item.setText(_translate("Form", "К-во п-ов, превышающих уст.СКП"))
        item = self.result_table.horizontalHeaderItem(6)
        item.setText(_translate("Form", "К-во итераций"))


if __name__ == "__main__":
    # import sys
    app = QApplication(sys.argv)
    ui = ASTinUI()
    ui.show()
    sys.exit(app.exec())
