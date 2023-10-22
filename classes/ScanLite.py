from sqlalchemy import select, desc
from sqlalchemy.exc import IntegrityError

from classes.ScanDB import ScanDB
from classes.abc_classes.PointABC import PointABC
from classes.abc_classes.ScanABC import ScanABC
from utils.scan_utils.Scan_metrics import update_scan_borders, insert_scan_in_db_from_scan
from utils.start_db import engine, Tables


class ScanLite(ScanABC):
    """
    Скан не связанный с базой данных
    Все данные, включая точки при переборе берутся из оперативной памяти
    """

    def __init__(self, scan_name):
        super().__init__(scan_name)
        self._points = []

    def __iter__(self):
        return iter(self._points)

    def __len__(self):
        return len(self._points)

    def add_point(self, point):
        """
        Добавляет точку в скан
        :param point: объект класса Point
        :return: None
        """
        if isinstance(point, PointABC):
            self._points.append(point)
            self.len += 1
            update_scan_borders(self, point)
        else:
            raise TypeError(f"Можно добавить только объект точки. "
                            f"Переданно - {type(point)}, {point}")

    @classmethod
    def create_from_another_scan(cls, scan, copy_with_points=True):
        """
        Создает скан типа ScanLite и копирует в него данные из другого скана
        :param scan: копируемый скан
        :param copy_with_points: определяет нужно ли копировать скан вместе с точками
        :type copy_with_points: bool
        :return: объект класса ScanLite
        """
        scan_lite = cls(scan.scan_name)
        scan_lite.id = scan.id
        scan_lite.len = 0
        scan_lite.min_X, scan_lite.min_Y, scan_lite.min_Z = scan.min_X, scan.min_Y, scan.min_Z
        scan_lite.max_X, scan_lite.max_Y, scan_lite.max_Z = scan.max_X, scan.max_Y, scan.max_Z
        if copy_with_points:
            scan_lite._points = [point for point in scan]
            scan_lite.len = len(scan_lite._points)
        return scan_lite

    @classmethod
    def create_from_scan_dict(cls, scan_dict):
        """Создает объект ScanLite с метрикаи переданными в словаре без точек
        :param scan_dict: словарь с метриками скана
        :return: объект класса ScanLite
        """
        scan_lite = cls(scan_dict["scan_name"])
        scan_lite.id = scan_dict["id"]
        scan_lite.len = scan_dict["len"]
        scan_lite.min_X, scan_lite.min_Y, scan_lite.min_Z = scan_dict["min_X"], scan_dict["min_Y"], scan_dict["min_Z"]
        scan_lite.max_X, scan_lite.max_Y, scan_lite.max_Z = scan_dict["max_X"], scan_dict["max_Y"], scan_dict["max_Z"]
        return scan_lite

    def save_to_db(self):
        """Сохраняет объект ScanLite в базе данных вместе с точками"""
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
                insert_scan_in_db_from_scan(self, db_connection)
                if len(points) > 0:
                    db_connection.execute(Tables.points_db_table.insert(), points)
                    db_connection.execute(Tables.points_scans_db_table.insert(), points_scans)
                db_connection.commit()
                return ScanDB(self.scan_name)
            except IntegrityError:
                self.logger.warning("Скан с таким именем уже есть в БД!")
                return ScanDB(self.scan_name)
