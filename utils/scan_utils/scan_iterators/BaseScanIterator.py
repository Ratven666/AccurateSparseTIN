from sqlalchemy import select

from classes.Point import Point
from utils.start_db import engine, Tables


class BaseScanIterator:
    """
    Универсальный иттератор для сканов из БД
    Реализован средствами sqlalchemy
    """
    def __init__(self, scan):
        self.__scan = scan
        self.__engine = engine.connect()
        self.__select = select(Tables.points_db_table).\
            join(Tables.points_scans_db_table, Tables.points_scans_db_table.c.point_id == Tables.points_db_table.c.id).\
            where(self.__scan.id == Tables.points_scans_db_table.c.scan_id)
        self.__query = self.__engine.execute(self.__select)
        self.__iterator = None

    def __iter__(self):
        self.__iterator = iter(self.__query)
        return self

    def __next__(self):
        try:
            row = next(self.__iterator)
            point = Point.parse_point_from_db_row(row)
            return point
        except StopIteration:
            self.__engine.close()
            raise StopIteration
        finally:
            self.__engine.close()
