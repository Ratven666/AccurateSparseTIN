import logging
from abc import ABC, abstractmethod
from os import remove

from sqlalchemy import delete

from CONFIG import LOGGER, POINTS_CHUNK_COUNT
from classes.ScanDB import ScanDB
from utils.scan_utils.Scan_metrics import update_scan_in_db_from_scan, update_scan_metrics
from utils.start_db import engine, Tables


class ScanFilterABC(ABC):
    logger = logging.getLogger(LOGGER)

    def __init__(self, scan):
        self.scan = scan

    def __scan_name_generator(self):
        return f"{self.scan.scan_name}_FB_{self.__class__.__name__}"

    @abstractmethod
    def _filter_logic(self, point):
        pass

    def filter_scan(self):
        self.__write_temp_points_scans_file(self.scan)
        with engine.connect() as db_connection:
            stmt = delete(Tables.points_scans_db_table) \
                .where(Tables.points_scans_db_table.c.scan_id == self.scan.id)
            db_connection.execute(stmt)
            db_connection.commit()
            for data in self.__parse_temp_points_scans_file():
                db_connection.execute(Tables.points_scans_db_table.insert(), data)
                self.logger.info(f"Пакет отфильтрованных точек загружен в БД")
            db_connection.commit()
        update_scan_metrics(self.scan)
        update_scan_in_db_from_scan(self.scan)
        return ScanDB(self.scan.scan_name)

    def create_new_filtered_scan(self, force_filter=False):
        new_scan = ScanDB(self.__scan_name_generator())
        if len(new_scan) != 0:
            if force_filter:
                with engine.connect() as db_connection:
                    stmt = delete(Tables.points_scans_db_table) \
                        .where(Tables.points_scans_db_table.c.scan_id == new_scan.id)
                    db_connection.execute(stmt)
                    db_connection.commit()
            else:
                self.logger.warning(f"Скан {self.scan.scan_name} уже отфильтрован с такими параметрами")
                return new_scan
        self.__write_temp_points_scans_file(new_scan)
        with engine.connect() as db_connection:
            for data in self.__parse_temp_points_scans_file():
                db_connection.execute(Tables.points_scans_db_table.insert(), data)
                self.logger.info(f"Пакет отфильтрованных точек загружен в БД")
            db_connection.commit()

        update_scan_metrics(new_scan)
        update_scan_in_db_from_scan(new_scan)
        return ScanDB(self.__scan_name_generator())

    def __write_temp_points_scans_file(self, scan):
        """
        Записывает временный файл, определяющий связь точек со сканом вокселя
        :param scan: базовый скан
        :return: None
        Рассчитывает и записывает во временный файл пару id точки и скана в вокселе
        Обновляет занчения метрик скана и вокселя в который попадает текущая точка
        """
        with open("temp_file.txt", "w", encoding="UTF-8") as file:
            for point in self.scan:
                if self._filter_logic(point) is True:
                    file.write(f"{point.id}, {scan.id}\n")

    @staticmethod
    def __parse_temp_points_scans_file():
        """
        Читает временный файл, определяющий связь точек со сканом вокселя,
        выдает пакетами данные для загрузки их в БД
        в конце удаляет временный файл
        :return: пакеты точек для загрузки в БД
        """
        points_scans = []
        try:
            with open("temp_file.txt", "r", encoding="UTF-8") as file:
                for line in file:
                    data = [int(p_ps) for p_ps in line.strip().split(",")]
                    points_scans.append({"point_id": data[0], "scan_id": data[1]})
                    if len(points_scans) == POINTS_CHUNK_COUNT:
                        yield points_scans
                        points_scans = []
            yield points_scans
        finally:
            try:
                remove("temp_file.txt")
            except FileNotFoundError:
                pass
