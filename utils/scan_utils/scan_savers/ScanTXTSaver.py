from os import path

from utils.start_db import logger


class ScanTXTSaver:

    def save_scan(self, scan, file_path, scan_name):
        if scan_name is None:
            file_path = path.join(file_path, f"{scan.scan_name}.txt")
        else:
            file_path = path.join(file_path, f"{scan_name}.txt")
        with open(file_path, "w", encoding="UTF-8") as file:
            for point in scan:
                point_line = f"{point.X} {point.Y} {point.Z} {point.R} {point.G} {point.B}\n"
                file.write(point_line)
        logger.info(f"Сохранение скана {scan} в файл {file_path} завершено")
