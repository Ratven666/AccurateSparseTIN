from utils.start_db import logger


class ScanTXTSaver:

    def save_scan(self, scan, file_name):
        if file_name is None:
            file_name = f"./{scan.scan_name}.txt"
        with open(file_name, "w", encoding="UTF-8") as file:
            for point in scan:
                point_line = f"{point.X} {point.Y} {point.Z} {point.R} {point.G} {point.B}\n"
                file.write(point_line)
        logger.info(f"Сохранение скана {scan} в файл {file_name} завершено")
