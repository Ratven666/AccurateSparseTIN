from utils.scan_utils.scan_samplers.ScanSamplerABC import ScanSamplerABC


class TotalPointCountScanSampler(ScanSamplerABC):
    """
    Разрядитель плотности точек в скане, который оставляет в результирующем скане
    количество точек не превышающее переданное значение total_point_count
    """
    def __init__(self, total_point_count):
        super().__init__()
        self.__total_point_count = total_point_count

    def do_sampling(self, scan):
        """
        Запускает процедуру разрежения
        :param scan: Исходный скан
        :return: объкет ScanLite с метриками базового скана и оставшимися точками
        """
        from classes.ScanLite import ScanLite

        sample_scan = ScanLite.create_from_another_scan(scan, copy_with_points=False)
        if len(scan) < self.__total_point_count:
            step = 1
        else:
            step = int(len(scan) / self.__total_point_count)
        counter = 0
        for point in scan:
            if counter == step:
                counter = 0
            if counter == 0:
                sample_scan.add_point(point)
            counter += 1
        return sample_scan

    @property
    def total_point_count(self):
        return self.__total_point_count

    @total_point_count.setter
    def total_point_count(self, new_count):
        if isinstance(new_count, int):
            self.__total_point_count = new_count
        else:
            raise TypeError(f"Должно быть целое количество точек. "
                            f"Переданно - {type(new_count)}, {new_count}")
