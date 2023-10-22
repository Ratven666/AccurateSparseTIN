from classes.ScanDB import ScanDB
from classes.VoxelDB import VoxelDB
from utils.scan_utils.Scan_metrics import update_scan_borders, update_scan_in_db_from_scan
from utils.start_db import engine
from utils.voxel_utils.Voxel_metrics import update_voxel_in_db_from_voxel
from utils.voxel_utils.Voxel_model_metrics import update_voxel_model_in_db_from_voxel_model
from utils.voxel_utils.voxel_model_iterators.VMFullBaseIterator import VMFullBaseIterator
from utils.voxel_utils.voxel_model_separators.VMSeparatorABC import VMSeparatorABC


class VMBruteForceSeparatorWithoutVoxelScansPoints(VMSeparatorABC):
    """Сепоратор воксельной модели через создание полной воксельной структуры"""

    def __init__(self):
        self.voxel_model = None
        self.voxel_structure = None

    def separate_voxel_model(self, voxel_model, scan):
        """
        Общая логика разбиения воксельной модели
        :param voxel_model: воксельная модель
        :param scan: скан
        :return: None
        1. Создается полная воксельная структура
        2. Скан разбивается на отдельные воксели
        3. Обновляет метрики сканов и вокселей, и удаляет пустые
        """
        voxel_model.logger.info(f"Начато создание структуры {voxel_model.vm_name}")
        self.__create_full_vxl_struct(voxel_model)
        voxel_model.logger.info(f"Структура {voxel_model.vm_name} создана")
        voxel_model.logger.info(f"Начат расчет метрик сканов и вокселей")
        self.__update_scan_and_voxel_data(scan)
        voxel_model.logger.info(f"Расчет метрик сканов и вокселей завершен")
        voxel_model.logger.info(f"Начата загрузка метрик сканов и вокселей в БД")
        self.__update_scan_and_voxel_data_in_db()
        voxel_model.logger.info(f"Загрузка метрик сканов и вокселей в БД завершена")

    def __create_full_vxl_struct(self, voxel_model):
        """
        Создается полная воксельная структура с записью каждого вокселя в БД
        :param voxel_model: воксельная модель
        :return: None
        """
        self.voxel_model = voxel_model
        with engine.connect() as db_connection:
            self.voxel_structure = [[[VoxelDB(voxel_model.min_X + x * voxel_model.step,
                                              voxel_model.min_Y + y * voxel_model.step,
                                              voxel_model.min_Z + z * voxel_model.step,
                                              voxel_model.step, voxel_model.id, db_connection)
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
        """
        Обновляет значения метрик скана (количество точек и границы)
        :param scan: обновляемый скан
        :param point: добавляемая в скан точка
        :return: None
        """
        scan.len += 1
        update_scan_borders(scan, point)

    @staticmethod
    def __update_voxel_data(voxel: VoxelDB, point):
        """
        Обновляет значения метрик вокселя (цвет и количество точек)
        :param voxel: обновляемый воксель
        :param point: точка, попавшая в воксель
        :return:
        """
        voxel.R = (voxel.R * voxel.len + point.R) / (voxel.len + 1)
        voxel.G = (voxel.G * voxel.len + point.G) / (voxel.len + 1)
        voxel.B = (voxel.B * voxel.len + point.B) / (voxel.len + 1)
        voxel.len += 1

    def __update_scan_and_voxel_data_in_db(self):
        """
        Обновляет значения метрик сканов и вокселей в БД
        и удаляет из БД пустые сканы и вокскли
        :return: None
        """
        voxel_counter = 0
        with engine.connect() as db_connection:
            for voxel in iter(VMFullBaseIterator(self.voxel_model)):
                if len(voxel) == 0:
                    VoxelDB.delete_voxel(voxel.id, db_connection)
                    ScanDB.delete_scan(voxel.scan_id, db_connection)
                else:
                    update_scan_in_db_from_scan(voxel.scan, db_connection)
                    update_voxel_in_db_from_voxel(voxel, db_connection)
                    voxel_counter += 1
            self.voxel_model.len = voxel_counter
            update_voxel_model_in_db_from_voxel_model(self.voxel_model, db_connection)
