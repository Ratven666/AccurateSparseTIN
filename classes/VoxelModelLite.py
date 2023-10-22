

from sqlalchemy import select, desc
from sqlalchemy.exc import IntegrityError

from classes.VoxelLite import VoxelLite
from classes.VoxelModelDB import VoxelModelDB
from classes.abc_classes.VoxelModelABC import VoxelModelABC
from utils.scan_utils.Scan_metrics import update_scan_borders
from utils.start_db import engine, Tables
from utils.voxel_utils.voxel_model_separators.VMSeparatorABC import VMSeparatorABC


class VMLiteSeparator(VMSeparatorABC):

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
        """
        Создается полная воксельная структура
        :param voxel_model: воксельная модель
        :return: None
        """
        self.voxel_model = voxel_model
        id_generator = (id_ for id_ in range(1, len(self.voxel_model)+1))
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
        """
        Пересчитывает метрики сканов и вокселей по базовому скану scan
        :param scan: скан по которому разбивается воксельная модель
        :return: None
        """
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
    def __update_voxel_data(voxel, point):
        """
        Обновляет значения метрик вокселя (цвет и количество точек)
        :param voxel: обновляемый воксель
        :param point: точка, попавшая в воксель
        :return: None
        """
        voxel.R = (voxel.R * voxel.len + point.R) / (voxel.len + 1)
        voxel.G = (voxel.G * voxel.len + point.G) / (voxel.len + 1)
        voxel.B = (voxel.B * voxel.len + point.B) / (voxel.len + 1)
        voxel.len += 1


class VMLiteFullBaseIterator:
    """
    Иттератор полной воксельной модели
    """
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

    def save_to_db(self):
        """Сохраняет объект ScanLite в базе данных вместе с точками"""
        with engine.connect() as db_connection:
            stmt = (select(Tables.voxels_db_table.c.id).order_by(desc("id")))
            last_voxel_id = db_connection.execute(stmt).first()
            if last_voxel_id:
                last_voxel_id = last_voxel_id[0]
            else:
                last_voxel_id = 0
            if self.id is None:
                stmt = (select(Tables.voxel_models_db_table.c.id).order_by(desc("id")))
                last_vm_id = db_connection.execute(stmt).first()
                if last_vm_id:
                    self.id = last_vm_id[0] + 1
                else:
                    self.id = 1
            voxels = []
            for voxel in self:
                last_voxel_id += 1
                voxel.id = last_voxel_id
                voxels.append({"id": voxel.id, "vxl_name": voxel.vxl_name,
                               "X": voxel.X, "Y": voxel.Y, "Z": voxel.Z,
                               "step": voxel.step, "len": voxel.len,
                               "R": voxel.R, "G": voxel.G, "B": voxel.B,
                               "scan_id": voxel.scan_id, "vxl_mdl_id": self.id
                               })
            try:
                db_connection.execute(Tables.voxels_db_table.insert(), voxels)
                db_connection.execute(Tables.voxel_models_db_table.insert(),
                                      [{"id": self.id, "vm_name": self.vm_name,
                                        "step": self.step,
                                        "dx": self.dx, "dy": self.dy, "dz": self.dz,
                                        "len": self.len,
                                        "X_count": self.X_count, "Y_count": self.Y_count, "Z_count": self.Z_count,
                                        "min_X": self.min_X, "max_X": self.max_X,
                                        "min_Y": self.min_Y, "max_Y": self.max_Y,
                                        "min_Z": self.min_Z, "max_Z": self.max_Z,
                                        "base_scan_id": self.base_scan_id}])
                db_connection.commit()
                return self
            except IntegrityError:
                self.logger.warning(f"Такие объекты уже присутствуют в Базе Данных!!")
                return VoxelModelDB(scan=self.base_scan, step=self.step,
                                    dx=self.dx, dy=self.dy, dz=self.dz,
                                    is_2d_vxl_mdl=self.is_2d_vxl_mdl)
