import logging
from copy import copy

from sqlalchemy import select

from CONFIG import LOGGER, VOXEL_IN_VM
from classes.MeshDB import MeshDB
from classes.MeshLite import MeshLite
from classes.MeshSegmentModelDB import MeshSegmentModelDB
from classes.Point import Point
from classes.VoxelModelLite import VoxelModelLite
from utils.mesh_utils.mesh_plotters.MeshPlotterPlotly import MeshPlotterPlotly
from utils.scan_utils.scan_samplers.VoxelDownsamplingScanSampler import VoxelDownsamplingScanSampler
from utils.start_db import Tables, engine


class MeshMSEConstDB:
    logger = logging.getLogger(LOGGER)
    db_table = Tables.meshes_db_table

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
        self.__init_mesh()

    def __iter__(self):
        return iter(self.mesh)

    def __len__(self):
        return len(self.mesh)

    def plot(self, plotter=MeshPlotterPlotly()):
        """
        Вывод отображения скана
        :@aram plotter: объект определяющий логику отображения поверхности
        :@return: None
        """
        plotter.max_mse = self.max_triangle_mse
        plotter.plot(self.mesh)

    def __name_generator(self):
        """
        @return: Возвращает имя поверхности с учетов входных для нее параметров
        """
        d_type = "2D" if self.is_2d else "3D"
        return (f"MESH_{self.scan.scan_name}"
                f"_max_brd_{self.max_border_length}"
                f"_max_mse_{self.max_triangle_mse}_n_{self.n}_{d_type}")

    def __init_mesh(self):
        """
        Выполняет проверку присутствия поверхности с такими параметрами для рассчета в БД.
        В случае такого присутствия - загружает данные из БД
        @return: None
        """
        select_ = select(self.db_table).where(self.db_table.c.mesh_name == self.mesh_name)
        with engine.connect() as db_connection:
            db_mesh_data = db_connection.execute(select_).mappings().first()
            if db_mesh_data is not None:
                mesh_id = db_mesh_data["id"]
                self.mesh = MeshDB.get_mesh_by_id(mesh_id)
            else:
                self.__calculate_mesh()
                self.__init_mesh()

    def __calculate_mesh(self):
        """
        Основная логика расчета
        @return: None
        """
        self.__do_prepare_calc()
        self.__do_basic_logic()
        if self.calk_with_brute_force:
            self.__do_brute_force_calk()
        self.__end_logic()
        self.vm = None

    def __do_prepare_calc(self):
        """
        Выполняет предвариельные рассчеты
        @return: None
        """
        self.sampled_scan = VoxelDownsamplingScanSampler(grid_step=self.max_border_length,
                                                         is_2d_sampling=self.is_2d,
                                                         average_the_data=False).do_sampling(self.scan)
        self.voxel_size = self.__get_voxel_size_for_vm()
        self.vm = VoxelModelLite(self.scan, self.voxel_size, is_2d_vxl_mdl=True)
        self.mesh = MeshLite(self.sampled_scan)
        self.mesh.calk_mesh_mse(self.scan, voxel_size=self.voxel_size, delete_temp_models=True)
        self.temp_mesh = copy(self.mesh)

    def __do_basic_logic(self):
        """
        Основная логика расчета
        @return: None
        """
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

            # self.temp_mesh.plot()
            print(self.loop_counter)

    def __do_brute_force_calk(self):
        """
        Выполнение вычислений по полной поверхности
        @return: None
        """
        while self.loop_counter < self.n:
            self.temp_mesh = MeshLite(self.sampled_scan)
            self.temp_mesh.calk_mesh_mse(self.scan, voxel_size=self.voxel_size, clear_previous_mse=True,
                                         delete_temp_models=True)

            # self.temp_mesh.plot()

            bad_triangles = self.__find_and_prepare_bad_triangles(self.temp_mesh)
            if len(bad_triangles) == 0:
                # print(f"AAAAAA{self.loop_counter}AAAAAA" * 50)
                break
            self.temp_mesh.triangles = bad_triangles
            bad_triangles = self.__calc_mesh_triangles_centroids(self.temp_mesh)
            for triangle in bad_triangles:
                self.sampled_scan.add_point(triangle.container_dict["c_point"])
            self.loop_counter += 1
            print(self.loop_counter)

    def __end_logic(self):
        """
        Заключительные операции по сохранению резульатов в БД
        @return: None
        """
        points = set()
        for point in self.sampled_scan:
            points.add(point)
        self.sampled_scan.scan_name = self.mesh_name[5:]
        self.sampled_scan._points = list(points)
        self.sampled_scan.save_to_db()
        self.mesh = MeshDB(self.sampled_scan)
        self.mesh.calk_mesh_mse(self.scan, delete_temp_models=True)

    @staticmethod
    def __get_new_triangles(prior_mesh, new_mesh):
        """
        Сравнивает две поверхности,
        возвращает список треугольникв присутствующих в new_mesh и отсутствующих в prior_mesh
        @param prior_mesh: Базовая поверхность
        @param new_mesh: ПОверхность из которой выделяются новые полигоны
        @return: Спасиок полигонов поверхности new_mesh не встретившихся в поверхности prior_mesh
        """
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
        """
        Определяет оптимальный размер вокселя для служебных моделей
        @return: Размер вокселя
        """
        voxel_size = VoxelModelLite.get_step_by_voxel_count(self.scan, VOXEL_IN_VM,
                                                            is_2d_vxl_mdl=True,
                                                            round_n=2)
        return voxel_size

    @staticmethod
    def __calc_centroid_point_in_triangle(triangle):
        """
        Рассчитывает ценройд треугольника
        @param triangle: Треугольник для которого рассчитывается центройд
        @return: Точка центройда
        """
        s_x, s_y, s_z = 0, 0, 0
        for point in triangle:
            s_x += point.X
            s_y += point.Y
            s_z += point.Z
        return Point(X=s_x / 3, Y=s_y / 3, Z=s_z / 3,
                     R=0, G=0, B=0)

    def __find_and_prepare_bad_triangles(self, mesh):
        """
        Выбирает из поверхности треугольники с СКП больше max_triangle_mse
        Добавляет в них атрибуты необходимые для поиска центральной точки
        @param mesh: ПОверхность из которой формируется выборка
        @return: Список полигонов СКП которых превышает установленное значение
        """
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
        """
        Ищет точку скана наиболее близкую к центройду треугольника
        @param mesh: Поверхность для октоорой ищутся точки
        @return: генератор полигонов поверхности
        """
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
