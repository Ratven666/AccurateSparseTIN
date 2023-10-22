import numpy as np
from scipy.spatial import Delaunay

from classes.Point import Point
from classes.ScanLite import ScanLite


class ScanTriangulator:

    def __init__(self, sampler=None, filters=[]):
        self.scan = None
        self.__sampler = sampler
        self.vertices = None
        self.vertices_colors = None
        self.faces = None
        self.face_colors = None
        self.filters = filters

    def __str__(self):
        return (f"{self.__class__.__name__} "
                f"[Name: {self.scan.scan_name}\t\t"
                f"Count_of_triangles: {len(self.faces)}]"
                )

    def __get_sample_data(self, scan):
        """
        Запускает процедуру разряжения скана если задан атрибут __sampler
        возвращает словарь с данными для визуализации разреженных данных
        :param scan: скан, который требуется разрядить и подготовить к визуализации
        :return: словарь с данными для визуализации
        """
        if self.__sampler is not None:
            scan = self.__sampler.do_sampling(scan)
        point_id, x_lst, y_lst, z_lst, c_lst = [], [], [], [], []
        for point in scan:
            point_id.append(point.id)
            x_lst.append(point.X)
            y_lst.append(point.Y)
            z_lst.append(point.Z)
            c_lst.append([point.R, point.G, point.B])
        return {"id": point_id, "x": x_lst, "y": y_lst, "z": z_lst, "color": c_lst, "scan": scan}

    def __calk_delone_triangulation(self, scan_data):
        """
        Рассчитываает треугольники между точками
        :param plot_data: Данные для которых рассчитывается триангуляция
        :return: славарь с указанием вершин треугольников
        """
        points2D = self.vertices[:, :2]
        tri = Delaunay(points2D)
        i_lst, j_lst, k_lst = ([triplet[c] for triplet in tri.simplices] for c in range(3))
        return {"i_lst": i_lst, "j_lst": j_lst, "k_lst": k_lst, "ijk": tri.simplices, "tri": tri}

    @staticmethod
    def __calk_faces_colors(ijk_dict, scan_data):
        """
        Рассчитывает цвета треугольников на основании усреднения цветов точек, образующих
        треугольник
        :param ijk_dict: словарь с вершинами треугольников
        :param plot_data: словарь с данными о точках отриовываемой модели
        :return: список цветов треугольников в формате библиотеки plotly
        """
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

    def triangulate(self, scan, filtrate=True):
        self.scan = scan
        scan_data = self.__get_sample_data(self.scan)
        self.vertices = np.vstack([scan_data["x"], scan_data["y"], scan_data["z"]]).T
        full_tri_data_dict = self.__calk_delone_triangulation(scan_data)
        self.faces = full_tri_data_dict["ijk"]
        self.face_colors = self.__calk_faces_colors(full_tri_data_dict, scan_data)
        if filtrate is True:
            for filter_ in self.filters:
                filter_.filtrate(self)


if __name__ == "__main__":
    p0 = Point(0, 0, 0, 0, 0, 0)
    p1 = Point(10, 0, 0, 0, 0, 0)
    p2 = Point(0, 10, 0, 0, 0, 0)
    p3 = Point(10, 10, 0, 0, 0, 0)
    p4 = Point(5, 5, 5, 0, 0, 0)
    scan = ScanLite("test")
    scan.add_point(p0)
    scan.add_point(p1)
    scan.add_point(p2)
    scan.add_point(p3)
    scan.add_point(p4)
    print(scan)
    tri = ScanTriangulator()
    tri.triangulate(scan)
    print(tri)
    print(tri.faces)
