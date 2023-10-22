import numpy as np

from classes.Point import Point
from classes.ScanLite import ScanLite
from utils.scan_utils.scan_triangulators.__ScanTriangulator import ScanTriangulator
from utils.scan_utils.scan_triangulators.mesh_filters.__MeshFilterABC import MeshFilterABC


class MaxEdgeLengthFilter(MeshFilterABC):

    def __init__(self, max_edge_length=None):
        super().__init__()
        self.max_edge_length = max_edge_length

    def __find_bad_faces(self):
        bad_faces_idx = []
        for idx, face in enumerate(self.triangulator.faces):
            idx1, idx2 = -1, 0
            for _ in range(3):
                point1 = self.triangulator.vertices[face[idx1]]
                point2 = self.triangulator.vertices[face[idx2]]
                dist = ((point1[0] - point2[0])**2 +
                        (point1[1] - point2[1])**2 +
                        (point1[2] - point2[2])**2) ** 0.5
                if dist > self.max_edge_length:
                    bad_faces_idx.append(idx)
                    break
                idx1 += 1
                idx2 += 1
        return bad_faces_idx

    def __find_bad_vertices(self, bad_faces_idx):
        pass

    def __delete_bad_faces(self, bad_faces_idx):
        self.triangulator.faces = np.delete(self.triangulator.faces, bad_faces_idx, 0)

    def __delete_bad_vertices(self):
        pass

    def _filter_logic(self):
        if self.max_edge_length is None:
            return
        bad_faces_idx = self.__find_bad_faces()
        self.__delete_bad_faces(bad_faces_idx)


if __name__ == "__main__":
    p0 = Point(0, 0, 0, 0, 0, 0)
    p1 = Point(10, 0, 0, 0, 0, 0)
    p2 = Point(0, 11, 0, 0, 0, 0)
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

    filter_ = MaxEdgeLengthFilter(tri, max_edge_length=10).filtrate()
    print(tri)
