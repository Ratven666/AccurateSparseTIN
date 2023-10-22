from classes.Line import Line
from utils.mesh_utils.mesh_filters.MeshFilterABC import MeshFilterABC


class MaxEdgeLengthMeshFilter(MeshFilterABC):

    def __init__(self, mesh, max_edge_length):
        super().__init__(mesh)
        self.max_edge_length = max_edge_length

    def _filter_logic(self, triangle):
        tr_edges = [Line(triangle.point_0, triangle.point_1),
                    Line(triangle.point_1, triangle.point_2),
                    Line(triangle.point_2, triangle.point_0)]
        for edge in tr_edges:
            if edge.get_distance() > self.max_edge_length:
                return False
        return True
