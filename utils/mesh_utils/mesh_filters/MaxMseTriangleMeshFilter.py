from utils.mesh_utils.mesh_filters.MeshFilterABC import MeshFilterABC


class MaxMseTriangleMeshFilter(MeshFilterABC):

    def __init__(self, mesh, max_mse, filter_triangles_with_none_mse=True):
        super().__init__(mesh)
        self.max_mse = max_mse
        self.filter_triangles_with_none_mse = filter_triangles_with_none_mse

    def _filter_logic(self, triangle):
        if triangle.mse is None:
            if self.filter_triangles_with_none_mse:
                return False
            return True
        if triangle.mse > self.max_mse:
            return False
        return True
