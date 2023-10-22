from utils.scan_utils.scan_filters.ScanFilterABC import ScanFilterABC


class ScanFilterForTrees(ScanFilterABC):

    def __init__(self, scan, dem_model, k_value=2.5):
        super().__init__(scan)
        self.dem_model = dem_model
        self.k_value = k_value

    # def _filter_logic(self, point):
    #     cell = self.dem_model.get_model_element_for_point(point)
    #     if cell is None or cell.mse is None:
    #         return False
    #     try:
    #         cell_z = cell.get_z_from_xy(point.X, point.Y)
    #     except TypeError:
    #         return False
    #     v = point.Z - cell_z
    #     if v <= cell.mse * self.k_value:
    #         return True
    #     else:
    #         return False


    def _filter_logic(self, point):
        cell = self.dem_model.get_model_element_for_point(point)
        if cell is None or cell.mse is None:
            return False
        try:
            cell_z = cell.get_z_from_xy(point.X, point.Y)
        except TypeError:
            return False
        v = point.Z - cell_z
        if v <= self.dem_model.mse_data * self.k_value:
            return True
        else:
            return False
