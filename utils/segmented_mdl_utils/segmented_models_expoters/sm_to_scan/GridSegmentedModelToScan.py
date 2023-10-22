from classes.Point import Point
from classes.ScanLite import ScanLite
from utils.segmented_mdl_utils.segmented_models_expoters.sm_to_scan.SegmentedModelToScanABC import \
    SegmentedModelToScanABC


class GridSegmentedModelToScan(SegmentedModelToScanABC):

    def __init__(self, segmented_model):
        super().__init__(segmented_model)

    def _get_next_point_on_grid(self, grid_densification):
        x_count, y_count = [round(count * grid_densification + 1) for count
                            in [self.segmented_model.voxel_model.X_count,
                                self.segmented_model.voxel_model.Y_count]]
        z_count = self.segmented_model.voxel_model.Z_count
        step = self.segmented_model.voxel_model.step
        min_x, min_y, min_z = self.segmented_model.voxel_model.min_X, \
                              self.segmented_model.voxel_model.min_Y, \
                              self.segmented_model.voxel_model.min_Z
        for k in range(z_count):
            for i in range(x_count):
                for j in range(y_count):
                    point = Point(X=min_x + i * step / grid_densification,
                                  Y=min_y + j * step / grid_densification,
                                  Z=min_z + k * step,
                                  R=0, G=0, B=0)
                    z = self.segmented_model.get_z_from_point(point)
                    if z is not None:
                        cell = self.segmented_model.get_model_element_for_point(point)
                        point.Z = round(z, 4)
                        point.R = cell.voxel.R
                        point.G = cell.voxel.G
                        point.B = cell.voxel.B
                        yield point

    def export_to_scan(self, grid_densification=1):
        scan = ScanLite(f"Scan_{self.segmented_model.model_name}_step="
                             f"{self.segmented_model.voxel_model.step / grid_densification}")
        for point in self._get_next_point_on_grid(grid_densification):
            scan.add_point(point)
        return scan
