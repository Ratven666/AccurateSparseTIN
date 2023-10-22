from classes.Point import Point
from classes.ScanLite import ScanLite
from utils.segmented_mdl_utils.segmented_models_expoters.sm_to_scan.SegmentedModelToScanABC import \
    SegmentedModelToScanABC


class CellCenterSegmentedModelToScan(SegmentedModelToScanABC):

    def __init__(self, segmented_model):
        super().__init__(segmented_model)

    def _get_next_point_on_grid(self):
        for cell in self.segmented_model:
            if cell is None:
                continue
            x = cell.voxel.X + (cell.voxel.step / 2)
            y = cell.voxel.Y + (cell.voxel.step / 2)
            z = cell.get_z_from_xy(x, y)
            if z is None:
                continue
            r, g, b = cell.voxel.R, cell.voxel.G, cell.voxel.B
            yield Point(X=x, Y=y, Z=round(z, 4), R=r, G=g, B=b)

    def export_to_scan(self):
        scan = ScanLite(f"Scan_from_cell_center_{self.segmented_model.model_name}_"
                        f"step={self.segmented_model.voxel_model.step}")
        for point in self._get_next_point_on_grid():
            scan.add_point(point)
        return scan
