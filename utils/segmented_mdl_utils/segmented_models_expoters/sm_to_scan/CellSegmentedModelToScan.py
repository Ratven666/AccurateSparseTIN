from classes.Point import Point
from classes.ScanLite import ScanLite
from utils.segmented_mdl_utils.segmented_models_expoters.sm_to_scan.SegmentedModelToScanABC import \
    SegmentedModelToScanABC


class CellSegmentedModelToScan(SegmentedModelToScanABC):

    def __init__(self, segmented_model):
        super().__init__(segmented_model)

    def _get_next_point_on_grid(self):
        DELTA = 1e-6
        for cell in self.segmented_model:
            if cell is None:
                continue
            for x, y in [(cell.voxel.X + DELTA, cell.voxel.Y + DELTA),
                         (cell.voxel.X + DELTA, cell.voxel.Y + cell.voxel.step - DELTA),
                         (cell.voxel.X + cell.voxel.step - DELTA, cell.voxel.Y + cell.voxel.step - DELTA),
                         (cell.voxel.X + cell.voxel.step - DELTA, cell.voxel.Y + DELTA)]:
                z = cell.get_z_from_xy(x, y)
                if z is None:
                    continue
                r, g, b = cell.voxel.R, cell.voxel.G, cell.voxel.B
                yield Point(X=x, Y=y, Z=round(z, 4), R=r, G=g, B=b)

    def export_to_scan(self):
        scan = ScanLite(f"Scan_{self.segmented_model.model_name}_step={self.segmented_model.voxel_model.step}")
        for point in self._get_next_point_on_grid():
            scan.add_point(point)
        return scan
