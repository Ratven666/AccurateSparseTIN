from utils.scan_utils.scan_filters.ScanFilterABC import ScanFilterABC


class ScanFilterByCylinderMSE(ScanFilterABC):

    def __init__(self, scan, cylinder, k_value=2.5):
        super().__init__(scan)
        self.cylinder = cylinder
        self.k_value = k_value

    def _filter_logic(self, point):
        if self.cylinder.mse is None:
            return True
        r = ((point.X - self.cylinder.X0) ** 2 + (point.Y - self.cylinder.Y0) ** 2) ** 0.5
        v = abs(r - self.cylinder.R)
        if v <= self.cylinder.mse * self.k_value:
            return True
        else:
            return False
