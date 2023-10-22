from utils.scan_utils.scan_filters.ScanFilterABC import ScanFilterABC


class ScanFilterByZminZmax(ScanFilterABC):

    def __init__(self, scan, z_min=None, z_max=None, k_value=2.5):
        super().__init__(scan, k_value)
        self.z_min = z_min
        self.z_max = z_max

    def _filter_logic(self, point):
        if self.z_min is not None and point.Z < self.z_min:
            return False
        if self.z_max is not None and point.Z > self.z_max:
            return False
        return True
