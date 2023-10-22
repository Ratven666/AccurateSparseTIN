from classes.Point import Point
from classes.ScanLite import ScanLite
from classes.VoxelModelLite import VoxelModelLite
from utils.scan_utils.Scan_metrics import update_scan_borders
from utils.scan_utils.scan_samplers.ScanSamplerABC import ScanSamplerABC


class VoxelDownsamplingScanSampler(ScanSamplerABC):

    def __init__(self, grid_step, is_2d_sampling=False, average_the_data=False):
        self.average_the_data = average_the_data
        self.is_2d_sampling = is_2d_sampling
        self.grid_step = grid_step
        self._voxel_model = None

    def do_sampling(self, scan):
        self.create_vm(scan)
        sample_func = self.average_sampling if self.average_the_data else self.central_point_sampling
        for point in scan:
            voxel = self.get_voxel_by_point(point)
            sample_func(voxel, point)
        sampled_scan = ScanLite(self.scan_name_generator(scan))
        for voxel in self._voxel_model:
            voxel_point = voxel.container_dict["voxel_point"]
            voxel_point.X = round(voxel_point.X, 3)
            voxel_point.Y = round(voxel_point.Y, 3)
            voxel_point.Z = round(voxel_point.Z, 3)
            voxel_point.R = round(voxel_point.R)
            voxel_point.G = round(voxel_point.G)
            voxel_point.B = round(voxel_point.B)
            sampled_scan.add_point(voxel_point)
            update_scan_borders(sampled_scan, voxel_point)
        return sampled_scan

    def create_vm(self, scan):
        vm = VoxelModelLite(scan=scan, step=self.grid_step, is_2d_vxl_mdl=self.is_2d_sampling)
        self._voxel_model = vm

    def get_voxel_by_point(self, point):
        vxl_md_X = int((point.X - self._voxel_model.min_X) // self._voxel_model.step)
        vxl_md_Y = int((point.Y - self._voxel_model.min_Y) // self._voxel_model.step)
        if self._voxel_model.is_2d_vxl_mdl:
            vxl_md_Z = 0
        else:
            vxl_md_Z = int((point.Z - self._voxel_model.min_Z) // self._voxel_model.step)
        return self._voxel_model.voxel_structure[vxl_md_Z][vxl_md_Y][vxl_md_X]

    def scan_name_generator(self, scan):
        sampler_type = "Voxel_Average" if self.average_the_data else "Voxel_Center"
        sample_dimension = "2D" if self.is_2d_sampling else "3D"
        scan_name = f"Sampled_{scan.scan_name}_by_{sampler_type}_{sample_dimension}_step_{self.grid_step}m"
        return scan_name

    @staticmethod
    def average_sampling(voxel, point):
        if len(voxel.container_dict) == 0:
            voxel.container_dict["voxel_point"] = point
            voxel.container_dict["point_count"] = 1
        else:
            v_point = voxel.container_dict["voxel_point"]
            p_count = voxel.container_dict["point_count"]
            v_point.X = (v_point.X * p_count + point.X) / (p_count + 1)
            v_point.Y = (v_point.Y * p_count + point.Y) / (p_count + 1)
            v_point.Z = (v_point.Z * p_count + point.Z) / (p_count + 1)
            v_point.R = (v_point.R * p_count + point.R) / (p_count + 1)
            v_point.G = (v_point.G * p_count + point.G) / (p_count + 1)
            v_point.B = (v_point.B * p_count + point.B) / (p_count + 1)
            voxel.container_dict["voxel_point"] = v_point
            voxel.container_dict["point_count"] += 1

    def central_point_sampling(self, voxel, point):
        def calc_dist_to_voxel_center(vm, voxel, point):
            if vm.is_2d_vxl_mdl is True:
                dist = ((point.X - voxel.container_dict["c_point"].X) ** 2 +
                        (point.Y - voxel.container_dict["c_point"].Y) ** 2) ** 0.5
            else:
                dist = ((point.X - voxel.container_dict["c_point"].X) ** 2 +
                        (point.Y - voxel.container_dict["c_point"].Y) ** 2 +
                        (point.Z - voxel.container_dict["c_point"].Z) ** 2) ** 0.5
            return dist

        if len(voxel.container_dict) == 0:
            voxel.container_dict["voxel_point"] = point
            voxel.container_dict["c_point"] = Point(X=voxel.X + voxel.step / 2,
                                                    Y=voxel.Y + voxel.step / 2,
                                                    Z=voxel.Z + voxel.step / 2,
                                                    R=voxel.R, G=voxel.G, B=voxel.B)
            voxel.container_dict["min_dist"] = calc_dist_to_voxel_center(self._voxel_model, voxel, point)
        else:
            new_dist = calc_dist_to_voxel_center(self._voxel_model, voxel, point)
            if new_dist < voxel.container_dict["min_dist"]:
                voxel.container_dict["voxel_point"] = point
                voxel.container_dict["min_dist"] = new_dist


class VoxelDownsamplingScanSamplerWithMixin(VoxelDownsamplingScanSampler):

    def __init__(self, grid_step, name_mixin="", dx=0, dy=0, dz=0, is_2d_sampling=False, average_the_data=False):
        self.name_mixin = name_mixin
        self.dx, self.dy, self.dz = dx, dy, dz
        super().__init__(grid_step, is_2d_sampling, average_the_data)

    def create_vm(self, scan):
        vm = VoxelModelLite(scan=scan, step=self.grid_step, dx=self.dx, dy=self.dy, dz=self.dz,
                            is_2d_vxl_mdl=self.is_2d_sampling)
        self._voxel_model = vm

    def scan_name_generator(self, scan):
        sampler_type = "Voxel_Average" if self.average_the_data else "Voxel_Center"
        sample_dimension = "2D" if self.is_2d_sampling else "3D"
        scan_name = (f"{self.name_mixin}_Sampled_{scan.scan_name}_by_"
                     f"{sampler_type}_{sample_dimension}_step_{self.grid_step}m"
                     f"_dx_{round(self.dx * self.grid_step, 3)}"
                     f"_dy_{round(self.dy * self.grid_step, 3)}"
                     f"_dz_{round(self.dz * self.grid_step, 3)}")
        return scan_name
