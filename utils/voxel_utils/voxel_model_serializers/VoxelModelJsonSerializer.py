import json
from copy import copy
from os import path

from classes.ScanLite import ScanLite
from classes.VoxelLite import VoxelLite
from classes.VoxelModelLite import VoxelModelLite
from classes.abc_classes.SerializerABC import SerializerABC
from utils.scan_utils.scan_serializers.ScanJsonSerializer import ScanJsonSerializer


class VoxelModelJsonSerializer(SerializerABC):

    def __init__(self, data):
        super().__init__(data)
        self.data_dict = None

    def init_data_dict(self):
        data_dict = {"voxel_model": {"voxel_model_data": {},
                                     "voxel_structure": []}
                     }
        data_dict["voxel_model"]["voxel_model_data"] = copy(self.data.__dict__)
        data_dict["voxel_model"]["voxel_model_data"].pop("voxel_model_separator")
        data_dict["voxel_model"]["voxel_model_data"].pop("base_scan")
        data_dict["voxel_model"]["voxel_model_data"].pop("voxel_structure")
        data_dict["voxel_model"]["voxel_model_data"]["base_scan"] = f"Scan_{self.data.base_scan.scan_name}.json"

        for voxel in self.data:
            voxel_dict = voxel.get_dict()
            voxel_list = [voxel_dict["id"], voxel_dict["X"], voxel_dict["Y"], voxel_dict["Z"],
                          voxel_dict["step"], voxel_dict["vxl_mdl_id"], voxel_dict["vxl_name"], voxel_dict["scan_id"],
                          voxel_dict["len"], voxel_dict["R"], voxel_dict["G"], voxel_dict["B"]]
            data_dict["voxel_model"]["voxel_structure"].append(voxel_list)

        self.data_dict = data_dict
        return data_dict

    def dump(self, file_path=".", dump_with_full_scan=False):
        self.init_data_dict()
        ScanJsonSerializer(self.data.base_scan).dump(file_path=file_path, dump_with_points=dump_with_full_scan)
        file_path = path.join(file_path, f"{self.data.vm_name.replace(':', '=')}.json")
        with open(file_path, "w") as write_file:
            json.dump(self.data_dict, write_file)
        return file_path

    @staticmethod
    def load(file_path):
        with open(file_path, "r") as read_file:
            data = json.load(read_file)
        try:
            scan = ScanJsonSerializer.load(file_path=path.join(path.dirname(file_path),
                                                               data["voxel_model"]["voxel_model_data"]["base_scan"]))
        except FileNotFoundError:
            scan = ScanLite("NotFoundScan")

        vm = VoxelModelLite(scan, step=data["voxel_model"]["voxel_model_data"]["step"],
                            dx=data["voxel_model"]["voxel_model_data"]["dx"],
                            dy=data["voxel_model"]["voxel_model_data"]["dy"],
                            dz=data["voxel_model"]["voxel_model_data"]["dz"],
                            is_2d_vxl_mdl=data["voxel_model"]["voxel_model_data"]["is_2d_vxl_mdl"])
        vm.id = data["voxel_model"]["voxel_model_data"]["id"]
        vm.vm_name = data["voxel_model"]["voxel_model_data"]["vm_name"]
        vm.len = data["voxel_model"]["voxel_model_data"]["len"]
        vm.X_count = data["voxel_model"]["voxel_model_data"]["X_count"]
        vm.Y_count = data["voxel_model"]["voxel_model_data"]["Y_count"]
        vm.Z_count = data["voxel_model"]["voxel_model_data"]["Z_count"]
        vm.min_X = data["voxel_model"]["voxel_model_data"]["min_X"]
        vm.max_X = data["voxel_model"]["voxel_model_data"]["max_X"]
        vm.min_Y = data["voxel_model"]["voxel_model_data"]["min_Y"]
        vm.max_Y = data["voxel_model"]["voxel_model_data"]["max_Y"]
        vm.min_Z = data["voxel_model"]["voxel_model_data"]["min_Z"]
        vm.max_Z = data["voxel_model"]["voxel_model_data"]["max_Z"]
        vm.base_scan_id = data["voxel_model"]["voxel_model_data"]["base_scan_id"]
        vm.base_scan = scan

        vm.voxel_structure = [VoxelLite.parse_voxel_from_data_row(voxel_row)
                              for voxel_row in data["voxel_model"]["voxel_structure"]]
        return vm
