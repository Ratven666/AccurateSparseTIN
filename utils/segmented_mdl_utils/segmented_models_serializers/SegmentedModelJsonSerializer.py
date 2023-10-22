import json
from copy import copy
from os import path

from classes.abc_classes.SerializerABC import SerializerABC
from utils.segmented_mdl_utils.segmented_models_serializers.SegmentModelFromJson import SegmentModelFromJson
from utils.voxel_utils.voxel_model_serializers.VoxelModelJsonSerializer import VoxelModelJsonSerializer


class SegmentedModelJsonSerializer(SerializerABC):

    def __init__(self, data):
        super().__init__(data)
        self.data_dict = None

    def init_data_dict(self):
        data_dict = {"segmented_model": {"segmented_model_data": {},
                                         "model_structure": {}}
                     }
        data_dict["segmented_model"]["segmented_model_data"] = copy(self.data.__dict__)
        data_dict["segmented_model"]["segmented_model_data"].pop("_model_structure")
        data_dict["segmented_model"]["segmented_model_data"].pop("cell_type")
        data_dict["segmented_model"]["segmented_model_data"].pop("voxel_model")
        data_dict["segmented_model"]["segmented_model_data"]["model_type"] = \
            str(data_dict["segmented_model"]["segmented_model_data"]["model_type"].name)

        data_dict["segmented_model"]["model_structure"] = self.data._model_structure
        for key, cell in data_dict["segmented_model"]["model_structure"].items():
            cell_dict = cell.get_db_raw_data()
            voxel_dict = cell.voxel.get_dict()
            voxel_list = [voxel_dict["id"], voxel_dict["X"], voxel_dict["Y"], voxel_dict["Z"],
                          voxel_dict["step"], voxel_dict["vxl_mdl_id"], voxel_dict["vxl_name"], voxel_dict["scan_id"],
                          voxel_dict["len"], voxel_dict["R"], voxel_dict["G"], voxel_dict["B"]]
            cell_dict["voxel"] = voxel_list
            data_dict["segmented_model"]["model_structure"][key] = cell_dict

        self.data_dict = data_dict
        return data_dict

    def dump(self, file_path=".", dump_with_full_scan=False):
        self.init_data_dict()
        vm_path = VoxelModelJsonSerializer(self.data.voxel_model).dump(file_path=file_path,
                                                                       dump_with_full_scan=dump_with_full_scan)
        self.data_dict["segmented_model"]["segmented_model_data"]["voxel_model"] = vm_path
        file_path = path.join(file_path, f"{self.data.model_name.replace(':', '=')}.json")
        with open(file_path, "w") as write_file:
            json.dump(self.data_dict, write_file)

    @staticmethod
    def load(file_path):
        with open(file_path, "r") as read_file:
            data = json.load(read_file)
        sm = SegmentModelFromJson(data)
        return sm
