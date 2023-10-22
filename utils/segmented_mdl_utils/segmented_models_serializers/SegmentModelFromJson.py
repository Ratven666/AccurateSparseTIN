from classes.BiCellDB import BiCellDB
from classes.DemCellDB import DemCellDB
from classes.DemTypeEnum import DemTypeEnum
from classes.PlaneCellDB import PlaneCellDB
from classes.VoxelLite import VoxelLite
from classes.abc_classes.SegmentedModelABC import SegmentedModelABC
from utils.voxel_utils.voxel_model_serializers.VoxelModelJsonSerializer import VoxelModelJsonSerializer


class SegmentModelFromJson(SegmentedModelABC):
    __base_cell_class = {"DEM": DemCellDB,
                         "PLANE": PlaneCellDB,
                         "BI_DEM_WITH_MSE": BiCellDB,
                         "BI_DEM_WITHOUT_MSE": BiCellDB,
                         "BI_PLANE_WITH_MSE": BiCellDB,
                         "BI_PLANE_WITHOUT_MSE": BiCellDB
                         }

    def __init__(self, sm_dict):
        self.id = sm_dict["segmented_model"]["segmented_model_data"]["id"]
        self.model_type = sm_dict["segmented_model"]["segmented_model_data"]["model_type"]
        self.model_name = sm_dict["segmented_model"]["segmented_model_data"]["model_name"]
        self.mse_data = sm_dict["segmented_model"]["segmented_model_data"]["mse_data"]
        self.base_voxel_model_id = sm_dict["segmented_model"]["segmented_model_data"]["base_voxel_model_id"]
        self.voxel_model = VoxelModelJsonSerializer \
            .load(sm_dict["segmented_model"]["segmented_model_data"]["voxel_model"])
        self.cell_type = self.__base_cell_class[self.model_type]
        self._model_structure = self._create_model_structure(sm_dict)

    def _create_model_structure(self, sm_dict):
        ms_dict = {}
        for key, value in sm_dict["segmented_model"]["model_structure"].items():
            voxel = VoxelLite.parse_voxel_from_data_row(value["voxel"])
            cell = self.cell_type(voxel, self)
            cell._copy_cell_data(value)
            ms_dict[key] = cell
        return ms_dict

    def _calk_segment_model(self):
        """
        Метод определяющий логику создания конкретной модели
        :return: None
        """
        pass
