from classes.abc_classes.SegmentedModelABC import SegmentedModelABC
from utils.segmented_mdl_utils.segmented_models_expoters.sm_to_scan.CellSegmentedModelToScan import \
    CellSegmentedModelToScan
from utils.segmented_mdl_utils.segmented_models_expoters.sm_to_scan.GridSegmentedModelToScan import \
    GridSegmentedModelToScan


class SegmentedModelToScan:

    def __init__(self, segmented_model: SegmentedModelABC, custom_exporter=None):
        self.segmented_model = segmented_model
        if custom_exporter is None:
            self.exporter = self.__chose_sm_exporter()
        else:
            self.exporter = custom_exporter(self.segmented_model)

    def __chose_sm_exporter(self):
        if self.segmented_model.model_type.name in ["BI_DEM_WITH_MSE", "BI_DEM_WITHOUT_MSE",
                                                    "BI_PLANE_WITH_MSE", "BI_PLANE_WITHOUT_MSE"]:
            return GridSegmentedModelToScan(self.segmented_model)
        elif self.segmented_model.model_type.name == "DEM":
            return CellSegmentedModelToScan(self.segmented_model)
        elif self.segmented_model.model_type.name == "PLANE":
            return CellSegmentedModelToScan(self.segmented_model)
        elif self.segmented_model.model_type.name == "MESH":
            return GridSegmentedModelToScan(self.segmented_model)

    def export_to_scan(self, *args, **kwargs):
        return self.exporter.export_to_scan(*args, **kwargs)
