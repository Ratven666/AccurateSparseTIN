import enum


class DemTypeEnum(enum.Enum):
    DEM = 1
    PLANE = 2
    BI_DEM_WITH_MSE = 3
    BI_DEM_WITHOUT_MSE = 4
    BI_PLANE_WITH_MSE = 5
    BI_PLANE_WITHOUT_MSE = 6
    MESH = 7
