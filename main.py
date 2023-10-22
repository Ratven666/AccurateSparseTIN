from os import remove
from os.path import basename, join

from CONFIG import DATABASE_NAME
from classes.branch_classes.CsvMeshDataExporter import CsvMeshDataExporter
from classes.branch_classes.MeshStatisticCalculator import MeshStatisticCalculator
from utils.logs.console_log_config import console_logger
from utils.mesh_utils.mesh_exporters.DxfMeshExporter import DxfMeshExporter
from utils.mesh_utils.mesh_exporters.PlyMeshExporter import PlyMeshExporter
from utils.mesh_utils.mesh_exporters.PlyMseMeshExporter import PlyMseMeshExporter
from utils.start_db import create_db, engine

from classes.branch_classes.MeshMSEConst import MeshMSEConstDB
from classes.ScanDB import ScanDB


def main():
    create_db()

    ####################################################################################
    FILE_PATH = "src/SKLD_Right_05_05.txt"

    MAX_BORDER_LENGTH_M = 10
    MSE = 0.15
    N = 25
    CALK_WITH_BRUTE_FORCE = False

    EXPORT_DXF = False
    EXPORT_PLY = False
    EXPORT_PLY_MSE = False
    EXPORT_SCAN = False

    SAVE_FULL_MESH_STATISTICS = False
    SAVE_BASE_STATISTICS = True

    SAVE_DISTRIBUTIONS_HIST_RMSE = True
    SAVE_DISTRIBUTIONS_HIST_R = True
    SAVE_DISTRIBUTIONS_HIST_AREA = True
    SAVE_DISTRIBUTIONS_HIST_PAIR_PLOT = True

    DELETE_DB = True
    ####################################################################################
    graf_dict = {"rmse": SAVE_DISTRIBUTIONS_HIST_RMSE,
                 "r": SAVE_DISTRIBUTIONS_HIST_R,
                 "area": SAVE_DISTRIBUTIONS_HIST_AREA,
                 "pair_plot": SAVE_DISTRIBUTIONS_HIST_PAIR_PLOT}

    scan_name = basename(FILE_PATH).split(".")[0]

    scan = ScanDB(scan_name=scan_name)
    scan.load_scan_from_file(file_name=FILE_PATH)

    mesh = MeshMSEConstDB(scan, max_border_length_m=MAX_BORDER_LENGTH_M,
                          max_triangle_mse_m=MSE,
                          n=N, calk_with_brute_force=CALK_WITH_BRUTE_FORCE)

    if EXPORT_DXF:
        DxfMeshExporter(mesh=mesh).export(file_path=".")
    if EXPORT_PLY:
        PlyMeshExporter(mesh=mesh).export(file_path=".")
    if EXPORT_PLY_MSE:
        PlyMseMeshExporter(mesh=mesh).export(file_path=".")
    if EXPORT_SCAN:
        mesh.mesh.scan.save_scan_in_file()
    if SAVE_BASE_STATISTICS:
        MeshStatisticCalculator(mesh.mesh).save_statistic()
    if SAVE_FULL_MESH_STATISTICS:
        CsvMeshDataExporter(mesh.mesh).export_mesh_data()

    MeshStatisticCalculator(mesh.mesh).save_distributions_histograms(graf_dict)

    mesh.plot()

    if DELETE_DB:
        engine.dispose()
        remove(join("", DATABASE_NAME))



if __name__ == "__main__":
    main()
