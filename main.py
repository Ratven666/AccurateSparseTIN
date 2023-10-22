import time
from utils.logs.console_log_config import console_logger
from utils.start_db import create_db

from classes.branch_classes.MeshMSEConst import MeshMSEConstDB
from classes.ScanDB import ScanDB

def main():
    create_db()

    scan = ScanDB("SKLD_4")
    scan.load_scan_from_file(file_name="src/SKLD_Right_05_05.txt")

    mse = 0.15

    mesh = MeshMSEConstDB(scan, max_border_length_m=5, max_triangle_mse_m=mse, n=5, calk_with_brute_force=True)
    mesh.plot()


if __name__ == "__main__":
    main()
