import json
from os import path

from classes.Point import Point
from classes.ScanLite import ScanLite
from classes.abc_classes.SerializerABC import SerializerABC


class ScanJsonSerializer(SerializerABC):

    def __init__(self, data):
        super().__init__(data)
        self.data_dict = None

    def init_data_dict(self, dump_with_points):
        data_dict = {"scan": {"scan_data": {},
                              "points": []}
                     }
        data_dict["scan"]["scan_data"] = self.data.__dict__

        if dump_with_points:
            for point in self.data:
                data_dict["scan"]["points"].append(list(point.get_dict().values()))
        self.data_dict = data_dict
        return data_dict

    def dump(self, file_path=".", dump_with_points=True):
        self.init_data_dict(dump_with_points)
        file_name = path.join(file_path, f"Scan_{self.data.scan_name}.json")
        with open(file_name, "w") as write_file:
            json.dump(self.data_dict, write_file)

    @staticmethod
    def load(file_path):
        with open(file_path, "r") as read_file:
            data = json.load(read_file)
        scan = ScanLite.create_from_scan_dict(data["scan"]["scan_data"])
        scan._points = [Point.parse_point_from_db_row(point_row) for point_row in data["scan"]["points"]]
        return scan
