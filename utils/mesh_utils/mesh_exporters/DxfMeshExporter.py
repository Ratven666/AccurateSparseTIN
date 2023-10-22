from os import path

import ezdxf

from utils.mesh_utils.mesh_exporters.MeshExporterABC import MeshExporterABC


class DxfMeshExporter(MeshExporterABC):

    def __init__(self, mesh):
        super().__init__(mesh)

    def __save_dxf(self, file_path):
        doc = ezdxf.new("R2000")
        msp = doc.modelspace()
        mesh = msp.add_mesh()
        mesh.dxf.subdivision_levels = 0
        with mesh.edit_data() as mesh_data:
            mesh_data.vertices = self.vertices
            mesh_data.faces = self.faces
        doc.saveas(file_path)

    def export(self, file_path="."):
        file_path = path.join(file_path, f"{self.mesh.mesh_name.replace(':', '=')}.dxf")
        self.__save_dxf(file_path)
