from sqlalchemy import Table, Column, Integer, ForeignKey, Float


def create_triangles_db_table(metadata):
    triangles_db_table = Table("triangles", metadata,
                               Column("id", Integer, primary_key=True),
                               Column("point_0_id", Integer, ForeignKey("points.id")),
                               Column("point_1_id", Integer, ForeignKey("points.id")),
                               Column("point_2_id", Integer, ForeignKey("points.id")),
                               Column("r", Integer, default=None),
                               Column("mse", Float, default=None),
                               Column("mesh_id", Integer, ForeignKey("meshes.id"))
                               )
    return triangles_db_table
