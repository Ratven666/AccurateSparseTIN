from sqlalchemy import Table, Column, Integer, Float, ForeignKey


def create_plane_cell_db_table(metadata):
    plane_cell_db_table = Table("plane_cells", metadata,
                                Column("voxel_id", Integer,
                                       ForeignKey("voxels.id", ondelete="CASCADE"),
                                       primary_key=True),
                                Column("base_model_id", Integer,
                                       ForeignKey("dem_models.id", ondelete="CASCADE"),
                                       primary_key=True),
                                Column("A", Float),
                                Column("B", Float),
                                Column("D", Float),
                                Column("mA", Float),
                                Column("mB", Float),
                                Column("mD", Float),
                                Column("r", Integer),
                                Column("MSE", Float, default=None)
                                )
    return plane_cell_db_table
