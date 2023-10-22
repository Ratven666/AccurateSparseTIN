from sqlalchemy import Table, Column, Integer, Float, ForeignKey


def create_dem_cell_db_table(metadata):
    dem_cell_db_table = Table("dem_cells", metadata,
                              Column("voxel_id", Integer,
                                     ForeignKey("voxels.id", ondelete="CASCADE"),
                                     primary_key=True),
                              Column("base_model_id", Integer,
                                     ForeignKey("dem_models.id", ondelete="CASCADE"),
                                     primary_key=True),
                              Column("Avr_Z", Float),
                              Column("r", Integer),
                              Column("MSE", Float, default=None)
                              )
    return dem_cell_db_table
