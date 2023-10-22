from sqlalchemy import Table, Column, Integer, Float, ForeignKey


def create_bi_cell_db_table(metadata):
    bi_cell_db_table = Table("bi_cells", metadata,
                             Column("voxel_id", Integer,
                                    ForeignKey("voxels.id", ondelete="CASCADE"),
                                    primary_key=True),
                             Column("base_model_id", Integer,
                                    ForeignKey("dem_models.id", ondelete="CASCADE"),
                                    primary_key=True),
                             Column("Z_ld", Float),
                             Column("Z_lu", Float),
                             Column("Z_rd", Float),
                             Column("Z_ru", Float),
                             Column("MSE_ld", Float),
                             Column("MSE_lu", Float),
                             Column("MSE_rd", Float),
                             Column("MSE_ru", Float),
                             Column("r", Integer),
                             Column("MSE", Float, default=None)
                             )
    return bi_cell_db_table
