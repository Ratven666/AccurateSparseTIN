import logging
from abc import ABC, abstractmethod

from sqlalchemy import delete

from CONFIG import LOGGER
from utils.start_db import engine
from utils.voxel_utils.Voxel_model_metrics import update_voxel_model_in_db_from_voxel_model


class SMFilterABC(ABC):
    logger = logging.getLogger(LOGGER)

    def __init__(self, segmented_model):
        self.model = segmented_model

    def __model_name_generator(self):
        return f"{self.model.model_name}_FB_{self.__class__.__name__}"

    @abstractmethod
    def _filter_logic(self, cell):
        pass

    def filter_model(self):
        cells = []
        cell_table = None
        for cell in self.model:
            if cell_table is None:
                cell_table = cell.db_table
            if self._filter_logic(cell) is True:
                cells.append(cell.get_db_raw_data())
        with engine.connect() as db_connection:
            stmt = delete(cell_table)\
                    .where(cell_table.c.base_model_id == self.model.id)
            db_connection.execute(stmt)
            db_connection.execute(cell_table.insert(), cells)
            db_connection.commit()
        self.model.__init__(self.model.voxel_model)
        with engine.connect() as db_connection:
            self.model._calk_model_mse(db_connection=db_connection)
        self.logger.info(f"Фильтрация завершена")
        return self.model
