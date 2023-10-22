from utils.segmented_mdl_utils.segmented_models_filters.SMFilterABC import SMFilterABC


class SMFilterByMaxMSE(SMFilterABC):

    def __init__(self, segmented_model, max_mse):
        super().__init__(segmented_model)
        self.max_mse = max_mse

    def _filter_logic(self, cell):
        if cell.mse is None:
            return False
        if cell.mse <= self.max_mse:
            return True
        else:
            return False
