import numpy as np

from utils.segmented_mdl_utils.segmented_models_filters.SMFilterABC import SMFilterABC


class SMFilterByMaxPercentile(SMFilterABC):

    def __init__(self, segmented_model, max_percentile=0.99):
        if max_percentile > 1:
            raise ValueError("Значение max_percentile должно быть в пределах от 0 до 1")
        super().__init__(segmented_model)
        self.max_percentile = max_percentile
        self.mse_max = None
        self.__calk_critical_value()

    def __calk_critical_value(self):
        mses = []
        for cell in self.model:
            if cell.mse is not None:
                mses.append(cell.mse)
        # self.mse_max = np.percentile(mses, self.max_percentile, method='midpoint')
        mses.sort()
        self.mse_max = mses[int(len(mses) * self.max_percentile)-1]

    def _filter_logic(self, cell):
        if cell.mse is None:
            return False
        if cell.mse <= self.mse_max:
            return True
        else:
            return False
