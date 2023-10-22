import numpy as np

from utils.segmented_mdl_utils.segmented_models_filters.SMFilterABC import SMFilterABC


class SMFilterPercentile(SMFilterABC):

    def __init__(self, segmented_model, k_value=1.5):
        super().__init__(segmented_model)
        self.k_value = k_value
        self.mse_min = None
        self.mse_max = None
        self.__calk_critical_value()

    # def __calk_critical_value(self):
    #     mses = []
    #     for cell in self.model:
    #         if cell.mse is not None:
    #             mses.append(cell.mse)
    #     Q1 = np.percentile(mses, 25, method='midpoint')
    #     Q3 = np.percentile(mses, 75, method='midpoint')
    #     IQR = Q3 - Q1
    #     self.mse_min = Q1 - self.k_value * IQR
    #     self.mse_max = Q3 + self.k_value * IQR

    def __calk_critical_value(self):
        mses = []
        for cell in self.model:
            if cell.mse is not None:
                mses.append(cell.mse)
        mses.sort()
        Q1 = mses[int(len(mses) * 0.25)-1]
        Q3 = mses[int(len(mses) * 0.75)-1]
        IQR = Q3 - Q1
        self.mse_min = Q1 - self.k_value * IQR
        self.mse_max = Q3 + self.k_value * IQR

    def _filter_logic(self, cell):
        if cell.mse is None:
            return False
        # if self.mse_min <= cell.mse <= self.mse_max:
        if cell.mse <= self.mse_max:
            return True
        else:
            return False

