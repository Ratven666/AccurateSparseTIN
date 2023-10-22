import numpy as np
from scipy. stats import chi2

import plotly.graph_objects as go


class HistMSEPlotterPlotly:

    def __init__(self, bin_size=1, plot_like_probability=False):
        self.models = None
        self.bin_size = bin_size
        self.plot_like_probability = plot_like_probability

    def plot(self, models):
        self.models = models
        fig_hist = go.Figure()
        fig_box = go.Figure()
        for model in models:
            mse_data = [cell.mse for cell in model if cell.mse is not None]

            max_mse = max(mse_data)
            mean = sum(mse_data) / len(mse_data)
            x = np.linspace(0, max_mse, 100)
            y = chi2.pdf(x, df=mean, scale=1)
            n = round(1 + 3.2 * np.log10(len(mse_data))) * 4
            n_1 = round(max_mse / (model.mse_data / 2)) * 4
            print(n/4, n_1)
            if self.plot_like_probability:
                fig_hist.add_trace(go.Histogram(x=mse_data, histnorm='probability density', nbinsx=n_1, name=model.model_name))
                fig_hist.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"X^2_{model.model_name}"))
            else:
                fig_hist.add_trace(go.Histogram(x=mse_data, name=model.model_name))
                fig_hist.add_trace(go.Scatter(x=x, y=y*len(mse_data), mode='lines', name=f"X^2_{model.model_name}"))

            fig_box.add_trace(go.Box(y=mse_data, name=model.model_name))

        fig_hist.update_layout(barmode='overlay')
        fig_hist.update_traces(opacity=0.75)
        fig_hist.update_layout(
            title_text='Sampled Results',  # title of plot
            xaxis_title_text='MSE in cells',  # xaxis label
            yaxis_title_text='Count',  # yaxis label
        )
        fig_hist.show()
        fig_box.show()
