# import seaborn as sns
# import matplotlib.pyplot as plt
#
# class HistMSEPlotterSeaborn:
#
#     def __init__(self, bin_size=.2, plot_like_probability=False):
#         self.models = None
#         self.bin_size = bin_size
#         self.plot_like_probability = plot_like_probability
#
#     def plot(self, models):
#         self.models = models
#         fig = go.Figure()
#         for model in models:
#             mse_data = [cell.mse for cell in model if cell.mse is not None]
#             if self.plot_like_probability:
#                 fig.add_trace(go.Histogram(x=mse_data, histnorm='probability', name=model.model_name))
#             else:
#                 fig.add_trace(go.Histogram(x=mse_data, name=model.model_name))
#
#         fig.update_layout(barmode='overlay')
#         fig.update_traces(opacity=0.75)
#         fig.update_layout(
#             title_text='Sampled Results',  # title of plot
#             xaxis_title_text='MSE in cells',  # xaxis label
#             yaxis_title_text='Count',  # yaxis label
#         )
#         fig.show()
