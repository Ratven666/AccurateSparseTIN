import numpy as np
import plotly.graph_objects as go


class SegmentModelPlotly:

    def __init__(self):
        self.model = None

    def plot(self, model):
        self.model = model

        x = np.full((self.model.voxel_model.X_count * 2), None)
        y = np.full((self.model.voxel_model.Y_count * 2), None)
        z = np.full((self.model.voxel_model.Y_count * 2,
                     self.model.voxel_model.X_count * 2), None)
        mse = np.full((self.model.voxel_model.Y_count * 2,
                       self.model.voxel_model.X_count * 2), None)

        for cell in self.model:
            x0 = cell.voxel.X
            y0 = cell.voxel.Y
            step = cell.voxel.step

            try:
                i, j = self.__calk_indexes(cell)
            except TypeError:
                continue
            z[j][i] = cell.get_z_from_xy(x0, y0)
            z[j + 1][i] = cell.get_z_from_xy(x0, y0 + step)
            z[j][i + 1] = cell.get_z_from_xy(x0 + step, y0)
            z[j + 1][i + 1] = cell.get_z_from_xy(x0 + step, y0 + step)
            mse[j][i] = cell.mse
            mse[j + 1][i] = cell.mse
            mse[j][i + 1] = cell.mse
            mse[j + 1][i + 1] = cell.mse
            x[i], x[i + 1] = x0, x0 + step
            y[j], y[j + 1] = y0, y0 + step
        fig = go.Figure()

        fig.add_trace(go.Surface(x=x,
                                 y=y,
                                 z=z,
                                 surfacecolor=mse,
                                 colorscale="RdYlGn_r"))

        fig.add_trace(go.Surface(x=x,
                                 y=y,
                                 z=z,
                                 surfacecolor=z,
                                 colorscale="Rainbow_r"))

        button_layer_1_height = 1.02
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[{"visible": [False, True]}],
                            label="Elevation",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [True, False]}],
                            label="MSE",
                            method="update"
                        )
                    ]),
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.37,
                    xanchor="left",
                    y=button_layer_1_height,
                    yanchor="top"
                ),
            ]
        )
        width, height = self.__calk_sizes()
        fig.update_layout(
            width=width,
            height=height,
            autosize=False,
            margin=dict(t=100, b=0, l=0, r=0),
        )

        pl = self.__calk_plot_limits()
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=pl["X_lim"]),
                yaxis=dict(range=pl["Y_lim"]),
                zaxis=dict(range=pl["Z_lim"])),
            margin=dict(r=10, l=10, b=10, t=10))

        fig.update_layout(title_text=f"Model {self.model.model_name}")

        fig.update_layout(
            annotations=[
                dict(text="Texture<br>Type", x=0.25, xref="paper", y=1.01,
                     yref="paper", showarrow=False),
            ])

        fig.update_layout(
            scene={
                'camera_eye': {"x": 0, "y": -2, "z": 1},
            })

        fig.show()

    def __calk_indexes(self, cell):
        """
        Рассчитывает индексы вокселя внутри модели по трем осям
        на основании координат вокселя, его размера и области модели
        :param cell: ячейка для которой выполняется расчет индекса
        :return: кортеж с индексами ячейки модели
        """
        try:
            voxel = cell.voxel
        except AttributeError:
            return None
        x0 = self.model.voxel_model.min_X
        y0 = self.model.voxel_model.min_Y
        i = (int((voxel.X - x0) / self.model.voxel_model.step)) * 2
        j = (int((voxel.Y - y0) / self.model.voxel_model.step)) * 2
        return i, j

    def __calk_sizes(self, width=800):
        c = self.model.voxel_model.Y_count / self.model.voxel_model.X_count
        height = round(width * c)
        return width, height

    def __calk_plot_limits(self):
        """
        Рассчитывает область построения скана для сохранения пропорций вдоль осей
        :return: Словарь с пределами построения модель вдоль трех осей
        """
        model = self.model.voxel_model
        min_x, min_y, min_z = model.min_X, model.min_Y, model.min_Z
        max_x, max_y, max_z = model.max_X, model.max_Y, model.max_Z
        limits = [max_x - min_x,
                  max_y - min_y,
                  max_z - min_z]
        length = max(limits) / 2
        x_lim = [((min_x + max_x) / 2) - length, ((min_x + max_x) / 2) + length]
        y_lim = [((min_y + max_y) / 2) - length, ((min_y + max_y) / 2) + length]
        z_lim = [((min_z + max_z) / 2) - length, ((min_z + max_z) / 2) + length]
        return {"X_lim": x_lim, "Y_lim": y_lim, "Z_lim": z_lim}
