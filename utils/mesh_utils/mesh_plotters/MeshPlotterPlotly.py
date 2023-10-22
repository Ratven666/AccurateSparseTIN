import plotly.graph_objects as go

from utils.mesh_utils.mesh_plotters.MeshPlotterABC import MeshPlotterABC


class MeshPlotterPlotly(MeshPlotterABC):

    def __init__(self, min_mse=None, max_mse=None):
        self.min_mse = min_mse
        self.max_mse = max_mse
        self.mesh = None
        self.plot_data = None

    def plot(self, mesh):
        """
        Запускает процедуру визуализации триангуляционной поверхности
        1. Упорядочивает необходимые для построения графика данные из объекта поверхности mesh
        2. Загружает данные в область построения fig
        3. Рассчитывает область построения графика и применяет их к области fig
        4. Запускает отображение построенных данных
        :param mesh: поверхность, которую визуализируем
        :return: None
        """
        self.mesh = mesh
        plot_data = self.__get_plot_data()
        c_data = self.__calk_faces_colors()
        c_lst = c_data["c_lst"]
        self.min_mse, self.max_mse = self.__get_mse_limits()
        c_mse = [self.__get_color_for_mse(mse) for mse in c_data["c_mse"]]
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=plot_data["x"],
                                        y=plot_data["y"],
                                        z=plot_data["z"],
                                        i=plot_data["i"],
                                        j=plot_data["j"],
                                        k=plot_data["k"],
                                        opacity=1,
                                        facecolor=c_mse,
                                        ))
        fig.add_trace(go.Mesh3d(x=plot_data["x"],
                                        y=plot_data["y"],
                                        z=plot_data["z"],
                                        i=plot_data["i"],
                                        j=plot_data["j"],
                                        k=plot_data["k"],
                                        opacity=1,
                                        facecolor=c_lst,
                                        ))
        button_layer_1_height = 1.02
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[{"visible": [False, True]}],
                            label="RGB",
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
        self.__set_plot_limits(fig)
        fig.show()

    def __get_plot_data(self):
        """
        Упорядочивает необходимые для построения графика данные из объекта поверхности
        """
        points_id_dict = {}
        plot_data = {"x": [], "y": [], "z": [],
                     "i": [], "j": [], "k": [],
                     "c_list": [], "mse": []}
        for triangle in self.mesh:
            for idx, point in enumerate(triangle):
                if point.id in points_id_dict:
                    point_idx = points_id_dict[point.id]
                else:
                    point_idx = len(plot_data["x"])
                    plot_data["x"].append(point.X)
                    plot_data["y"].append(point.Y)
                    plot_data["z"].append(point.Z)
                    plot_data["c_list"].append([point.R, point.G, point.B])
                    points_id_dict[point.id] = point_idx
                if idx == 0:
                    plot_data["i"].append(point_idx)
                elif idx == 1:
                    plot_data["j"].append(point_idx)
                else:
                    plot_data["k"].append(point_idx)
            plot_data["mse"].append(triangle.mse)
        self.plot_data = plot_data
        return plot_data

    def __calk_faces_colors(self):
        """
        Рассчитывает цвета треугольников на основании усреднения цветов точек, образующих
        треугольник
        :return: список цветов треугольников в формате библиотеки plotly
        """
        c_lst = []
        c_mse = []
        for idx in range(len(self.plot_data["i"])):
            c_i = self.plot_data["c_list"][self.plot_data["i"][idx]]
            c_j = self.plot_data["c_list"][self.plot_data["j"][idx]]
            c_k = self.plot_data["c_list"][self.plot_data["k"][idx]]
            r = round((c_i[0] + c_j[0] + c_k[0]) / 3)
            g = round((c_i[1] + c_j[1] + c_k[1]) / 3)
            b = round((c_i[2] + c_j[2] + c_k[2]) / 3)
            c_lst.append(f"rgb({r}, {g}, {b})")
            c_mse.append(self.plot_data["mse"][idx])
        return {"c_lst": c_lst, "c_mse": c_mse}

    def __set_plot_limits(self, fig):
        """
        Устанавливает в графике области построения для сохранения пропорций вдоль осей
        :param fig: объект фигуры для которого будет задаваться настройка осей
        :return: None
        """
        pl = self.__calk_plot_limits()
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=pl["X_lim"]),
                yaxis=dict(range=pl["Y_lim"]),
                zaxis=dict(range=pl["Z_lim"])),
            margin=dict(r=10, l=10, b=10, t=10))

    def __calk_plot_limits(self):
        """
        Рассчитывает область построения модели для сохранения пропорций вдоль осей
        :return: Словарь с пределами построения модель вдоль трех осей
        """
        min_x, min_y, min_z = min(self.plot_data["x"]), min(self.plot_data["y"]), min(self.plot_data["z"])
        max_x, max_y, max_z = max(self.plot_data["x"]), max(self.plot_data["y"]), max(self.plot_data["z"])
        limits = [max_x - min_x,
                  max_y - min_y,
                  max_z - min_z]
        length = max(limits) / 2
        x_lim = [((min_x + max_x) / 2) - length, ((min_x + max_x) / 2) + length]
        y_lim = [((min_y + max_y) / 2) - length, ((min_y + max_y) / 2) + length]
        z_lim = [((min_z + max_z) / 2) - length, ((min_z + max_z) / 2) + length]
        return {"X_lim": x_lim, "Y_lim": y_lim, "Z_lim": z_lim}

    def __get_mse_limits(self):
        if self.min_mse is not None and self.max_mse is not None:
            return self.min_mse, self.max_mse
        min_mesh_mse = float("inf")
        max_mesh_mse = 0
        for triangle in self.mesh:
            mse = triangle.mse
            if mse is None:
                continue
            if mse < min_mesh_mse:
                min_mesh_mse = mse
            if mse > max_mesh_mse:
                max_mesh_mse = mse
        if min_mesh_mse - max_mesh_mse == float("inf"):
            return 0, 0
        if self.min_mse is not None:
            return self.min_mse, max_mesh_mse
        if self.max_mse is not None:
            return min_mesh_mse, self.max_mse
        return min_mesh_mse, max_mesh_mse

    def __get_color_for_mse(self, mse):
        if mse is None or mse == 0:
            return "rgb(0, 0, 255)"
        if mse > self.max_mse:
            return "rgb(255, 0, 0)"
        if mse < self.min_mse:
            return "rgb(0, 255, 0)"
        half_mse_delta = (self.max_mse - self.min_mse) / 2
        mse = mse - half_mse_delta - self.min_mse
        gradient_color = 255 - round((255 * abs(mse)) / half_mse_delta)
        if mse > 0:
            return f"rgb(255, {gradient_color}, 0)"
        elif mse < 0:
            return f"rgb({gradient_color}, 255, 0)"
        else:
            return "rgb(255, 255, 0)"
