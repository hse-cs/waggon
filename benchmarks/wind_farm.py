from waggon import functions as f

import numpy as np
import matplotlib.pyplot as plt

import floris.layout_visualization as layoutviz
from floris.flow_visualization import visualize_cut_plane
from floris import (
    FlorisModel,
    WindRose,
    TimeSeries,
)


class WindFarm(f.Function):
    def __init__(self,
                n_obs,
                minimise=True,
                number_of_turbines=8,
                wind_directions=np.arange(220.0, 360.0, 3.0),
                wind_speeds=np.arange(2.0, 18.0, 1.0),
                freq_table=None,
                path_to_config='./gch.yaml'):
        super(f.Function, self).__init__()

        constraints_per_turbine = np.array([[0, 1000], [0, 1000], [0, 360]])


        self.wind_directions = wind_directions
        self.wind_speeds = wind_speeds
        self.freq_table = freq_table
        if freq_table is None:
            self.freq_table = np.random.rand(len(self.wind_directions), len(self.wind_speeds))
            self.freq_table /= self.freq_table.sum()
        self.number_of_turbines = number_of_turbines
        self.dim           = 24
        self.domain        = np.tile(constraints_per_turbine, (number_of_turbines, 1))
        self.name          = 'wind_farm'
        self.fmodel        = FlorisModel(path_to_config)
        self.f_min         = None
        self.glob_min	   = None
        self.wind_rose = WindRose(
            wind_directions=self.wind_directions,
            wind_speeds=self.wind_speeds,
            freq_table=self.freq_table,
            ti_table=0.06 * np.ones((len(wind_directions), len(wind_speeds))),
        )
        # self.wind_rose.assign_value_piecewise_linear(value_zero_ws=25 * 1.425, slope_2=-25 * 0.135)
        self.fmodel.set(
           wind_data=self.wind_rose
        )
        # self.glob_min      = yaw_optimization(self.fmodel, np.column_stack((np.arange(self.number_of_turbines) * 100, np.arange(self.number_of_turbines) * 100, np.random.uniform(0, 90, self.number_of_turbines))))
        self.f             = lambda x: self.__call__(x)
        self.log_transform = True
        self.log_eps       = 1e-8
        self.sigma         = 1e-1
        self.n_obs         = n_obs
        self.minimise      = minimise
        self.seed          = 73
        self.time_series =  TimeSeries(
                                wind_directions=self.wind_directions,
                                wind_speeds=8.0,
                                turbulence_intensities=0.06,
                            )

        self.fmodel = FlorisModel(path_to_config)

    def __call__(self, X : np.array):
        results = []

        for config in X:
            # print(config)
            config = config.reshape(self.number_of_turbines, 3)

            self.fmodel.set(layout_x=config[:,0], layout_y=config[:,1])
            self.fmodel.set(yaw_angles=np.expand_dims(config[:,2], 0).astype(np.float64))

            wind_sample = np.array([np.random.choice(self.wind_directions)])
            wind_speed = np.array([np.random.choice(self.wind_speeds)])

            wind_rose_temp = WindRose(
                wind_directions=wind_sample,
                wind_speeds=wind_speed,
                ti_table=0.06,
                compute_zero_freq_occurrence=True
            )

            self.fmodel.set(wind_data=wind_rose_temp)

            self.fmodel.run()

            # print(np.sum(distance.pdist(X))*1.5)
            aep = self.fmodel.get_farm_AEP()
            # turbine_powers = np.nan_to_num(self.fmodel.get_turbine_powers()) - np.sum(distance.pdist(config))*1.5
            results.append(aep/1E9)

        # print(turbine_powers)
        results = np.array(results)
        # average_turbine_powers = np.array([ np.array(results).mean()] * len(results))
        if self.minimise:
            return -1 * np.expand_dims(results, 0) # TODO remake
        else:
            return np.expand_dims(results, 0)
        
    def plot_rose(self, ax=None):
        return self.wind_rose.plot(ax=ax)
        # wind_rose_aggregate = self.downsample(wd_step, ws_step, inplace=False)
        # wd_bins = wind_rose_aggregate.wind_directions
        # ws_bins = wind_rose_aggregate.wind_speeds
        # freq_table = wind_rose_aggregate.freq_table

        # # Set up figure
        # if ax is None:
        #     _, ax = plt.subplots(subplot_kw={"polar": True})

        # # Get the wd_step
        # if wd_step is None:
        #     if len(wd_bins) >= 2:
        #         wd_step = wd_bins[1] - wd_bins[0]
        #     else:
        #         # This admittedly an odd edge case
        #         wd_step = 360.0

        # # Get a color array
        # color_array = plt.get_cmap(color_map, len(ws_bins))
        # norm_ws = mpl.colors.Normalize(vmin=np.min(ws_bins), vmax=np.max(ws_bins))
        # sm_ws = mpl.cm.ScalarMappable(norm=norm_ws, cmap=color_array)

        # for wd_idx, wd in enumerate(wd_bins):
        #     rects = []
        #     freq_table_sub = freq_table[wd_idx, :].flatten()
        #     for ws_idx, ws in reversed(list(enumerate(ws_bins))):
        #         plot_val = freq_table_sub[: ws_idx + 1].sum()
        #         rects.append(
        #             ax.bar(
        #                 np.radians(wd),
        #                 plot_val,
        #                 width=0.9 * np.radians(wd_step),
        #                 color=color_array(ws_idx),
        #                 edgecolor="k",
        #             )
        #         )

        # # Configure the plot
        # ax.figure.colorbar(sm_ws, ax=ax, **legend_kwargs)
        # ax.figure.tight_layout()
        # ax.set_theta_direction(-1)
        # ax.set_theta_offset(np.pi / 2.0)
        # ax.set_theta_zero_location("N")
        # ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
        # ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

        # return ax
  
    def sample(self, x):

        # x_ = np.array([])
        # std_x = x[:,0].std()
        # std_y = x[:,1].std()
        # std_yaw = x[:,2].std()
        # for _ in range(self.n_obs):

        # y_ = np.array([])
        # for _ in range(self.n_obs):
        #     new_layout = np.zeros(x.shape)
        #     new_layout[:,0] = x[:,0] + np.random.normal(0, std_x, x[:,0].shape)
        #     new_layout[:,1] = x[:,1] + np.random.normal(0, std_y, x[:,1].shape)
        #     new_layout[:,2] = x[:,2] + np.random.normal(0, std_yaw, x[:,2].shape)
        #     new_value = self(new_layout)

        #     if len(x_) == 0:
        #         x_ = new_layout
        #         y_ = new_value
        #     else:
        #         x_ = np.concatenate([x_, new_layout], axis=0)
        #         y_ = np.concatenate([y_, new_value], axis=1)

        # y_ = np.transpose(y_)

        return x, self(x).T


    def plot(self, x : np.array):

        x = x.reshape(self.number_of_turbines, 3)
        # self.fmodel.set(wind_data=self.wind_rose)

        self.fmodel = FlorisModel('gch.yaml')
        self.fmodel.set(layout_x=x[:,0], layout_y=x[:,1])
        self.fmodel.set(yaw_angles=np.expand_dims(x[:,2], 0).astype(np.float64))
        self.fmodel.set(wind_speeds=[8.0], wind_directions=[290.0], turbulence_intensities=[0.06])
        # self.fmodel.run()

        # ax = axarr[0]
        fig, axarr = plt.subplots(1, 1, figsize=(16, 10), sharex=False)

        # self.fmodel.set(yaw_angles=np.expand_dims(x[:,2], 0).astype(np.float64))
        horizontal_plane = self.fmodel.calculate_horizontal_plane(height=90.0)
        visualize_cut_plane(horizontal_plane, ax=axarr, min_speed=1.0, max_speed=18.)
        layoutviz.plot_turbine_rotors(self.fmodel, ax=axarr, yaw_angles=np.expand_dims(x[:,2], 0).astype(np.float64))
        axarr.set_title("Flow visualization with yawed turbine")

        plt.show()
        import warnings
        warnings.filterwarnings('ignore')

        # ax = axarr[1]

        # bbox = ax.get_position()
        # # Удаляем исходный subplot
        # ax.remove()
        # # Создаем полярный subplot на том же месте
        # ax = fig.add_axes(bbox, projection='polar')
        # self.rose.plot(ax=ax,legend_kwargs={"label": "WS"})
        # ax.set_title("Wind Direction and Wind Speed Frequencies")
        # plt.tight_layout()
        # fig.suptitle("WindRose Plot")
