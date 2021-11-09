"""Generate Map Plot Figure."""
# Standard library
import locale
import os
from datetime import time
from typing import Sequence

# Third-party
# map plotting packages
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.io.shapereader import Record  # type: ignore
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# plt.rcParams["figure.autolayout"] = True


def create_coord_dict(altitude_levels):
    assert (
        altitude_levels <= 10
    ), "It is not possible, to generate altitude plots for more than 10 different starting altitudes."

    coord_dict = {}

    key_name = "altitude_"

    i = 1

    while i < altitude_levels + 1:
        altitude_dict = {
            "origin": None,
            "y_type": None,
            "alt_level": None,
            "subplot_index": None,
            "traj_0": {
                "lon": [],
                "lat": [],
                "time": [],
                "z_type": None,
                "alpha": 1,
            },  # main trajectory
            "traj_1": {
                "lon": [],
                "lat": [],
                "time": [],
                "z_type": None,
                "alpha": 0.5,
            },  # side trajectory 1
            "traj_2": {
                "lon": [],
                "lat": [],
                "time": [],
                "z_type": None,
                "alpha": 0.5,
            },  # side trajectory 2
            "traj_3": {
                "lon": [],
                "lat": [],
                "time": [],
                "z_type": None,
                "alpha": 0.5,
            },  # side trajectory 3
            "traj_4": {
                "lon": [],
                "lat": [],
                "time": [],
                "z_type": None,
                "alpha": 0.5,
            },  # side trajectory 4
        }
        coord_dict[key_name + str(i)] = altitude_dict
        i += 1

    return coord_dict


def plot_map(trajectory_dict, output_dir, separator, language, domain):
    for key in trajectory_dict:  # iterate through the trajectory dict
        print(f"--- defining trajectory plot properties for {key}")

        coord_dict = create_coord_dict(
            altitude_levels=trajectory_dict[key]["altitude_levels"].loc[0]
        )

        trajectory_df = trajectory_dict[key]  # extract df for given key

        # not sure if necessary for anything...
        dt = abs(trajectory_df["time"].loc[0] - trajectory_df["time"].loc[1])

        if str(dt)[-2:] == ".3":
            dt += 0.2

        number_of_times = trajectory_df["block_length"].iloc[
            0
        ]  # block length is constant, because it depends on the runtime of the model and the timestep, which both are constant for a given traj file
        number_of_trajectories = trajectory_df["#trajectories"].iloc[
            0
        ]  # this parameter, corresponds to the number of rows, present in the start file, thus also constant
        x = trajectory_df["datetime"].iloc[
            0:number_of_times
        ]  # shared x-axis is the time axis, which is constant for a given traj file

        row_index = 0  # corresponds to row from start file
        alt_index = (
            1  # altitude 1,2,3,4,... (however many starting altitudes there are)
        )
        traj_index = 0  # 0=main, 1=east, 2=north, 3=west, 4=south

        while row_index < number_of_trajectories:

            lower_row = row_index * number_of_times
            upper_row = row_index * number_of_times + number_of_times

            origin = trajectory_df["origin"].loc[lower_row]
            altitude_levels = trajectory_df["altitude_levels"].loc[lower_row]
            subplot_index = trajectory_df["subplot_index"].loc[lower_row]
            max_start_altitude = trajectory_df["max_start_altitude"].loc[lower_row]

            if trajectory_df["side_traj"].loc[
                lower_row
            ]:  # if there are side trajectories, each subplot contains 5 trajectories
                traj_per_plot = 5
                side_traj = 1
            else:
                traj_per_plot = 1  # else, there is only one trajectory
                side_traj = 0

            if side_traj:
                if (
                    separator not in origin  # replaced '~' with separator
                ):  # the y_surf information, is only taken from the main trajectories (not side trajectories)
                    # print(f'row_index = {row_index} corresponds to origin {origin}')
                    coord_dict["altitude_" + str(alt_index)]["origin"] = origin
                    coord_dict["altitude_" + str(alt_index)][
                        "subplot_index"
                    ] = subplot_index
                    # print(trajectory_df['hsurf'][lower_row:upper_row])

                coord_dict["altitude_" + str(alt_index)]["y_type"] = trajectory_df[
                    "z_type"
                ][lower_row]
                coord_dict["altitude_" + str(alt_index)]["alt_level"] = trajectory_df[
                    "start_altitude"
                ][lower_row]

                coord_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                    "lon"
                ] = trajectory_df["lon"][lower_row:upper_row]

                coord_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                    "lat"
                ] = trajectory_df["lat"][lower_row:upper_row]

                coord_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                    "time"
                ] = trajectory_df["time"][lower_row:upper_row]

                coord_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                    "z_type"
                ] = trajectory_df["z_type"][lower_row]

                row_index += 1
                traj_index += 1

                if traj_index == 5:
                    traj_index = 0
                    alt_index += 1

                if alt_index > altitude_levels:
                    alt_index = 1
                    generate_map_plot(
                        coord_dict=coord_dict,
                        key=key,
                        side_traj=side_traj,
                        output_dir=output_dir,
                        altitude_levels=altitude_levels,
                        language=language,
                        domain=domain,
                        dt=dt,
                    )

            else:
                # print(f'row_index = {row_index} corresponds to origin {origin}')
                coord_dict["altitude_" + str(alt_index)]["origin"] = origin
                coord_dict["altitude_" + str(alt_index)][
                    "subplot_index"
                ] = subplot_index
                coord_dict["altitude_" + str(alt_index)]["y_type"] = trajectory_df[
                    "z_type"
                ][lower_row]
                coord_dict["altitude_" + str(alt_index)]["alt_level"] = trajectory_df[
                    "start_altitude"
                ][lower_row]

                coord_dict["altitude_" + str(alt_index)]["traj_0"][
                    "lon"
                ] = trajectory_df["lon"][lower_row:upper_row]

                coord_dict["altitude_" + str(alt_index)]["traj_0"][
                    "lat"
                ] = trajectory_df["lat"][lower_row:upper_row]

                coord_dict["altitude_" + str(alt_index)]["traj_0"][
                    "time"
                ] = trajectory_df["time"][lower_row:upper_row]

                # since there are no side-traj, these keys remain empty
                coord_dict["altitude_" + str(alt_index)]["traj_1"]["lon"] = []
                coord_dict["altitude_" + str(alt_index)]["traj_1"]["lat"] = []
                coord_dict["altitude_" + str(alt_index)]["traj_2"]["lon"] = []
                coord_dict["altitude_" + str(alt_index)]["traj_2"]["lat"] = []
                coord_dict["altitude_" + str(alt_index)]["traj_3"]["lon"] = []
                coord_dict["altitude_" + str(alt_index)]["traj_3"]["lat"] = []
                coord_dict["altitude_" + str(alt_index)]["traj_4"]["lon"] = []
                coord_dict["altitude_" + str(alt_index)]["traj_4"]["lat"] = []

                coord_dict["altitude_" + str(alt_index)]["traj_0"][
                    "z_type"
                ] = trajectory_df["z_type"][lower_row]

                coord_dict["altitude_" + str(alt_index)]["traj_1"]["z_type"] = None
                coord_dict["altitude_" + str(alt_index)]["traj_2"]["z_type"] = None
                coord_dict["altitude_" + str(alt_index)]["traj_3"]["z_type"] = None
                coord_dict["altitude_" + str(alt_index)]["traj_4"]["z_type"] = None

                row_index += 1
                alt_index += 1
                if alt_index > altitude_levels:
                    alt_index = 1
                    generate_map_plot(
                        coord_dict=coord_dict,
                        key=key,
                        side_traj=side_traj,
                        output_dir=output_dir,
                        altitude_levels=altitude_levels,
                        language=language,
                        domain=domain,
                        dt=dt,
                    )
    return


def add_features(map):
    map.coastlines()
    map.add_feature(cfeature.LAND)
    map.add_feature(cfeature.COASTLINE)
    map.add_feature(cfeature.BORDERS, linestyle="--")
    map.add_feature(cfeature.OCEAN)
    map.add_feature(cfeature.LAKES)
    map.add_feature(cfeature.RIVERS)
    # map.add_feature(cfeature.STATES)
    return


def crop_map(map, domain):

    # got these pre-defined domain boundaries from: https://github.com/MeteoSwiss-APN/oprtools/blob/master/dispersion/lib/get_domain.pro

    domain_dict = {
        "centraleurope": {
            "lat_0": 2,  # x_0
            "lat_1": 18,  # x_1
            "lon_0": 42,  # y_0
            "lon_2": 52,  # y_1
            "domain": [2, 18, 42, 52],  # [lat0,lat1,lon0,lon1]
        },
        "ch": {  # zoom domain Switzerland
            "lat_0": 5.8,  # x_0
            "lat_1": 10.6,  # x_1
            "lon_0": 45.4,  # y_0
            "lon_2": 48.2,  # y_1
            "domain": [5.8, 10.6, 45.4, 48.2],  # [lat0,lat1,lon0,lon1]
        },
        "alps": {  # zoom domain alps for IFS-HRES
            "lat_0": 2,  # x_0
            "lat_1": 14,  # x_1
            "lon_0": 43,  # y_0
            "lon_2": 50,  # y_1
            "domain": [2, 14, 43, 50],  # [lat0,lat1,lon0,lon1]
        },
        "europe": {  # zoom domain Europe (for IFS-HRES)
            "lat_0": -10,  # x_0
            "lat_1": 47,  # x_1
            "lon_0": 35,  # y_0
            "lon_2": 65,  # y_1
            "domain": [-10, 47, 35, 65],  # [lat0,lat1,lon0,lon1]
        },
        "ch_hd": {  # zoom domain larger Siwtzerlad area for COSMO-7
            "lat_0": 3.5,  # x_0
            "lat_1": 12.6,  # x_1
            "lon_0": 44.1,  # y_0
            "lon_2": 49.4,  # y_1
            "domain": [3.5, 12.6, 44.1, 49.4],  # [lat0,lat1,lon0,lon1]
        },
    }

    domain_boundaries = domain_dict[domain]["domain"]

    map.set_extent(
        domain_boundaries, crs=ccrs.PlateCarree()
    )  # Central Europe for IFS-HRES

    return domain_boundaries


def add_time_interval_points(coord_dict, map, i, linestyle):
    lon_important, lat_important = retrieve_interval_points(coord_dict, i)

    # marker styles: https://matplotlib.org/stable/api/markers_api.html
    # add 6 hour interval points
    if i == 1:
        map.scatter(
            lon_important,
            lat_important,
            marker="d",
            color=linestyle[:-1],
            label="6,12,...h",
        )
    else:
        map.scatter(lon_important, lat_important, marker="d", color=linestyle[:-1])


def retrieve_interval_points(coord_dict, i):
    lat_df_tmp = pd.DataFrame(coord_dict["altitude_" + str(i)]["traj_0"]["lat"].items())
    lon_df_tmp = pd.DataFrame(coord_dict["altitude_" + str(i)]["traj_0"]["lon"].items())
    time_df_tmp = pd.DataFrame(
        coord_dict["altitude_" + str(i)]["traj_0"]["time"].items()
    )
    # delete index columns
    del lon_df_tmp[0]
    del lat_df_tmp[0]
    del time_df_tmp[0]
    # rename remaining column
    lon_df = lon_df_tmp.rename(columns={1: "lon"})
    lat_df = lat_df_tmp.rename(columns={1: "lat"})
    time_df = time_df_tmp.rename(columns={1: "time"})
    # combine columns to one df
    comb_df = pd.concat([lat_df, lon_df, time_df], axis=1, join="inner")
    # extract position every 6 hours into important_points dataframe
    important_points_tmp = comb_df[comb_df["time"] % 6 == 0]
    important_points = important_points_tmp.iloc[1:]
    lon_important = important_points["lon"]
    lat_important = important_points["lat"]
    return lon_important, lat_important


def is_visible(city) -> bool:
    return True
    """Check if a point is inside the domain."""
    px_geo: float = city.geometry.x
    py_geo: float = city.geometry.y
    # pylint: disable=E0633  # unpacking-non-sequence
    px_ax, py_ax = self.trans.geo_to_axes(px_geo, py_geo)
    in_domain = 0.0 <= px_ax <= 1.0 and 0.0 <= py_ax <= 1.0
    if not in_domain:
        return False
    if self.ref_dist_box is None:
        behind_ref_dist_box = False
    else:
        behind_ref_dist_box = (
            self.ref_dist_box.x0_box <= px_ax <= self.ref_dist_box.x1_box
            and self.ref_dist_box.y0_box <= py_ax <= self.ref_dist_box.y1_box
        )
    return not behind_ref_dist_box


def is_of_interest(city) -> bool:
    """Check if a city fulfils certain importance criteria."""
    is_capital = city.attributes["FEATURECLA"].startswith("Admin-0 capital")
    is_large = city.attributes["GN_POP"] > 5000000
    excluded_cities = [
        "Incheon",
    ]
    is_excluded = city.attributes["name_en"] in excluded_cities
    return (is_capital or is_large) and not is_excluded


def get_name(city) -> str:
    """Fetch city name in current language, hand-correcting some."""
    name = city.attributes[f"name_en"]
    if name.startswith("Freiburg im ") and name.endswith("echtland"):
        name = "Freiburg"
    return name


def add_cities(map, domain_boundaries):
    # retrieve cities from shapefile
    cities: Sequence[Record] = shpreader.Reader(
        shpreader.natural_earth(
            resolution="110m", category="cultural", name="populated_places"
        )
    ).records()

    cities_by_name = {get_name(city): city for city in cities}
    for name, city in sorted(cities_by_name.items()):
        lon, lat = city.geometry.x, city.geometry.y
        if is_visible(city) and is_of_interest(city):
            print(f'{name} has roughly {city.attributes["GN_POP"]} population')
            plt.scatter(
                x=lon,
                y=lat,
                marker="o",
                color="black",
            )
            # Note: `clip_on=True` doesn't work in cartopy v0.18
        #     text.set_clip_path(plot_domain)


def generate_map_plot(
    coord_dict, key, side_traj, output_dir, altitude_levels, language, domain, dt
):
    origin = coord_dict["altitude_1"]["origin"]
    print(f"--- generating map plot for {origin}")

    # this dict defines the subplot position and figure size for the different domains,
    # which all have different aspect ratios
    figsize_dict = {
        "centraleurope": {
            "figsize": (16, 10),
            "left": 0.85,
            "right": 0.9,
            "top": 0.7,
            "bottom": 0.3,
        },
        "ch": {"figsize": [9, 5], "left": 0.98, "right": 1, "top": 0.9, "bottom": 0.2},
        "alps": {
            "figsize": (12, 7),
            "left": 0.85,
            "right": 0.9,
            "top": 0.7,
            "bottom": 0.3,
        },
        "europe": {
            "figsize": [2 * 5.7, 2 * 3.0],
            "left": 0.85,
            "right": 0.9,
            "top": 0.7,
            "bottom": 0.3,
        },
        "ch_hd": {
            "figsize": (9.1, 5.3),
            "left": 0.9,
            "right": 0.95,
            "top": 0.7,
            "bottom": 0.3,
        },
    }

    fig = plt.figure(
        figsize=figsize_dict[domain]["figsize"],
        # constrained_layout=False,  # enable, s.t. everything fits nicely on the figure --> incompatible w/ cartopy
    )

    map = fig.add_subplot(
        1,
        1,
        1,
        projection=ccrs.PlateCarree(),  # choose projection
        frameon=True,  # add/remove frame of map. I think it looks better w/o a frame
    )

    # position subplot on figure, depending on aspect ratio of domain
    fig.subplots_adjust(
        left=figsize_dict[domain]["left"],
        bottom=figsize_dict[domain]["bottom"],
        right=figsize_dict[domain]["right"],
        top=figsize_dict[domain]["top"],
    )
    domain_boundaries = crop_map(map=map, domain=domain)
    print(domain_boundaries)

    gl = map.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="k",
        alpha=0.3,
        linestyle="-.",
    )  # define grid line properties

    gl.top_labels = False  # no x-axis on top
    gl.right_labels = False  # no y-axis on the right

    subplot_properties_dict = {
        0: "k-",
        1: "g-",
        2: "b-",
        3: "r-",
        4: "c-",
        5: "m-",
        6: "y-",
        7: "deepskyblue-",
        8: "crimson-",
        9: "lightgreen-",
    }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # HERE, THE ACTUAL PLOTTING HAPPENS.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    i = 1
    while i <= altitude_levels:
        alt_level = coord_dict["altitude_" + str(i)]["alt_level"]
        sub_index = int(coord_dict["altitude_" + str(i)]["subplot_index"])
        textstr = (
            str(coord_dict["altitude_" + str(i)]["alt_level"])
            + " "
            + coord_dict["altitude_" + str(i)]["y_type"]
        )

        # print(f'altitude_{i} = {alt_level} --> subplot {sub_index} (have {altitude_levels} alt levels/subplots)')

        if side_traj:
            traj_index = [0, 1, 2, 3, 4]

            for traj in traj_index:

                # textstr = (
                #     str(coord_dict["altitude_" + str(i)]["alt_level"])
                #     + " "
                #     + coord_dict["altitude_" + str(i)]["y_type"]
                # )

                latitude = coord_dict["altitude_" + str(i)]["traj_" + str(traj)]["lat"]
                longitude = coord_dict["altitude_" + str(i)]["traj_" + str(traj)]["lon"]

                ystart = latitude.iloc[0]
                xstart = longitude.iloc[0]

                linestyle = subplot_properties_dict[sub_index]
                # print(f'linestyle for subplot {sub_index}: {linestyle}')
                alpha = coord_dict["altitude_" + str(i)]["traj_" + str(traj)]["alpha"]

                if (
                    coord_dict["altitude_" + str(i)]["traj_" + str(traj)]["alpha"] == 1
                ):  # only add legend & startpoint for the main trajectories

                    # plot main trajectory
                    map.plot(
                        longitude,  # define x-axis
                        latitude,  # define y-axis
                        linestyle,  # define linestyle
                        alpha=alpha,  # define line opacity
                        label=textstr,
                    )

                    add_time_interval_points(coord_dict, map, i, linestyle)

                    # add start point triangle
                    map.plot(
                        xstart,
                        ystart,
                        marker="^",
                        markersize=10,
                        markeredgecolor="red",
                        markerfacecolor="white",
                    )

                else:
                    map.plot(
                        longitude,  # define x-axis
                        latitude,  # define y-axis
                        linestyle,  # define linestyle
                        alpha=alpha,  # define line opacity
                    )

        else:  # no side traj

            latitude = coord_dict["altitude_" + str(i)]["traj_0"]["lat"]
            longitude = coord_dict["altitude_" + str(i)]["traj_0"]["lon"]

            ystart = latitude.iloc[0]
            xstart = longitude.iloc[0]

            linestyle = subplot_properties_dict[sub_index]
            # print(f'linestyle for subplot {sub_index}: {linestyle}')
            alpha = coord_dict["altitude_" + str(i)]["traj_0"]["alpha"]

            # plot main trajectory
            map.plot(
                longitude,  # define x-axis
                latitude,  # define y-axis
                linestyle,  # define linestyle
                alpha=alpha,  # define line opacity
                label=textstr,
            )

            add_time_interval_points(
                coord_dict=coord_dict, map=map, i=i, linestyle=linestyle
            )

            # add start point triangle
            map.plot(
                xstart,
                ystart,
                marker="^",
                markersize=10,
                markeredgecolor="red",
                markerfacecolor="white",
            )

            # coord_dict[sub_index].legend()

        i += 1

    add_features(map=map)  # add features: coastlines,lakes,...
    map.legend()  # add legend
    add_cities(map=map, domain_boundaries=domain_boundaries)
    # TODO: add cities to the map
    # perhaps useful: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.itertuples.html
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    outpath = os.getcwd() + "/" + output_dir + "/plots/" + key + "/"
    os.makedirs(
        outpath, exist_ok=True
    )  # create plot folder if it doesn't already exist

    title = False  # don't want title for map plot as of now
    if title:
        if language == "en":
            locale.setlocale(locale.LC_ALL, "en_GB")
            fig.suptitle("Air trajectories originating from " + origin)
        if language == "de":
            fig.suptitle("Luft-Trajektorien Karte für " + origin)

    plt.savefig(outpath + origin + "_" + domain + ".png", dpi=100)
    plt.close(fig)

    return
