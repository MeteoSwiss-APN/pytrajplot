"""Generate PDF w/ altitude figure and map plot."""

# Standard library
import os

# Third-party
import cartopy.crs as ccrs
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np

plt.rc("axes", labelsize=8)  # change the font size of the axis labels


# Local
from .plotting.plot_altitude import generate_altitude_plot
from .plotting.plot_info_header import generate_info_header
from .plotting.plot_map import generate_map_plot
from .plotting.plot_map import get_dynamic_domain


def create_plot_dict(altitude_levels):
    """Create dict of dicts to be filled with information for all y-axes.

    Args:
        altitude_levels (int): Number of y-axis dicts (for each alt. level one dict) that need to be added to y

    Returns:
        y (dict): Dict of dicts. For each altitude level, one dict is present in this dict. Each of those 'altitude dicts' contains the relevant information to plot the corresponding subplot.

    """
    assert (
        altitude_levels <= 10
    ), "It is not possible, to generate altitude plots for more than 10 different starting altitudes."

    plot_dict = {}

    key_name = "altitude_"

    i = 1
    while i < altitude_levels + 1:
        altitude_dict = {
            "origin": None,
            "y_surf": None,
            "y_type": None,
            "alt_level": None,
            "subplot_index": None,
            "max_start_altitude": None,
            "traj_0": {
                "z": [],
                "lon": [],
                "lat": [],
                "time": [],
                "z_type": None,
                "line": None,  # "line": color_dict[i],
                "alpha": 1,
            },  # main trajectory
            "traj_1": {
                "z": [],
                "lon": [],
                "lat": [],
                "time": [],
                "z_type": None,
                "line": None,  # "line": color_dict[i],
                "alpha": 0.3,
            },  # side trajectory 1
            "traj_2": {
                "z": [],
                "lon": [],
                "lat": [],
                "time": [],
                "z_type": None,
                "line": None,  # "line": color_dict[i],
                "alpha": 0.3,
            },  # side trajectory 2
            "traj_3": {
                "z": [],
                "lon": [],
                "lat": [],
                "time": [],
                "z_type": None,
                "line": None,  # "line": color_dict[i],
                "alpha": 0.3,
            },  # side trajectory 3
            "traj_4": {
                "z": [],
                "lon": [],
                "lat": [],
                "time": [],
                "z_type": None,
                "line": None,  # "line": color_dict[i],
                "alpha": 0.3,
            },  # side trajectory 4
        }
        plot_dict[key_name + str(i)] = altitude_dict
        i += 1

    return plot_dict


def get_projection(plot_dict, altitude_levels, side_traj):
    # check if HRES or COSMO
    if plot_dict["altitude_1"]["y_type"] == "hpa":
        case = "HRES"
        # check if dateline is crossed or not
        central_longitude, _, _, cross_dateline = get_dynamic_domain(
            coord_dict=plot_dict, altitude_levels=altitude_levels, side_traj=side_traj
        )
        projection = ccrs.PlateCarree(central_longitude=central_longitude)
    else:
        case = "COSMO"
        cross_dateline = False
        projection = ccrs.RotatedPole(
            pole_longitude=-170, pole_latitude=43
        )  # define rotation of COSMO model
    return projection, cross_dateline


def assemble_pdf(
    plot_info_dict,
    x,
    plot_dict,
    key,
    side_traj,
    output_dir,
    altitude_levels,
    language,
    max_start_altitude,
    domain,
):

    # DEFINE FIGURE PROPERTIES AND GRID SPECIFICATION
    fig = plt.figure(figsize=(11.69, 8.27), dpi=200)  # A4 size
    # fig = plt.figure(figsize=(12, 8), dpi=200)
    widths = [2, 1]
    heights = [0.5] + [1] * (altitude_levels)
    # create grid spec oject
    grid_specification = gs.GridSpec(
        nrows=altitude_levels + 1, ncols=2, width_ratios=widths, height_ratios=heights
    )

    origin = plot_dict["altitude_1"]["origin"]

    # ADD INFO HEADER TO PDF
    info_ax = plt.subplot(grid_specification[0, :])
    # plot_dummy_info(data=np.random.normal(0, 1, 500), ax=info_ax)
    generate_info_header(
        language=language,
        plot_info=plot_info_dict,
        plot_data=plot_dict,
        ax=info_ax,
        domain=domain,
    )

    # ADD MAP TO PDF
    projection, cross_dateline = get_projection(
        plot_dict=plot_dict, altitude_levels=altitude_levels, side_traj=side_traj
    )
    map_ax = plt.subplot(grid_specification[1:, 0], projection=projection)
    # plot_dummy_map(ax=map_ax, projection=ccrs.PlateCarree())
    generate_map_plot(
        cross_dateline=cross_dateline,
        coord_dict=plot_dict,
        side_traj=side_traj,
        altitude_levels=altitude_levels,
        domain=domain,
        ax=map_ax,
    )

    # ADD ALTITUDE PLOT TO PDF
    alt_index = 1
    while alt_index <= altitude_levels:
        subplot_index = int(plot_dict["altitude_" + str(alt_index)]["subplot_index"])
        tmp_ax = plt.subplot(grid_specification[subplot_index + 1, 1])
        generate_altitude_plot(
            x=x,
            y=plot_dict,
            key=key,
            side_traj=side_traj,
            altitude_levels=altitude_levels,
            language=language,
            max_start_altitude=max_start_altitude,
            ax=tmp_ax,
            alt_index=alt_index,
            sub_index=subplot_index,
        )

        # plot_dummy_altitude(ax=tmp_ax, plot_index=subplot_index, altitude_levels=altitude_levels)
        alt_index += 1

    # SAVE PDF
    # plt.tight_layout()
    outpath = os.getcwd() + "/" + output_dir + "/plots/" + key + "/"
    os.makedirs(
        outpath, exist_ok=True
    )  # create plot folder if it doesn't already exist
    plt.savefig(outpath + origin + "_" + domain + ".pdf")
    plt.close(fig)
    return


def assemble_pdf_new(
    plot_info_dict,
    x,
    plot_dict,
    key,
    side_traj,
    output_dir,
    altitude_levels,
    language,
    max_start_altitude,
    domain,
):

    # DEFINE FIGURE PROPERTIES AND GRID SPECIFICATION
    fig = plt.figure(figsize=(11.69, 8.27), dpi=200)  # A4 size
    # fig = plt.figure(figsize=(12, 8), dpi=200)

    gs0 = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[0.125, 1])

    gs00 = gs0[1, 0].subgridspec(
        10, 1, width_ratios=[1], height_ratios=[0.5] + [0.1] * 9
    )  # Expected the given number of height ratios to match the number of rows of the grid
    gs01 = gs0[1, 1].subgridspec(altitude_levels + 1, 1)

    origin = plot_dict["altitude_1"]["origin"]

    # ADD INFO HEADER TO PDF
    info_ax = plt.subplot(gs0[0, :])
    # plot_dummy_info(data=np.random.normal(0, 1, 500), ax=info_ax)
    generate_info_header(
        language=language,
        plot_info=plot_info_dict,
        plot_data=plot_dict,
        ax=info_ax,
        domain=domain,
    )

    # ADD MAP TO PDF
    projection, cross_dateline = get_projection(
        plot_dict=plot_dict, altitude_levels=altitude_levels, side_traj=side_traj
    )
    map_ax = plt.subplot(gs00[0:9, :], projection=projection)
    # plot_dummy_map(ax=map_ax, projection=ccrs.PlateCarree())
    generate_map_plot(
        cross_dateline=cross_dateline,
        coord_dict=plot_dict,
        side_traj=side_traj,
        altitude_levels=altitude_levels,
        domain=domain,
        ax=map_ax,
    )

    # ADD ALTITUDE PLOT TO PDF
    alt_index = 1
    while alt_index <= altitude_levels:
        subplot_index = int(plot_dict["altitude_" + str(alt_index)]["subplot_index"])
        tmp_ax = plt.subplot(gs01[subplot_index + 1, 0])
        generate_altitude_plot(
            x=x,
            y=plot_dict,
            key=key,
            side_traj=side_traj,
            altitude_levels=altitude_levels,
            language=language,
            max_start_altitude=max_start_altitude,
            ax=tmp_ax,
            alt_index=alt_index,
            sub_index=subplot_index,
        )

        # plot_dummy_altitude(ax=tmp_ax, plot_index=subplot_index, altitude_levels=altitude_levels)
        alt_index += 1

    # SAVE PDF
    # plt.tight_layout()
    outpath = os.getcwd() + "/" + output_dir + "/plots/" + key + "/"
    os.makedirs(
        outpath, exist_ok=True
    )  # create plot folder if it doesn't already exist
    plt.savefig(outpath + origin + "_" + domain + ".pdf")
    plt.close(fig)
    return


def generate_pdf(
    trajectory_dict, plot_info_dict, output_dir, separator, language, domains
):
    """Iterate through trajectory dict. For each key, generate altitude plots for all origins/altitude levels.

    Args:
        trajectory_dict (dict): [description]
        plot_info_dict (dict): [description]
        output_dir (str): Path to output directory (where the subdirectories containing all plots should be saved)
        separator (str): Separator string to identify main- and side-trajectories (default: '~')
        language (str): Language for plot annotations.
        domains (list): List of domains, for which PDFs need to be created.

    """
    for key in trajectory_dict:  # iterate through the trajectory dict
        trajectory_df = trajectory_dict[key]  # extract df for given key
        altitude_levels = trajectory_df["altitude_levels"].loc[0]
        number_of_times = number_of_times = trajectory_df["block_length"].iloc[0]
        number_of_trajectories = trajectory_df["#trajectories"].iloc[0]

        # time axis for altitude plots (= x-axis)
        time_axis = trajectory_df["datetime"].iloc[0:number_of_times]

        plot_dict = create_plot_dict(altitude_levels=altitude_levels)

        row_index = 0
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
            side_traj = trajectory_df["side_traj"].loc[lower_row]

            if side_traj:
                if separator not in origin:
                    plot_dict["altitude_" + str(alt_index)]["origin"] = origin
                    plot_dict["altitude_" + str(alt_index)]["y_surf"] = trajectory_df[
                        "hsurf"
                    ][lower_row:upper_row]
                    plot_dict["altitude_" + str(alt_index)][
                        "subplot_index"
                    ] = subplot_index
                    plot_dict["altitude_" + str(alt_index)][
                        "max_start_altitude"
                    ] = max_start_altitude

                plot_dict["altitude_" + str(alt_index)]["y_type"] = trajectory_df[
                    "z_type"
                ][lower_row]
                plot_dict["altitude_" + str(alt_index)]["alt_level"] = trajectory_df[
                    "start_altitude"
                ][lower_row]
                plot_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                    "z"
                ] = trajectory_df["z"][lower_row:upper_row]
                plot_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                    "lon"
                ] = trajectory_df["lon"][lower_row:upper_row]
                plot_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                    "lat"
                ] = trajectory_df["lat"][lower_row:upper_row]
                plot_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                    "time"
                ] = trajectory_df["time"][lower_row:upper_row]
                plot_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                    "z_type"
                ] = trajectory_df["z_type"][lower_row]

                row_index += 1
                traj_index += 1

                if traj_index == 5:
                    traj_index = 0
                    alt_index += 1

                if alt_index > altitude_levels:
                    alt_index = 1

                    for domain in domains:
                        assemble_pdf(
                            plot_info_dict=plot_info_dict,
                            x=time_axis,
                            plot_dict=plot_dict,
                            key=key,
                            side_traj=side_traj,
                            output_dir=output_dir,
                            altitude_levels=altitude_levels,
                            language=language,
                            max_start_altitude=max_start_altitude,
                            domain=domain,
                        )

            else:
                plot_dict["altitude_" + str(alt_index)]["origin"] = origin
                plot_dict["altitude_" + str(alt_index)]["subplot_index"] = subplot_index
                plot_dict["altitude_" + str(alt_index)][
                    "max_start_altitude"
                ] = max_start_altitude
                plot_dict["altitude_" + str(alt_index)]["y_type"] = trajectory_df[
                    "z_type"
                ][lower_row]
                plot_dict["altitude_" + str(alt_index)]["alt_level"] = trajectory_df[
                    "start_altitude"
                ][lower_row]

                # traj_0 -> relevant information
                plot_dict["altitude_" + str(alt_index)]["y_surf"] = trajectory_df[
                    "hsurf"
                ][lower_row:upper_row]
                plot_dict["altitude_" + str(alt_index)]["traj_0"]["z"] = trajectory_df[
                    "z"
                ][lower_row:upper_row]
                plot_dict["altitude_" + str(alt_index)]["traj_0"][
                    "lon"
                ] = trajectory_df["lon"][lower_row:upper_row]
                plot_dict["altitude_" + str(alt_index)]["traj_0"][
                    "lat"
                ] = trajectory_df["lat"][lower_row:upper_row]
                plot_dict["altitude_" + str(alt_index)]["traj_0"][
                    "time"
                ] = trajectory_df["time"][lower_row:upper_row]
                # traj_1 -> empty
                plot_dict["altitude_" + str(alt_index)]["traj_1"]["z"] = []
                plot_dict["altitude_" + str(alt_index)]["traj_1"]["lon"] = []
                plot_dict["altitude_" + str(alt_index)]["traj_1"]["lat"] = []
                # traj_2 -> empty
                plot_dict["altitude_" + str(alt_index)]["traj_2"]["z"] = []
                plot_dict["altitude_" + str(alt_index)]["traj_2"]["lon"] = []
                plot_dict["altitude_" + str(alt_index)]["traj_2"]["lat"] = []
                # traj_3 -> empty
                plot_dict["altitude_" + str(alt_index)]["traj_3"]["z"] = []
                plot_dict["altitude_" + str(alt_index)]["traj_3"]["lon"] = []
                plot_dict["altitude_" + str(alt_index)]["traj_3"]["lat"] = []
                # traj_4 -> empty
                plot_dict["altitude_" + str(alt_index)]["traj_4"]["z"] = []
                plot_dict["altitude_" + str(alt_index)]["traj_4"]["lon"] = []
                plot_dict["altitude_" + str(alt_index)]["traj_4"]["lat"] = []
                # further empty keys
                plot_dict["altitude_" + str(alt_index)]["traj_0"][
                    "z_type"
                ] = trajectory_df["z_type"][lower_row]
                plot_dict["altitude_" + str(alt_index)]["traj_1"]["z_type"] = None
                plot_dict["altitude_" + str(alt_index)]["traj_2"]["z_type"] = None
                plot_dict["altitude_" + str(alt_index)]["traj_3"]["z_type"] = None
                plot_dict["altitude_" + str(alt_index)]["traj_4"]["z_type"] = None

                row_index += 1
                alt_index += 1
                if alt_index > altitude_levels:
                    alt_index = 1
                    for domain in domains:
                        assemble_pdf(
                            plot_info_dict=plot_info_dict,
                            x=time_axis,
                            plot_dict=plot_dict,
                            key=key,
                            side_traj=side_traj,
                            output_dir=output_dir,
                            altitude_levels=altitude_levels,
                            language=language,
                            max_start_altitude=max_start_altitude,
                            domain=domain,
                        )
    return
