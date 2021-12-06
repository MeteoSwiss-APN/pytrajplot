"""Generate PDF w/ altitude figure and map plot."""

# Standard library
import cProfile
import io
import os
import pstats

# Third-party
import cartopy.crs as ccrs
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt


def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return wrapper


plt.rc("axes", labelsize=8)  # change the font size of the axis labels


# Local
from .plotting.plot_altitude import generate_altitude_plot
from .plotting.plot_info_header import generate_info_header
from .plotting.plot_map import generate_map_plot
from .plotting.plot_map import get_dynamic_domain


def create_plot_dict(altitude_levels):
    """Create dict of dicts to be filled with information for all plots.

    Args:
        altitude_levels:        int         #altitdue levels = #keys per dict

    Returns:
        plot_dict.              dict        Dict containing for each altitude level all relevent information

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
            "trajectory_direction": None,
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
    """Define the projection (PlateCarree for HRES or RotatedPole for COSMO).

    Args:
    plot_dict:              dict                        Dictionary containing the combined information of the start & trajectory file
    altitude_levels:        int                         Number of starting altitudes
    side_traj:              bool                        Bool, to specify whether there are side trajectories or not

    Returns:
        projection:         cartopy.crs.<projection>    Projection of the cartopy GeoAxes
        cross_dateline:     bool                        Bool to specify whether the dateline gets crossed or not

    """
    # check if HRES or COSMO
    if plot_dict["altitude_1"]["y_type"] == "hpa":
        # case = "HRES"
        # check if dateline is crossed or not
        central_longitude, _, _, cross_dateline = get_dynamic_domain(
            coord_dict=plot_dict, altitude_levels=altitude_levels, side_traj=side_traj
        )
        projection = ccrs.PlateCarree(central_longitude=central_longitude)
    else:
        # case = "COSMO"
        cross_dateline = False
        projection = ccrs.RotatedPole(
            pole_longitude=-170, pole_latitude=43
        )  # define rotation of COSMO model

    return projection, cross_dateline


def generate_filename(plot_info_dict, plot_dict, origin, domain, key):
    """Generate the filename of the output file.

    Args:
        plot_info_dict:     dict        Dictionary containing the information of the plot-info file
        plot_dict:          dict        Dictionary containing the combined information of the start & trajectory file
        origin:             str         Origin where the trajectory started
        domain:             str         Domain of the map.
        key:                str         ID of the start and trajectory file

    Returns:
        final_filename:    str          Name of output file

    """
    date = (
        plot_info_dict["mbt"][0:4]
        + plot_info_dict["mbt"][5:7]
        + plot_info_dict["mbt"][8:10]
    )
    trajectory_direction = plot_dict["altitude_1"]["trajectory_direction"]
    final_filename = (
        date
        + f"T{key[0:3]}"
        + f"_{origin}_"
        + f"LAGRANTO-{plot_info_dict['model_name']}_"
        + f"Trajektorien_"
        + f"{trajectory_direction}_"
        + f"{key[4:7]}_"
        + f"{domain}"
    )
    return final_filename


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
    output_types,
):
    """Assemble final output pdf/png.

    Args:
        plot_info_dict:         dict                Dictionary containing the information of the plot-info file
        x:                      pandas series       Pandas Series (array-like) w/ the dates. x-axis information for the altitude plot
        plot_dict:              dict                Dictionary containing the combined information of the start & trajectory file
        key:                    str                 ID of the start and trajectory file
        side_traj:              bool                Bool, to specify whether there are side trajectories or not
        output_dir:             str                 Path to output directory
        altitude_levels:        int                 Number of starting altitudes
        language:               str                 Language of plot annotations
        max_start_altitude:     float               Highest starting altitude
        domain:                 str                 Domain of the map.
        output_types:           tuple               Tuple containing the file types of the output files. (pdf and/or png)

    """
    # DEFINE FIGURE PROPERTIES AND GRID SPECIFICATION
    # fig = plt.figure(figsize=(11.69, 8.27), dpi=200)  # A4 size
    fig = plt.figure(figsize=(16, 9), dpi=200)

    # create grid spec oject
    grid_specification = gs.GridSpec(
        nrows=2, ncols=2, width_ratios=[1, 0.4], height_ratios=[0.1, 1]
    )

    # optimise placement of sub gridspec objects
    plt.subplots_adjust(
        left=0.1,  # left margin = 0.1
        bottom=0.08,
        right=0.9,  # right margin = 0.1
        top=0.92,
        wspace=0.15,  # space between the first and second column
        hspace=0.08,
    )

    # ADD INFO HEADER TO PDF
    gs_info = grid_specification[0, 0].subgridspec(
        1, 3, width_ratios=[1.5, 1, 0.5]
    )  # placement of info header optimal
    info_ax = plt.subplot(gs_info[:, 1:])
    generate_info_header(
        language=language,
        plot_info=plot_info_dict,
        plot_data=plot_dict,
        ax=info_ax,
        domain=domain,
    )

    # ADD MAP TO PDF
    gs_map = grid_specification[1, 0].subgridspec(
        10, 1
    )  # size of map, as of now not optimal in my opinion
    projection, cross_dateline = get_projection(
        plot_dict=plot_dict, altitude_levels=altitude_levels, side_traj=side_traj
    )
    map_ax = plt.subplot(gs_map[:, 0], projection=projection)
    generate_map_plot(
        cross_dateline=cross_dateline,
        coord_dict=plot_dict,
        side_traj=side_traj,
        altitude_levels=altitude_levels,
        domain=domain,
        ax=map_ax,
    )

    # ADD ALTITUDE PLOT TO PDF
    if altitude_levels <= 3:
        gs_alt = grid_specification[1, 1].subgridspec(
            32, 1
        )  # use this gs_alt only if altitude levels > 3, for 1-3 write separate functions
        alt_index = 1
        while alt_index <= altitude_levels:
            subplot_index = int(
                plot_dict["altitude_" + str(alt_index)]["subplot_index"]
            )

            if altitude_levels == 1:
                tmp_ax = plt.subplot(gs_alt[22:, 0])
            if altitude_levels == 2:
                if subplot_index == 0:
                    tmp_ax = plt.subplot(gs_alt[11:21, 0])
                if subplot_index == 1:
                    tmp_ax = plt.subplot(gs_alt[22:, 0])
            if altitude_levels == 3:
                if subplot_index == 0:
                    tmp_ax = plt.subplot(gs_alt[0:10, 0])
                if subplot_index == 1:
                    tmp_ax = plt.subplot(gs_alt[11:21, 0])
                if subplot_index == 2:
                    tmp_ax = plt.subplot(gs_alt[22:, 0])

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
            alt_index += 1

    if altitude_levels > 3:
        gs_alt = grid_specification[1, 1].subgridspec(
            altitude_levels, 1
        )  # use this gs_alt only if altitude levels > 3, for 1-3 write separate functions
        alt_index = 1
        while alt_index <= altitude_levels:
            subplot_index = int(
                plot_dict["altitude_" + str(alt_index)]["subplot_index"]
            )
            tmp_ax = plt.subplot(gs_alt[subplot_index, 0])
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
            alt_index += 1

    # SAVE PDF
    origin = plot_dict["altitude_1"]["origin"]
    outpath = os.getcwd() + "/" + output_dir + "/plots/" + key + "/"
    os.makedirs(
        outpath, exist_ok=True
    )  # create plot folder if it doesn't already exist

    filename = generate_filename(plot_info_dict, plot_dict, origin, domain, key)
    for file_type in output_types:
        # plt.savefig(outpath + "new." + file_type)
        plt.savefig(outpath + filename + "." + file_type)
    plt.close(fig)
    return


# @profile
def generate_pdf(
    trajectory_dict,
    plot_info_dict,
    output_dir,
    separator,
    language,
    domains,
    output_types,
):
    """Iterate through trajectory dict. For each key, generate altitude plots for all origins/altitude levels.

    Args:
        trajectory_dict:        dict                Dictionary containig for each key one dataframe w/ all information
        plot_info_dict:         dict                Dictionary containing the information of the plot info file
        output_dir:             str                 Path to output directory (where the subdirectories containing all plots should be saved)
        separator:              str                 Separator string to identify main- and side-trajectories (default: '~')
        language:               str                 Language for plot annotations.
        domains:                list                List of domains, for which PDFs need to be created.
        output_types:           tuple               Tuple containing the file types of the output files. (pdf and/or png)

    """
    print("--- Assembling PDF")
    for key in trajectory_dict:  # iterate through the trajectory dict
        # print(key)
        trajectory_df = trajectory_dict[key]  # extract df for given key
        altitude_levels = trajectory_df["altitude_levels"].loc[0]
        trajectory_direction = trajectory_df["trajectory_direction"].loc[0]
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
                plot_dict["altitude_" + str(alt_index)][
                    "trajectory_direction"
                ] = trajectory_direction
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
                            output_types=output_types,
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
                plot_dict["altitude_" + str(alt_index)][
                    "trajectory_direction"
                ] = trajectory_direction

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
                            output_types=output_types,
                        )
    return
