"""Generate PDF w/ altitude figure and map plot."""

# Standard library
import cProfile
import io
import os
import pstats
import time
from pathlib import Path

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
            "lon_precise": None,
            "lat_precise": None,
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
    domains,
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
        domains:                str                 Domain of the map.
        output_types:           tuple               Tuple containing the file types of the output files. (pdf and/or png)

    """
    # generate the output directory if it doesn't exist
    origin = plot_dict["altitude_1"]["origin"]
    output_dir = Path(output_dir)
    outpath = Path(output_dir / "plots" / key)
    outpath.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(tight_layout=False, figsize=(16, 9))

    subfigs = fig.subfigures(2, 1, height_ratios=[0.2, 1])

    # add watermark / copyright disclaimer
    subfigs[0].suptitle(
        f"MeteoSwiss 2021 ©",
        fontdict={
            "fontsize": 10,
            "color": "#eeeeee",
        },
    )

    subfigsnest = subfigs[1].subfigures(1, 2, width_ratios=[1, 0.4])

    plt.subplots_adjust(
        left=0.08,
        right=0.92,
        bottom=0.1,
        top=0.99,
        wspace=0.5,  # space between the first and second column
        hspace=0.05,
    )

    # ADD ALTITUDE PLOT
    # TODO: reconsider the case for 1-3 altitude subplots!
    # TODO: reconsider the aspect ratio for the various domains!
    axsnest1 = subfigsnest[1].subplots(altitude_levels, 1)

    if False:
        subfigs[0].set_facecolor("0.75")  # header
        subfigsnest[0].set_facecolor("0.70")  # map
        subfigsnest[1].set_facecolor("0.65")  # altitude

    # collect references to subplots
    subplot_dict = {}
    for nn, ax in enumerate(axsnest1):
        subplot_dict[nn] = ax

    alt_index = 1
    while alt_index <= altitude_levels:
        subplot_index = int(plot_dict["altitude_" + str(alt_index)]["subplot_index"])
        tmp_ax = subplot_dict[subplot_index]
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

    for domain in domains:
        # ADD INFO HEADER
        axTop = subfigs[0].subplots()
        generate_info_header(
            language=language,
            plot_info=plot_info_dict,
            plot_data=plot_dict,
            domain=domain,
            ax=axTop,
        )

        # ADD MAP
        projection, cross_dateline = get_projection(
            plot_dict=plot_dict, altitude_levels=altitude_levels, side_traj=side_traj
        )
        map_ax = subfigsnest[0].add_subplot(111, projection=projection)
        generate_map_plot(
            cross_dateline=cross_dateline,
            coord_dict=plot_dict,
            side_traj=side_traj,
            altitude_levels=altitude_levels,
            domain=domain,
            ax=map_ax,
        )

        # SAVE FIGURE
        filename = generate_filename(plot_info_dict, plot_dict, origin, domain, key)
        for file_type in output_types:
            # plt.savefig(outpath + "new." + file_type)
            plt.savefig(str(outpath) + f"/{filename}.{file_type}")

        # CLEAR HEADER/MAP AXES FOR NEXT ITERATION
        map_ax.remove()
        axTop.remove()

    plt.close(fig)


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
        altitude_levels = int(trajectory_df["altitude_levels"].loc[0])
        trajectory_direction = str(trajectory_df["trajectory_direction"].loc[0])
        number_of_times = int(trajectory_df["block_length"].iloc[0])
        number_of_trajectories = int(trajectory_df["#trajectories"].iloc[0])

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
            lon_precise = trajectory_df["lon_precise"].loc[lower_row]
            lat_precise = trajectory_df["lat_precise"].loc[lower_row]
            altitude_levels = trajectory_df["altitude_levels"].loc[lower_row]
            subplot_index = trajectory_df["subplot_index"].loc[lower_row]
            max_start_altitude = trajectory_df["max_start_altitude"].loc[lower_row]
            side_traj = trajectory_df["side_traj"].loc[lower_row]
            if side_traj:
                if separator not in origin:
                    plot_dict["altitude_" + str(alt_index)]["origin"] = origin
                    plot_dict["altitude_" + str(alt_index)]["lon_precise"] = lon_precise
                    plot_dict["altitude_" + str(alt_index)]["lat_precise"] = lat_precise
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
                    start_tmp = time.perf_counter()
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
                        domains=domains,
                        output_types=output_types,
                    )
                    end_tmp = time.perf_counter()
                    print(
                        f"Assemble pdf took {end_tmp-start_tmp} sec from the whole generate_pdf pipeline."
                    )

            else:
                plot_dict["altitude_" + str(alt_index)]["origin"] = origin
                plot_dict["altitude_" + str(alt_index)]["lon_precise"] = lon_precise
                plot_dict["altitude_" + str(alt_index)]["lat_precise"] = lat_precise
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
                        domains=domains,
                        output_types=output_types,
                    )
    return
