"""Generate PDF w/ altitude figure and map plot."""

# Standard library
import cProfile
import io
import pstats
import time
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local
from .scratch.dateline.scratch_dateline import _analyse_trajectories
from .scratch.dateline.scratch_dateline import _check_dateline_crossing
from .scratch.dateline.scratch_dateline import _create_plot
from .scratch.dateline.scratch_dateline import _get_traj_dict


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
            "start_time": None,
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
    start_time = plot_dict["altitude_1"]["start_time"]
    date = start_time.strftime("%Y%m%d")

    # model run time = absolute difference between the two numbers in key
    runtime = abs(int(key[4:7]) - int(key[0:3]))

    trajectory_direction = plot_dict["altitude_1"]["trajectory_direction"]
    final_filename = (
        date
        + f"T{int(start_time.hour):02}"
        + f"_{origin}_"
        + f"LAGRANTO-{plot_info_dict['model_name']}_"
        + f"Trajektorien_"
        + f"{trajectory_direction}_"
        + f"{runtime:03}_"
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
    projection,
    trajectory_expansion,
    cross_dateline,
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
        projection:             cartopy projection  Projection of map
        trajectory_expansion:   list                Expansion of trajectory (formerly called dynamic_boundaries)
        cross_dateline:         bool                True/False if dateline gets crossed

    """
    # generate the output directory if it doesn't exist
    origin = plot_dict["altitude_1"]["origin"]
    output_dir = Path(output_dir)
    outpath = Path(output_dir / "plots" / key)
    outpath.mkdir(parents=True, exist_ok=True)

    # compute the trajectory shift for side trajectories, if there are any
    # traj_index - shift_direction mapping: 0=main, 1=east, 2=north, 3=west, 4=south
    if side_traj:
        # L = π*R*a/180; L - arc length [km]; R - radius of a circle [km]; a - angle [degrees].
        R = 6371  # [km], average radius of the earth
        lat0 = plot_dict["altitude_1"]["traj_0"]["lat"].iloc[
            0
        ]  # latitude of main trajectory origin
        lat1 = plot_dict["altitude_1"]["traj_1"]["lat"].iloc[
            0
        ]  # latitude of side trajectory origin (shifted to the east w.r.t to main trajectory)
        traj_shift_degrees = abs(lat1 - lat0)
        traj_shift = int(np.pi * R * traj_shift_degrees / 180)

    base_time = plot_info_dict["mbt"][0:13] + " " + plot_info_dict["mbt"][-3:]

    fig = plt.figure(tight_layout=False, figsize=(16, 9))

    subfigs = fig.subfigures(2, 1, height_ratios=[0.2, 1])

    subfigsnest = subfigs[1].subfigures(1, 2, width_ratios=[1, 0.4])

    plt.subplots_adjust(
        left=0.08,
        right=0.92,
        bottom=0.1,
        top=0.99,
        wspace=0.5,  # space between the first and second column
        hspace=0.05,
    )

    # add facecolor to subfigures
    if False:
        subfigs[0].set_facecolor("0.75")  # header
        subfigsnest[0].set_facecolor("0.70")  # map
        subfigsnest[1].set_facecolor("0.65")  # altitude

    # ADD ALTITUDE PLOT; compute only once for given origin
    subplot_dict = {}

    if altitude_levels < 3:
        axsnest1 = subfigsnest[1].subplots(3, 1)

        # collect references to subplots
        for nn, ax in enumerate(axsnest1):
            if altitude_levels == 1:
                if nn == 0 or nn == 1:
                    ax.axis("off")  # create a blank plot for subplots 0 and 1
                else:
                    generate_altitude_plot(
                        x=x,
                        y=plot_dict,
                        key=key,
                        side_traj=side_traj,
                        altitude_levels=1,
                        language=language,
                        max_start_altitude=max_start_altitude,
                        ax=ax,
                        alt_index=1,
                        sub_index=0,
                    )

            if altitude_levels == 2:
                if nn == 0:
                    subplot_dict[nn] = ax.axis(
                        "off"
                    )  # create a blank plot for suplot 0
                else:
                    subplot_dict[nn] = ax  # create a regular plot for subplots 1 and 2

        # create 2 altitude plots
        if altitude_levels == 2:
            alt_index = 1
            while alt_index <= 2:
                if int(plot_dict["altitude_" + str(alt_index)]["subplot_index"]) == 0:
                    subplot_index = 1
                    tmp_ax = subplot_dict[subplot_index]
                    generate_altitude_plot(
                        x=x,
                        y=plot_dict,
                        key=key,
                        side_traj=side_traj,
                        altitude_levels=2,
                        language=language,
                        max_start_altitude=max_start_altitude,
                        ax=tmp_ax,
                        alt_index=alt_index,
                        sub_index=0,
                    )

                if int(plot_dict["altitude_" + str(alt_index)]["subplot_index"]) == 1:
                    subplot_index = 2
                    tmp_ax = subplot_dict[subplot_index]
                    generate_altitude_plot(
                        x=x,
                        y=plot_dict,
                        key=key,
                        side_traj=side_traj,
                        altitude_levels=2,
                        language=language,
                        max_start_altitude=max_start_altitude,
                        ax=tmp_ax,
                        alt_index=alt_index,
                        sub_index=1,
                    )

                alt_index += 1

    # if altitude_levels >= 3:
    else:
        axsnest1 = subfigsnest[1].subplots(altitude_levels, 1)

        plt.subplots_adjust(hspace=0.1)

        # collect references to subplots
        for nn, ax in enumerate(axsnest1):
            subplot_dict[nn] = ax

        alt_index = 1
        while alt_index <= altitude_levels:
            subplot_index = int(
                plot_dict["altitude_" + str(alt_index)]["subplot_index"]
            )
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

    # ADD MAP, HEADER & FOOTER; compute for each domain
    for domain in domains:
        # ADD FOOTER
        # start = time.perf_counter()
        if language == "en":
            if side_traj:
                footer = (
                    f"LAGRANTO based on {plot_info_dict['model_name']} {base_time}  |  "
                    + f"Add. traj. @ {traj_shift} km N/E/S/W  |  "
                    + f"MeteoSwiss ©"
                )
            else:
                footer = (
                    f"LAGRANTO based on {plot_info_dict['model_name']} {base_time}  |  "
                    + f"MeteoSwiss ©"
                )

        if language == "de":
            if side_traj:
                footer = (
                    f"LAGRANTO basierend auf {plot_info_dict['model_name']} {base_time}  |  "
                    + f"Zus. traj. @ {traj_shift} km N/O/S/W  |  "
                    + f"MeteoSwiss ©"
                )
            else:
                footer = (
                    f"LAGRANTO basierend auf {plot_info_dict['model_name']} {base_time}  |  "
                    + f"MeteoSwiss ©"
                )

        subfigsnest[0].suptitle(
            footer,
            x=0.08,
            y=0.035,
            horizontalalignment="left",  # 'center', 'left', 'right'; default: center
            verticalalignment="top",  # 'top', 'center', 'bottom', 'baseline'; default: top
            fontdict={
                "size": 6,
                "color": "k",
                # "color": "#5B5B5B",
            },
        )
        # end = time.perf_counter()
        # print(f"Adding footer took:\t\t{end-start} seconds")

        # ADD INFO HEADER & TITLE
        # start = time.perf_counter()
        axTop = subfigs[0].subplots()
        generate_info_header(
            language=language,
            plot_data=plot_dict,
            ax=axTop,
        )
        # end = time.perf_counter()
        # print(f"Adding info header took:\t{end-start} seconds")

        # ADD MAP & FOOTER
        # start = time.perf_counter()
        map_ax = subfigsnest[0].add_subplot(111, projection=projection)
        generate_map_plot(
            cross_dateline=cross_dateline,
            coord_dict=plot_dict,
            side_traj=side_traj,
            altitude_levels=altitude_levels,
            domain=domain,
            trajectory_expansion=trajectory_expansion,
            ax=map_ax,
        )
        # end = time.perf_counter()
        # print(f"Adding map took:\t\t{end-start} seconds")

        # SAVE FIGURE
        filename = generate_filename(plot_info_dict, plot_dict, origin, domain, key)

        for file_type in output_types:
            # print(f"key:{key}, domain:{domain}, origin:{origin}")
            # start = time.perf_counter()
            plt.savefig(str(outpath) + f"/{filename}.{file_type}")
            # end = time.perf_counter()
            # print(f"Saving plot took:\t\t{end-start} seconds")

        # CLEAR HEADER/MAP AXES FOR NEXT ITERATION
        map_ax.remove()
        axTop.remove()

    plt.close(fig)


def get_map_settings(lon, lat, case, number_of_times):
    """Figure out, which map settings (projection, cross dateline,...) should be used and whether the dateline gets crossed.

    Args:
        lon (pandas series): longitude values
        lat (pandas series): latitude values
        case (str]): COSMO/HRES
        number_of_times (int): #lines that make up one trajectory

    """
    if case == "COSMO":
        central_longitude = 0
        cross_dateline = False
        trajectory_expansion = [0, 0, 0, 0]
        projection = ccrs.RotatedPole(
            pole_longitude=-170, pole_latitude=43
        )  # define rotation of COSMO model
        return central_longitude, projection, trajectory_expansion, cross_dateline

    if case == "HRES":
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # implement scratch_dateline.py (only relevant for HRES trajectories though)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # 0) concat the longitude/latitude dataframes into a new dataframe
        data = {
            "lon": lon,
            "lat": lat,
        }

        number_of_trajectories = int(len(lat) / number_of_times)

        lon_lat_df = pd.concat(data, axis=1)

        # 1) check if dateline gets crossed;
        # if the dateline does not get crossed -->  longitude expansion = left & right boundary of trajectories
        cross_dateline, longitude_expansion = _check_dateline_crossing(lon=lon)

        # 2) split lon/lat lists into corresponding trajectories. for each trajectory one key should be assigned.
        (
            traj_dict,
            dateline_crossing_trajectories,
            latitude_expansion,
            eastern_longitudes,
        ) = _get_traj_dict(
            data=lon_lat_df,
            number_of_trajectories=number_of_trajectories,
            traj_length=number_of_times,
        )

        # 3) compute central longitude dynamic domain if dateline does not get crossed
        central_longitude, dynamic_domain = _analyse_trajectories(
            traj_dict=traj_dict,
            cross_dateline=cross_dateline,
            dateline_crossing_trajectories=dateline_crossing_trajectories,
            latitude_expansion=latitude_expansion,
            longitude_expansion=longitude_expansion,
            eastern_longitudes=eastern_longitudes,
        )

        projection = ccrs.PlateCarree(central_longitude=central_longitude)

        # TODO: remove later
        # 5) plot trajectories & save plot (just temporarily to check wheter all information is correct)
        # _create_plot(traj_dict, central_longitude, dynamic_domain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        return central_longitude, projection, dynamic_domain, cross_dateline


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
        # ~~~~~~~~~~~~~~~~~~~~ save dataframe ~~~~~~~~~~~~~~~~~~~~ #
        # trajectory_df.to_csv(f'{output_dir}{key}_df.csv', index = False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        start_time = trajectory_df["datetime"].loc[0]
        altitude_levels = int(trajectory_df["altitude_levels"].loc[0])
        trajectory_direction = str(trajectory_df["trajectory_direction"].loc[0])
        number_of_times = int(trajectory_df["block_length"].iloc[0])
        number_of_trajectories = int(trajectory_df["#trajectories"].iloc[0])

        if "HRES" in plot_info_dict["model_name"]:
            model = "HRES"
        else:
            model = "COSMO"

        # time axis for altitude plots (= x-axis)
        time_axis = trajectory_df["datetime"].iloc[0:number_of_times]

        plot_dict = create_plot_dict(altitude_levels=altitude_levels)

        trajectory_longitude_expansion, trajectory_latitude_expansion = (
            pd.Series(),
            pd.Series(),
        )

        row_index = 0
        alt_index = 1  # altitude 1,2,3,4,...
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
                plot_dict["altitude_" + str(alt_index)]["start_time"] = start_time
                plot_dict["altitude_" + str(alt_index)]["alt_level"] = trajectory_df[
                    "start_altitude"
                ][lower_row]
                plot_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                    "z"
                ] = trajectory_df["z"][lower_row:upper_row]

                # ~~~~~~~~~~~~~~~~~~~~ add lon/lat to plot_dict & trajectory_expansion_df ~~~~~~~~~~~~~~~~~~~~ #
                plot_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                    "lon"
                ] = trajectory_df["lon"][lower_row:upper_row]
                plot_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                    "lat"
                ] = trajectory_df["lat"][lower_row:upper_row]
                trajectory_latitude_expansion = trajectory_latitude_expansion.append(
                    trajectory_df["lat"][lower_row:upper_row], ignore_index=True
                )
                trajectory_longitude_expansion = trajectory_longitude_expansion.append(
                    trajectory_df["lon"][lower_row:upper_row], ignore_index=True
                )
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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
                    # start = time.perf_counter()
                    (
                        central_longitude,
                        projection,
                        trajectory_expansion,
                        cross_dateline,
                    ) = get_map_settings(
                        lon=trajectory_longitude_expansion,
                        lat=trajectory_latitude_expansion,
                        case=model,
                        number_of_times=number_of_times,
                    )
                    # end = time.perf_counter()
                    # print(f"Computing dynamic domain took\t{end-start} seconds.")
                    alt_index = 1

                    # start = time.perf_counter()
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
                        projection=projection,
                        trajectory_expansion=trajectory_expansion,
                        cross_dateline=cross_dateline,
                    )

                    # ~~~~~~~~~~~~~~~~~ check if #altitude levels is variable within one traj-file ~~~~~~~~~~~ #
                    if row_index < (number_of_trajectories - 1):
                        next_number_of_altitudes = trajectory_df["altitude_levels"].loc[
                            (row_index + 1) * number_of_times
                        ]
                        if next_number_of_altitudes is not altitude_levels:
                            altitude_levels = next_number_of_altitudes
                            plot_dict = create_plot_dict(
                                altitude_levels=altitude_levels
                            )
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

                    # reset the traj_expansion pandas series before next iteration
                    trajectory_longitude_expansion, trajectory_latitude_expansion = (
                        pd.Series(),
                        pd.Series(),
                    )

                    # end = time.perf_counter()
                    # print(
                    #     f"Assemble pdf took\t\t{end-start} sec from the whole generate_pdf pipeline."
                    # )

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
                plot_dict["altitude_" + str(alt_index)]["start_time"] = start_time

                # traj_0 -> relevant information
                plot_dict["altitude_" + str(alt_index)]["y_surf"] = trajectory_df[
                    "hsurf"
                ][lower_row:upper_row]
                plot_dict["altitude_" + str(alt_index)]["traj_0"]["z"] = trajectory_df[
                    "z"
                ][lower_row:upper_row]
                plot_dict["altitude_" + str(alt_index)]["traj_0"][
                    "time"
                ] = trajectory_df["time"][lower_row:upper_row]
                # ~~~~~~~~~~~~~~~~~~~~ add lon/lat to plot_dict & trajectory_expansion_df ~~~~~~~~~~~~~~~~~~~~ #
                plot_dict["altitude_" + str(alt_index)]["traj_0"][
                    "lon"
                ] = trajectory_df["lon"][lower_row:upper_row]
                plot_dict["altitude_" + str(alt_index)]["traj_0"][
                    "lat"
                ] = trajectory_df["lat"][lower_row:upper_row]
                trajectory_latitude_expansion = trajectory_latitude_expansion.append(
                    trajectory_df["lat"][lower_row:upper_row], ignore_index=True
                )
                trajectory_longitude_expansion = trajectory_longitude_expansion.append(
                    trajectory_df["lon"][lower_row:upper_row], ignore_index=True
                )
                # ~~~~~~~~~~~~~~~~~~~~ add lon/lat to plot_dict & trajectory_expansion_df ~~~~~~~~~~~~~~~~~~~~ #

                # keys for trajectories 1-4 remain empty
                plot_dict["altitude_" + str(alt_index)]["traj_1"]["z"] = []
                plot_dict["altitude_" + str(alt_index)]["traj_1"]["lon"] = []
                plot_dict["altitude_" + str(alt_index)]["traj_1"]["lat"] = []
                plot_dict["altitude_" + str(alt_index)]["traj_2"]["z"] = []
                plot_dict["altitude_" + str(alt_index)]["traj_2"]["lon"] = []
                plot_dict["altitude_" + str(alt_index)]["traj_2"]["lat"] = []
                plot_dict["altitude_" + str(alt_index)]["traj_3"]["z"] = []
                plot_dict["altitude_" + str(alt_index)]["traj_3"]["lon"] = []
                plot_dict["altitude_" + str(alt_index)]["traj_3"]["lat"] = []
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
                    # start = time.perf_counter()
                    (
                        central_longitude,
                        projection,
                        trajectory_expansion,
                        cross_dateline,
                    ) = get_map_settings(
                        lon=trajectory_longitude_expansion,
                        lat=trajectory_latitude_expansion,
                        case=model,
                        number_of_times=number_of_times,
                    )
                    # end = time.perf_counter()
                    # print(f"Computing dynamic domain took\t{end-start} seconds.")
                    alt_index = 1

                    # start = time.perf_counter()
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
                        projection=projection,
                        trajectory_expansion=trajectory_expansion,
                        cross_dateline=cross_dateline,
                    )
                    # reset the traj_expansion pandas series before next iteration
                    trajectory_longitude_expansion, trajectory_latitude_expansion = (
                        pd.Series(),
                        pd.Series(),
                    )

                    # end = time.perf_counter()
                    # print(
                    #     f"Assemble pdf took\t\t{end-start} sec from the whole generate_pdf pipeline."
                    # )
    return
