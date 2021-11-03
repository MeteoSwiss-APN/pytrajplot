"""Util functions to get data."""
# Standard library
import os
from enum import unique

# Third-party
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
# Standard library
import datetime
from datetime import timedelta

# import pdb  # use: python debugger, i.e. pdb.set_trace()

plot_steps = True


def read_plot_info(plot_info_path):
    """Read the pure txt file containing the plot_info to variables for later purposes."""
    if plot_steps:
        print("--- reading plot_info into dict")

    with open(plot_info_path, "r") as f:
        plot_info_tmp = [line.strip() for line in f.readlines()]
    plot_info = list(
        filter(str, plot_info_tmp)
    )  # remove first and last empty line entries
    plot_info_dict = {
        "mbt": plot_info[0][44:],  # model base time
        "model_name": plot_info[1][44:],  # model name
        "np_long": float(plot_info[2][44:]),  # north pole longitude
        "np_lat": float(plot_info[3][44:]),  # north pole latitude
        "start_long": float(plot_info[4][44:]),  # start longitude
        "start_lat": float(plot_info[5][44:]),  # start latitude
        "inc_long": float(plot_info[6][44:]),  # increment in longitudinal direction
        "inc_lat": float(plot_info[7][44:]),  # increment in latitudinal direction
        "no_pts_long": float(plot_info[8][44:]),  # #points in long. direction
        "no_pts_lat": float(plot_info[9][44:]),  # #point in lat. direction
    }

    return plot_info_dict


def map_altitudes_and_subplots(unit, unique_start_altitudes, current_altitude):
    altitude_levels = len(unique_start_altitudes)
    # map altitudes to subplots. i.e. w/ 4 start altitudes > alt1:sp3, alt2:sp2, alt3:sp1, alt4:sp0
    altitude_mapping_dict = {}
    alt_levels_tmp = (
        altitude_levels - 1
    )  # -1, because the subplots index start at 0 and the altitude index starts at 1
    alt_dict = {}
    tmp_01 = 1  # first start altitude index

    while tmp_01 <= altitude_levels:
        alt_dict[tmp_01] = alt_levels_tmp
        tmp_01 += 1
        alt_levels_tmp -= 1

    if unit == "m":  # regular y-axis
        unique_start_altitudes_sorted = list(np.sort(unique_start_altitudes))
    if unit == "hPa":  # inverted y-axis
        unique_start_altitudes_sorted = list(np.flip(np.sort(unique_start_altitudes)))

    tmp_02 = 0
    while tmp_02 < altitude_levels:
        altitude_index = (
            unique_start_altitudes_sorted.index(unique_start_altitudes[tmp_02]) + 1
        )
        subplot_index = alt_dict[altitude_index]
        # print(f"altitude: {unique_start_altitudes[tmp_02]} {unit} --> altitude_{altitude_index} --> subplot_{subplot_index}")
        altitude_mapping_dict[unique_start_altitudes[tmp_02]] = subplot_index

        tmp_02 += 1
    return altitude_mapping_dict[current_altitude]


def read_startf(startf_path, separator):
    if plot_steps:
        print("--- reading startf file")

    start_df = pd.read_csv(
        startf_path,
        skiprows=0,
        skipfooter=0,
        sep=" ",
        header=None,
        names=["lon", "lat", "z", "z_type", "origin"],
        engine="python",
        skipinitialspace=True,
    )

    alt_type = start_df["z_type"].iloc[0]
    if alt_type == "hpa":
        unit = "hPa"
    else:
        unit = "m"

    start_df["side_traj"] = None  # 1 if there are side trajectories, else 0
    start_df[
        "altitude_levels"
    ] = None  # #altitude levels for certain origin; variable within start file
    start_df["subplot_index"] = None

    i = 0
    unique_origins_list = []
    unique_altitude_levels_dict = {}

    # if there is only one altitude, plot this one twice (to ensure, no error occurs; TEMPORARY SOLUTION)
    if len(start_df) == 1:
        unique_origins_list.append(start_df["origin"].loc[0])
        start_df["side_traj"].loc[i : i + 3] = 0
        start_df["altitude_levels"].loc[i : i + 3] = start_df.loc[
            start_df.origin == start_df["origin"].loc[i], "origin"
        ].count()

    else:
        while i < len(
            start_df
        ):  # len(start_df) = number of rows (=#traj.) in start file
            if i < len(start_df) - 1:
                if (
                    separator in start_df["origin"].loc[i + 1]
                ):  # if Ǝ separator --> have side trajectories for given origin
                    start_df["side_traj"].loc[
                        i : i + 4
                    ] = 1  # corresponding (5) rows have a 1 (=True) in the side_traj column
                    start_df["altitude_levels"].loc[
                        i : i + 4
                    ] = start_df.loc[  # corresponding (5) rows have #alt_levels in the altitude_levels column
                        start_df.origin == start_df["origin"].loc[i], "origin"
                    ].count()

                    if start_df["origin"].loc[i] not in unique_origins_list:
                        unique_origins_list.append(start_df["origin"].loc[i])
                        # NEW FROM HERE
                        origin = start_df["origin"].loc[i]
                        altitude_levels = start_df["altitude_levels"].loc[
                            i
                        ]  # altitude levels for given location
                        altitude = start_df["z"].loc[i]
                        # add information to unique_altitude_levels_dict
                        # print(f'{origin}: rows {i} - {i + 5*altitude_levels} correspond to altitude level {altitude} {unit}')
                        unique_altitude_levels_dict[start_df["origin"].loc[i]] = (
                            start_df["z"].loc[i : (i + 5 * altitude_levels)].unique()
                        )

                    # print(f'{origin}: rows {i} - {i + 5} correspond to altitude level {start_df["z"].loc[i]} {unit}')

                    start_df["subplot_index"].loc[
                        i : i + 5
                    ] = map_altitudes_and_subplots(
                        unit=unit,
                        unique_start_altitudes=unique_altitude_levels_dict[origin],
                        current_altitude=start_df["z"].loc[i],
                    )
                    i += 5

                else:
                    start_df["side_traj"].loc[i : i + 3] = 0

                    start_df["altitude_levels"].loc[i : i + 3] = start_df.loc[
                        start_df.origin == start_df["origin"].loc[i], "origin"
                    ].count()

                    if start_df["origin"].loc[i] not in unique_origins_list:
                        unique_origins_list.append(start_df["origin"].loc[i])

                        # NEW FROM HERE
                        origin = start_df["origin"].loc[i]
                        altitude_levels = start_df["altitude_levels"].loc[
                            i
                        ]  # altitude levels for given location
                        # add information to unique_altitude_levels_dict
                        print(f"{origin}: rows {i} - {i + altitude_levels}")
                        unique_altitude_levels_dict[start_df["origin"].loc[i]] = (
                            start_df["z"].loc[i : (i + altitude_levels)].unique()
                        )

                    # print(f'{origin}: row {i} corresponds to altitude level {start_df["z"].loc[i]} {unit}')

                    tmp = i
                    while tmp < i + 4:
                        start_df["subplot_index"].loc[tmp] = map_altitudes_and_subplots(
                            unit=unit,
                            unique_start_altitudes=unique_altitude_levels_dict[origin],
                            current_altitude=start_df["z"].loc[tmp],
                        )
                        tmp += 1

                    i += 4

    return start_df


def traj_helper_fct(case, file_path, firstline, start_df):
    if case == "COSMO":
        # extract useful information from header
        info = pd.read_csv(
            file_path,
            skiprows=16,
            skipfooter=0,
            nrows=2,
            sep=":",
            header=None,
            engine="python",
            skipinitialspace=True,
        )
        number_of_trajectories = info[1][0]
        number_of_times = info[1][1]

    if case == "HRES":
        number_of_trajectories = len(start_df)
        T = int(firstline[43:49]) / 60
        f = open(file_path)
        lines_to_read = [
            6
        ]  # read the line, after the first dt (see what the timestep was)

        for position, line in enumerate(f):
            if position in lines_to_read:
                dt = float(line[3:7])
                if (
                    dt == 0.3
                ):  # CAUTION: the decimal does neither correspond to #minutes, nor to the fraction of one hour...
                    dt = 0.5

        # number_of_times = 1/dt * T + 1
        number_of_times = int(T / dt + 1)

    if False:
        print(
            f"#Trajectories = {number_of_trajectories} \n#Points/Trajectory = {number_of_times}"
        )

    return number_of_trajectories, number_of_times


def convert_time(plot_info_dict, traj_df, case):
    init_time = plot_info_dict["mbt"][:16]
    format = "%Y-%m-%d %H:%M"
    dt_object = datetime.datetime.strptime(init_time, format)
    if case == "HRES":
        counter = 0
    else:
        counter = 2

    traj_df["datetime"] = None
    for row in traj_df["time"]:
        if case == "HRES":
            delta_t = abs(row)
            if ".3" in str(delta_t):
                delta_t = delta_t + 0.2
            if row < 0:
                date = dt_object - timedelta(hours=delta_t)
            else:
                date = dt_object + timedelta(hours=delta_t)

        if case == "COSMO":
            delta_t = float(row)
            date = dt_object + timedelta(hours=delta_t)

        traj_df["datetime"].loc[counter] = date

        counter += 1

    return traj_df


def read_trajectory(trajectory_file_path, start_df, plot_info_dict):
    if plot_steps:
        print("--- reading trajectory file")

    # read first line of trajectory file to check which case it is.
    with open(trajectory_file_path) as f:
        firstline = f.readline().rstrip()

    if firstline[:8] == "LAGRANTO":  # case: COSMO trajectory file
        case = "COSMO"
        skiprows = 21
        number_of_trajectories, number_of_times = traj_helper_fct(
            case=case,
            file_path=trajectory_file_path,
            firstline=firstline,
            start_df=start_df,
        )

    if firstline[:9] == "Reference":  # case: HRES trajectory file
        case = "HRES"
        skiprows = 5
        number_of_trajectories, number_of_times = traj_helper_fct(
            case=case,
            file_path=trajectory_file_path,
            firstline=firstline,
            start_df=start_df,
        )

    traj_df = pd.read_csv(
        trajectory_file_path,
        skiprows=skiprows,
        skipfooter=0,
        sep=" ",
        names=[
            "time",
            "lon",
            "lat",
            "z",
            "hsurf",
        ],  # to make the header uniform for both types of tra_geom files
        engine="python",
        skipinitialspace=True,
    )

    if (
        case == "COSMO"
    ):  # clean up the df in case its generated from a COSMO trajectory file
        traj_df.loc[(traj_df["z"] < 0), "z"] = np.NaN
        traj_df.loc[(traj_df["hsurf"] < 0), "hsurf"] = np.NaN
        # traj_df["z"] = traj_df["z"].clip(lower=0)  # remove negative values in z column
        # traj_df["hsurf"] = traj_df["hsurf"].clip(lower=0)  # remove negative values in hsurf column
        traj_df.dropna(
            subset=["lon"], inplace=True
        )  # remove rows containing only the origin/z_type

    if (
        case == "HRES"
    ):  # clean up the df in case its generated from a HRES trajectory file

        traj_df.loc[(traj_df["lon"] == -999.00), "lon"] = np.NaN
        traj_df.loc[(traj_df["lat"] == -999.00), "lat"] = np.NaN
        traj_df.loc[(traj_df["z"] == -999), "z"] = np.NaN

    traj_df["z_type"] = None  # add z_type key to dataframe
    traj_df["origin"] = None  # add origin key to dataframe
    traj_df["side_traj"] = None  # add side trajectory key dataframe
    traj_df["start_altitude"] = np.NaN
    traj_df["altitude_levels"] = None
    traj_df["#trajectories"] = number_of_trajectories
    traj_df["block_length"] = number_of_times
    traj_df["subplot_index"] = np.NaN

    # add z_type, origin, side_traj (bool) and alt_levels columns to trajectory dataframe
    tmp = 0
    while tmp < number_of_trajectories:
        lower_row = tmp * number_of_times
        upper_row = tmp * number_of_times + number_of_times
        # print(traj_df['time'][lower_row:upper_row]) # check which rows are being changed
        z_type = start_df["z_type"][tmp]
        origin = start_df["origin"][tmp]
        start_altitude = start_df["z"][tmp]
        side_traj = start_df["side_traj"][tmp]
        altitude_levels = start_df["altitude_levels"][tmp]
        subplot_index = int(start_df["subplot_index"][tmp])
        traj_df["z_type"].iloc[lower_row:upper_row] = z_type
        traj_df["origin"].iloc[lower_row:upper_row] = origin
        traj_df["side_traj"].iloc[lower_row:upper_row] = side_traj
        traj_df["altitude_levels"].iloc[lower_row:upper_row] = altitude_levels
        traj_df["start_altitude"].iloc[lower_row:upper_row] = start_altitude
        traj_df["subplot_index"].iloc[lower_row:upper_row] = subplot_index
        tmp += 1

    traj_df = convert_time(plot_info_dict=plot_info_dict, traj_df=traj_df, case=case)
    traj_df.reset_index(drop=True, inplace=True)

    if True:
        traj_df.to_csv("trajdf.csv", index=True)
        start_df.to_csv("startf.csv", index=True)

    return traj_df, number_of_trajectories, number_of_times


def check_input_dir(
    input_dir, prefix_dict, separator
):  # iterate through the input folder containing the trajectory files
    if plot_steps:
        print("--- iterating through input directory")

    counter = 0
    start_dict, trajectory_dict, files, keys, traj_blocks = {}, {}, [], [], {}

    # iterate through the directory, first reading the start & plot_info files (necessary, to parse the trajectory files afterwards)
    for filename in os.listdir(input_dir):

        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
            if filename[: len(prefix_dict["start"])] == prefix_dict["start"]:
                files.append(filename)

                if filename[len(prefix_dict["start"]) :] not in keys:
                    keys.append(
                        filename[len(prefix_dict["start"]) :]
                    )  # append keys of trajectory_df to a list for later use

                    traj_blocks[filename[len(prefix_dict["start"]) :]] = {
                        "number_of_times": 0,
                        "number_of_trajectories": 0,
                    }  # create for each start/traj pair a dict, containing the length of a trajectory block

                counter += 1
                start_dict[filename[len(prefix_dict["start"]) :]] = read_startf(
                    startf_path=file_path, separator=separator
                )

            if filename[: len(prefix_dict["plot_info"])] == prefix_dict["plot_info"]:
                files.append(filename)
                plot_info_dict = read_plot_info(plot_info_path=file_path)

    # iterate through the directory, reading the trajectory and plot_info file after having read the start file
    for filename in os.listdir(input_dir):

        file_path = os.path.join(input_dir, filename)

        if os.path.isfile(file_path):
            if filename[: len(prefix_dict["trajectory"])] == prefix_dict["trajectory"]:
                files.append(filename)

                # if filename[len(prefix_dict['trajectory']):] not in keys:
                #     keys.append(filename[len(prefix_dict['trajectory']):])

                (
                    trajectory_dict[filename[len(prefix_dict["trajectory"]) :]],
                    number_of_trajectories,
                    number_of_times,
                ) = read_trajectory(
                    trajectory_file_path=file_path,
                    start_df=start_dict[filename[len(prefix_dict["trajectory"]) :]],
                    plot_info_dict=plot_info_dict,
                )

                traj_blocks[filename[len(prefix_dict["trajectory"]) :]][
                    "number_of_times"
                ] = number_of_times
                traj_blocks[filename[len(prefix_dict["trajectory"]) :]][
                    "number_of_trajectories"
                ] = number_of_trajectories

    if False:
        print("Plot info dict:\n", plot_info_dict)
        print("Trajectory dict:\n", trajectory_dict)
        print("Keys:\n", keys)

    return trajectory_dict, plot_info_dict, keys
