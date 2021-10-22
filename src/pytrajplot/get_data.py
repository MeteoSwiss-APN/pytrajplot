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


# not sure if needed in the future, thus just commented out
if False:
    # TODO: rewrite functions, s.t. it doesnt alter the start_df dataframe
    def count_unique_origins(data):
        # print(data["origin"])
        i = 0
        for str in data["origin"]:
            # print(str)
            if "~" in str:
                data["origin"].iloc[i] = str[
                    : (len(str) - 2)
                ]  # data["origin"][i] = str[: (len(str) - 2)]
            i += 1
        # print(data["origin"])
        unique_origins = data["origin"].nunique()
        return unique_origins


def read_startf(startf_path):
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

    # start_df_tmp = start_df
    # this function call also alters the origins column in the start_df, which it shouldn't!
    # unique_origins = count_unique_origins(data=start_df_tmp)
    # print(f'number of unique origins: {unique_origins}')

    # start_df.to_csv("start.csv", index=False)

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
    counter = 0
    for row in traj_df["time"]:
        if case == "HRES":
            delta_t = abs(row)
            if ".3" in str(delta_t):
                delta_t = delta_t + 0.2

            date = dt_object - timedelta(hours=delta_t)

        if case == "COSMO":
            delta_t = float(row)
            date = dt_object + timedelta(hours=delta_t)

        traj_df.iat[
            counter, 0
        ] = date  # replace the elapsed time with the corresponding datetime object
        counter += 1
    traj_df.to_csv("src/pytrajplot/plt/traj.csv", index=False)
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
        traj_df["z"] = traj_df["z"].clip(lower=0)  # remove negative values in z column
        traj_df["hsurf"] = traj_df["hsurf"].clip(
            lower=0
        )  # remove negative values in hsurf column
        traj_df.dropna(
            subset=["lon"], inplace=True
        )  # remove rows containing only the origin/z_type

    traj_df["z_type"] = None  # add z_type key to dataframe
    traj_df["origin"] = None  # add origin key to dataframe
    traj_df["#trajectories"] = number_of_trajectories
    traj_df["block_length"] = number_of_times

    # add z_type and origin columns to trajectory dataframe
    if True:
        tmp = 0
        while tmp < number_of_trajectories:
            lower_row = tmp * number_of_times
            upper_row = tmp * number_of_times + number_of_times
            # print(traj_df['time'][lower_row:upper_row]) # check which rows a
            # re being changed
            z_type = start_df["z_type"][tmp]
            origin = start_df["origin"][tmp]
            traj_df["z_type"].iloc[lower_row:upper_row] = z_type
            traj_df["origin"].iloc[lower_row:upper_row] = origin
            tmp += 1

    traj_df = convert_time(plot_info_dict=plot_info_dict, traj_df=traj_df, case=case)

    return traj_df, number_of_trajectories, number_of_times


def check_input_dir(
    dir_path, prefix_dict
):  # iterate through the input folder containing the trajectorie coordinates
    if plot_steps:
        print("--- iterating through input directory")

    counter = 0
    start_dict, trajectory_dict, files, keys, traj_blocks = {}, {}, [], [], {}

    # iterate through the directory, first reading the start & plot_info files (necessary, to parse the trajectory files afterwards)
    for filename in os.listdir(dir_path):

        file_path = os.path.join(dir_path, filename)
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
                    startf_path=file_path
                )

            if filename[: len(prefix_dict["plot_info"])] == prefix_dict["plot_info"]:
                files.append(filename)
                plot_info_dict = read_plot_info(plot_info_path=file_path)

    # iterate through the directory, reading the trajectory and plot_info file after having read the start file
    for filename in os.listdir(dir_path):

        file_path = os.path.join(dir_path, filename)

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

    return plot_info_dict, trajectory_dict, keys, traj_blocks
