"""Util functions to get data."""
# Standard library
import os

# Third-party
import pandas as pd

# import pdb  # use: python debugger, i.e. pdb.set_trace()


plot_steps = True
global_path = os.getcwd() + "/zmichel/21101100_407/lagranto_c/"


# REMARK: the plot_info can be differentiated from the other files, because the first line is empty (or just simply by its name of course)
def read_plot_info(plot_info_path):
    """Read the pure txt file containing the plot_info to variables for later purposes."""
    if plot_steps:
        print("--- reading plot_info into dict")

    # TODO: make path to plot_info file a CLI (command line input), furthermore check
    # if the file exists, otherwise print error message and abort
    # plot_info_path = global_path + "plot_info"

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


def count_unique_origins(data):
    # print(data["origin"])
    i = 0
    for str in data["origin"]:
        # print(str)
        if "~" in str:
            data["origin"][i] = str[: (len(str) - 2)]
        i += 1
    # print(data["origin"])
    return data["origin"].nunique()


def read_startf(startf_path):
    if plot_steps:
        print("--- reading startf file")

    # TODO: make path to plot_info file a CLI (command line input), furthermore check
    # if the file exists, otherwise print error message and abort
    # startf_file = global_path + 'startf_000-033F' #"startf_003-033F"

    start_df = pd.read_csv(
        startf_path,
        skiprows=0,
        skipfooter=0,
        sep=" ",
        header=None,
        names=["lon", "lat", "z", "z_type", "origin"],
        engine="python",
        # parse_dates=["termin"],
        skipinitialspace=True,
    )

    unique_origins = count_unique_origins(data=start_df)
    # print(unique_origins)
    return start_df


def read_tra_files():
    if plot_steps:
        print("--- reading trajectory file")

    trajectory_file_path = global_path + "tra_geom_000-033F"

    info = pd.read_csv(
        trajectory_file_path,
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

    print(
        f"number of trajecotries = {number_of_trajectories} \nnumber of times = {number_of_times}"
    )

    header_line = pd.read_csv(
        trajectory_file_path,
        skiprows=19,
        skipfooter=0,
        nrows=1,
        sep=" ",
        header=None,
        engine="python",
        skipinitialspace=True,
    )
    print(header_line)

    traj_df = pd.read_csv(
        trajectory_file_path,
        skiprows=23,
        skipfooter=0,
        sep=" ",
        names=header_line.values.astype(str)[0, :],
        engine="python",
        skipinitialspace=True,
    )
    print(traj_df.head())
    traj_df.to_csv("trajectory_data.csv", index=False)


def check_input_dir():  # iterate through the input folder containing the trajectorie coordinates
    if plot_steps:
        print("--- iterating through input directory")

    plot_info_dict = {}
    frames = []
    # start_df = {}

    for filename in os.listdir(global_path):
        f = os.path.join(global_path, filename)  # add filename to global_path
        if os.path.isfile(f):  # check if it's a file
            # print(f)
            with open(f, "r") as file:
                first_line = file.readline()
                print(
                    f"First line of file {filename}: {first_line} ({len(first_line)})"
                )

                if len(first_line) == 1:
                    plot_info_dict = read_plot_info(f)

                # if first_line[:8] == 'LAGRANTO':
                #     # TODO: parse the trajectory files
                # else:
                #     start_df = read_startf(f)
                #     print('start_df looks like: ', start_df)
                #     frames.append(start_df)
                #     plot_info_dict = pd.concat(frames, keys=['long', 'lat', 'z', 'z_type', 'origin'])

    return plot_info_dict
