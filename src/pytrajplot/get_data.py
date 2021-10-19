"""Util functions to get data."""
# Standard library
import os

# Third-party
import pandas as pd

# import pdb  # use: python debugger, i.e. pdb.set_trace()


plot_steps = True
global_path = os.getcwd() + "/zmichel/21101100_407/lagranto_c/"

# REMARK: the plot_info can be differentiated from the other files, because the first line is empty (or just simply by its name of course)
def read_plot_info():
    """Read the pure txt file containing the plot_info to variables for later purposes."""
    if plot_steps:
        print("--- reading plot_info into dict")

    # TODO: make path to plot_info file a CLI (command line input), furthermore check
    # if the file exists, otherwise print error message and abort
    plot_info_path = global_path + "plot_info"

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


def read_startf():
    if plot_steps:
        print("--- reading startf file")

    # TODO: make path to plot_info file a CLI (command line input), furthermore check
    # if the file exists, otherwise print error message and abort
    startf_file = global_path + "startf_003-033F"  # 'startf_000-033F'

    data = pd.read_csv(
        startf_file,
        skiprows=0,
        skipfooter=0,
        sep=" ",
        header=None,
        names=["lon", "lat", "z", "z_type", "origin"],
        engine="python",
        # parse_dates=["termin"],
        skipinitialspace=True,
    )

    # print(data)
    print(data["origin"])

    i = 0
    for str in data["origin"]:
        # print(str)
        if "~" in str:
            data["origin"][i] = str[: (len(str) - 2)]
        i += 1
    print(data["origin"])
    return data
