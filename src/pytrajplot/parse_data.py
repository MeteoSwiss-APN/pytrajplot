"""Parse data from start, trajectory & plot info files."""
# Standard library
import datetime
import math
import os
from datetime import timedelta

# Third-party
import numpy as np
import pandas as pd

# Local
from .parsing.plot_info import PLOT_INFO


def map_altitudes_and_subplots(unit, unique_start_altitudes, current_altitude):
    """Map (randomly sorted) start altitudes of the trajectories to the subplot indeces in descending order.

    Args:
        unit (str): [m] or [hPa] - altitude axis in [hPa] is inverted when compared to [m]
        unique_start_altitudes (array): array, containing the different start altitudes
        current_altitude (float): current altitude for which the subplot index needs to be assigned

    Returns:
        - (int): subplot index, corresponding to the current altitude

    """
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
        altitude_mapping_dict[unique_start_altitudes[tmp_02]] = subplot_index
        tmp_02 += 1

    return altitude_mapping_dict[current_altitude]


def read_startf(startf_path, separator):
    """Read the start file, containing initial information of all trajectories for a corresponding trajectory file.

    Args:
        startf_path (str):  Path to start file
        separator (str):    String, to indentify side- & main-trajectories. I.e. <origin><separator><trajectory index>

    Returns:
        start_df (pandas df): Dataframe containing the information of the start file

    """
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

    # determine the altitude type. this is a way to distinguish between HRES & COSMO trajectories.
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
    start_df["max_start_altitude"] = None

    unique_origins_list = []
    unique_altitude_levels_dict = {}

    if len(start_df) == 1:  # exactly one trajectory w/o side trajectories in start file
        start_df["side_traj"] = 0  # 1 if there are side trajectories, else 0
        start_df[
            "altitude_levels"
        ] = 1  # #altitude levels for certain origin; variable within start file
        start_df["subplot_index"] = 0
        start_df["max_start_altitude"] = start_df["z"].loc[0]

    else:
        i = 0
        # len(start_df) = #rows in start file (= #trajectories)
        while i < len(start_df):
            if i < len(start_df) - 1:
                # have side trajectories, if Ǝ separator in origin of next trajectory
                if separator in start_df["origin"].loc[i + 1]:
                    # The 5 rows, that correspond to this origin, are assigned: 1 (=True) to the side_traj column
                    start_df.loc[i : i + 4, "side_traj"] = 1
                    # The 5 rows, that correspond to this origin, are assigned: the number of times, this origin occurs in this start file, to the #alt_levels column
                    current_origin = start_df["origin"].loc[i]
                    start_df.loc[i : i + 4, "altitude_levels"] = start_df.loc[
                        start_df.origin == current_origin, "origin"
                    ].count()

                    if current_origin not in unique_origins_list:
                        unique_origins_list.append(current_origin)
                        origin = current_origin
                        altitude_levels = start_df["altitude_levels"].loc[i]

                        # add information to unique_altitude_levels_dict
                        unique_altitude_levels_dict[current_origin] = (
                            start_df["z"]
                            .loc[i : (i + 5 * altitude_levels) - 1]
                            .unique()
                        )

                    # The correct subplot index is assigned to the main + side trajectories. This can be achieved w/ the
                    # map_altidude_and_subplots function. The order of altitude levels in the start file is of no significance.
                    start_df.loc[
                        i : i + 4, "subplot_index"
                    ] = map_altitudes_and_subplots(
                        unit=unit,
                        unique_start_altitudes=unique_altitude_levels_dict[origin],
                        current_altitude=start_df["z"].loc[i],
                    )

                    if unit == "hPa":
                        # if altidude is measured w/ hPa, the highest altitude is the SMALLEST number
                        start_df.loc[i : i + 4, "max_start_altitude"] = np.min(
                            unique_altitude_levels_dict[origin]
                        )
                    if unit == "m":
                        # conversely, the max_start_altidue in meters is the LARGEST number
                        start_df.loc[i : i + 4, "max_start_altitude"] = np.max(
                            unique_altitude_levels_dict[origin]
                        )
                    # skip to the next origin, by skipping over the next four rows, which are just the side trajectories of the current origin
                    i += 5

                # fill the start_df for trajectories, that have no side trajectories.
                # analogous to the case with side_trajectories. see comments there for reference.
                else:  # no side trajectories
                    start_df.loc[i : i + 3, "side_traj"] = 0

                    start_df.loc[i : i + 3, "altitude_levels"] = start_df.loc[
                        start_df.origin == start_df["origin"].loc[i], "origin"
                    ].count()

                    if start_df["origin"].loc[i] not in unique_origins_list:
                        unique_origins_list.append(start_df["origin"].loc[i])
                        origin = start_df["origin"].loc[i]
                        altitude_levels = start_df["altitude_levels"].loc[
                            i
                        ]  # altitude levels for given location
                        # add information to unique_altitude_levels_dict
                        unique_altitude_levels_dict[start_df["origin"].loc[i]] = (
                            start_df["z"].loc[i : (i + altitude_levels)].unique()
                        )

                    if unit == "hPa":
                        start_df.loc[i : i + 3, "max_start_altitude"] = np.min(
                            unique_altitude_levels_dict[origin]
                        )
                    if unit == "m":
                        start_df.loc[i : i + 3, "max_start_altitude"] = np.max(
                            unique_altitude_levels_dict[origin]
                        )

                    tmp = i
                    while tmp < i + 4:
                        start_df.loc[tmp, "subplot_index"] = map_altitudes_and_subplots(
                            unit=unit,
                            unique_start_altitudes=unique_altitude_levels_dict[origin],
                            current_altitude=start_df["z"].loc[tmp],
                        )
                        tmp += 1

                    i += 4

    return start_df


def traj_helper_fct(case, file_path, firstline, start_df):
    """Handle the different type of trajectory files (COSMO/HRES). Compute the number of rows that make up one trajectory and the number of trajectories.

    Args:
        case (str): HRES/COSMO
        file_path (str): Path to trajectory file
        firstline (str): Content of the first line of trajectory file (relevant for HRES case)
        start_df (df): Dataframe containing the information of the corresponding start file

    Returns:
        number_of_trajectories (int):   #rows in trajectory file, that make up one trajectory (const. for a given traj. file)
        number_of_times (int):          #trajectory blocks (= #rows in start file)

    """
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
        # for HRES trajectories, the number_of_trajectories & number_of_times needs to be computed
        number_of_trajectories = len(start_df)
        T = int(firstline[43:49]) / 60
        f = open(file_path)
        lines_to_read = [
            6
        ]  # read the line, after the first dt (see what the timestep was)

        for position, line in enumerate(f):
            if position in lines_to_read:
                dt = float(line[3:7])
                if dt == 0.3:
                    dt = 0.5

        number_of_times = int(T / dt + 1)

    return number_of_trajectories, number_of_times


def convert_time(plot_info_dict, traj_df, key, case, reference_time):
    """Convert time steps into datetime objects.

    Args:
        plot_info_dict (dict): Dict containing the information of plot_info file.
        traj_df (df): Trajectory dataframe containing the time step column (parsed form the trajectory file)
        key (str): Key, containing information about the runtime
        case (str): HRES / COSMO
        reference_time (str): reference time (yyyy-mm-dd HH:MM)

    Returns:
        traj_df (df): Trajectory dataframe w/ added datetime column.

    """
    # reference time for relative times
    format = "%Y-%m-%d %H:%M"
    dt_object = datetime.datetime.strptime(reference_time, format)

    # add new empty key to the traj_df
    traj_df["datetime"] = None

    # add leadtime to model base time
    if case == "HRES":
        # the time discretisation of HRES trajectories is: hours.minutes
        # --> convert to fraction of hour (1.30 hours.minutes = 1.5 hours)
        for counter, time in enumerate(traj_df["time"]):
            minutes = 100 * round(
                math.modf(time)[0], 2
            )  # part after decimal point, converted to minutes
            hour_fraction = (
                minutes / 60
            )  # number of minutes converted to fraction of hour
            hours = math.modf(time)[1]
            delta_t = hours + hour_fraction
            date = dt_object + timedelta(hours=delta_t)
            traj_df.loc[counter, "datetime"] = date

    else:  # Case: COSMO
        for counter, time in enumerate(traj_df["time"]):
            delta_t = float(time)
            date = dt_object + timedelta(hours=delta_t)
            traj_df.loc[counter, "datetime"] = date

    return traj_df


def read_trajectory(trajectory_file_path, start_df, plot_info_dict):
    """Parse trajectory file.

    Args:
        trajectory_file_path (str): Path to trajectory file.
        start_df (df): Dataframe containing the information of the corresponding start file
        plot_info_dict (dict): Dict containing the information of plot_info file

    Returns:
        traj_df (df):                   Dataframe containig information of trajectory file

    """
    # read first line of trajectory file to check which case it is.
    with open(trajectory_file_path) as f:
        firstline = f.readline().rstrip()
        secondline = f.readline().rstrip()

    # based on the structure of the **current** trajectory files, the two cases: HRES & COSMO can (and must) be distinguished.
    if firstline[:8] == "LAGRANTO":  # case: COSMO trajectory file
        case = "COSMO"
        skiprows = 21
        number_of_trajectories, number_of_times = traj_helper_fct(
            case=case,
            file_path=trajectory_file_path,
            firstline=firstline,
            start_df=start_df,
        )
        reference_time = secondline[16:32]

    elif firstline[:9] == "Reference":  # case: HRES trajectory file
        case = "HRES"
        skiprows = 5
        number_of_trajectories, number_of_times = traj_helper_fct(
            case=case,
            file_path=trajectory_file_path,
            firstline=firstline,
            start_df=start_df,
        )
        reference_time = "{y}-{m}-{d} {H}:{M}".format(
            y=firstline[15:19],
            m=firstline[19:21],
            d=firstline[21:23],
            H=firstline[24:26],
            M=firstline[26:28],
        )

    else:  # case: unknown trajectory file format
        raise Exception("Unknown trajectory file format (cannot parse header info)")

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

    # clean up the df in case its generated from a COSMO trajectory file
    if case == "COSMO":
        # at missing data points (i.e. if trajectory leaves computational domain), the z & hsurf values default to -999
        traj_df.loc[(traj_df["z"] < 0), "z"] = np.NaN
        traj_df.loc[(traj_df["hsurf"] < 0), "hsurf"] = np.NaN
        traj_df.dropna(
            subset=["lon"], inplace=True
        )  # remove rows containing only the origin/z_type
        traj_df.reset_index(inplace=True)

    # clean up the df in case its generated from a HRES trajectory file
    if case == "HRES":
        traj_df.loc[(traj_df["lon"] == -999.00), "lon"] = np.NaN
        traj_df.loc[(traj_df["lat"] == -999.00), "lat"] = np.NaN
        traj_df.loc[(traj_df["z"] == -999), "z"] = np.NaN

    # add various (empty) keys to the trajectory dataframe
    traj_df["z_type"] = None
    traj_df["origin"] = None
    traj_df["side_traj"] = None
    traj_df["start_altitude"] = np.NaN
    traj_df["lon_precise"] = np.NaN
    traj_df["lat_precise"] = np.NaN
    traj_df["altitude_levels"] = None
    traj_df["#trajectories"] = number_of_trajectories
    traj_df["block_length"] = number_of_times
    traj_df["trajectory_direction"] = trajectory_file_path[
        -1:
    ]  # last letter of key = F/B --> direction of trajectory
    traj_df["subplot_index"] = np.NaN
    traj_df["max_start_altitude"] = np.NaN

    # add information to newly created (empty) keys in trajectory dataframe
    # basically, at this point the information from the start_df, gets merged into the trajectory dataframe
    tmp = 0
    while tmp < number_of_trajectories:
        first_row = tmp * number_of_times
        next_first_row = tmp * number_of_times + number_of_times
        # get info from start_df
        z_type = start_df["z_type"][tmp]
        origin = start_df["origin"][tmp]
        lon_precise = start_df["lon"][tmp]
        lat_precise = start_df["lat"][tmp]
        start_altitude = start_df["z"][tmp]
        side_traj = start_df["side_traj"][tmp]
        altitude_levels = start_df["altitude_levels"][tmp]
        subplot_index = int(start_df["subplot_index"][tmp])
        max_start_altitude = start_df["max_start_altitude"][tmp]
        # add info to traj_df
        traj_df.loc[
            first_row:next_first_row,
            [
                "z_type",
                "origin",
                "lon_precise",
                "lat_precise",
                "side_traj",
                "altitude_levels",
                "start_altitude",
                "subplot_index",
                "max_start_altitude",
            ],
        ] = (
            z_type,
            origin,
            lon_precise,
            lat_precise,
            side_traj,
            altitude_levels,
            start_altitude,
            subplot_index,
            max_start_altitude,
        )

        tmp += 1

    # add column called 'datetime' w/ datetime objects. these are the corresponding time steps from the 'time' column, relative to the
    # init time of the model (taking into consideration the leadtime of course)
    traj_df = convert_time(
        plot_info_dict=plot_info_dict,
        traj_df=traj_df,
        key=trajectory_file_path[-8:],
        case=case,
        reference_time=reference_time,
    )

    # change the lon/lat values where the trajectory leaves the domain from their computational domain-boundary values to np.NaN.
    traj_df.loc[np.isnan(traj_df["z"]), ["lon", "lat"]] = np.NaN

    return traj_df


def check_input_dir(input_dir, prefix_dict, separator):
    """Iterate through the input directory, containg all files to be parsed and plotted.

    Args:
        input_dir (str):    Path to input directory
        prefix_dict (dict): Dict, defining the files which should be parsed (start/plot info/trajectory files)
        separator (str):    String, to indentify side- & main-trajectories.

    Returns:
        trajectory_dict (dict): Dictionary containing a dataframe for each key (start/trajectory file pair). Containing all relevant information for plotting pipeline.
        plot_info_dict (dict):  Dictionary containing the information of the plot info file
        keys (list):            List containing all keys that are present in the trajectory_dict. I.e. ['000-048F'] if there is only one start/traj. file pair.

    """
    print("--- Parsing Input Files")
    start_dict, trajectory_dict = {}, {}
    keys = []
    # iterate through the directory and read the start & plot_info files. collect keys, parse corresponding trajectory files afterwards
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if filename.startswith(prefix_dict["start"]):
            # if filename starts w/ start file prefix
            key = filename[len(prefix_dict["start"]) :]
            if (
                key not in keys
            ):  # filename[len(prefix_dict["start"]) :] ≡ key (i.e. 000-048F)
                keys.append(key)
                # check, that there is a matching trajectory file for each start file
                assert os.path.isfile(
                    os.path.join(
                        input_dir,
                        prefix_dict["trajectory"] + key,
                    )
                ), f"There is no matching trajectory file for the startfile: {prefix_dict['start']+key}"

            start_dict[key] = read_startf(startf_path=file_path, separator=separator)

        if filename.startswith(prefix_dict["plot_info"]):
            # if filename ≡ plot_info filename
            plot_info_dict = PLOT_INFO(file=file_path).data

    # check if plot_info has been found
    if plot_info_dict is None:
        raise RuntimeError("ERROR: plot_info file not found:", prefix_dict["plot_info"])

    # for each start file (identified by its unique key) a corresponding trajectory file exists
    for key in keys:
        file_path = os.path.join(input_dir, prefix_dict["trajectory"] + key)
        trajectory_dict[key] = read_trajectory(
            trajectory_file_path=file_path,
            start_df=start_dict[key],
            plot_info_dict=plot_info_dict,
        )
    return trajectory_dict, plot_info_dict
