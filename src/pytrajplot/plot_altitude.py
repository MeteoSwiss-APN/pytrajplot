"""Generate Altitude Figure."""
# Standard library
import datetime
import locale
import os

# Third-party
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.units as munits
import numpy as np


def create_y_dict(altitude_levels):
    """Create dict of dicts to be filled with information for all y-axes.

    Args:
        altitude_levels (int): Number of y-axis dicts (for each alt. level one dict) that need to be added to y

    Returns:
        y (dict): Dict of dicts. For each altitude level, one dict is present in this dict. Each of those 'altitude dicts' contains the relevant information to plot the corresponding subplot.

    """
    assert (
        altitude_levels <= 10
    ), "It is not possible, to generate altitude plots for more than 10 different starting altitudes."

    y = {}

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
            "y0": {
                "z": [],
                "z_type": None,
                "line": None,  # "line": color_dict[i],
                "alpha": 1,
            },  # main trajectory
            "y1": {
                "z": [],
                "z_type": None,
                "line": None,  # "line": color_dict[i],
                "alpha": 0.5,
            },  # side trajectory 1
            "y2": {
                "z": [],
                "z_type": None,
                "line": None,  # "line": color_dict[i],
                "alpha": 0.5,
            },  # side trajectory 2
            "y3": {
                "z": [],
                "z_type": None,
                "line": None,  # "line": color_dict[i],
                "alpha": 0.5,
            },  # side trajectory 3
            "y4": {
                "z": [],
                "z_type": None,
                "line": None,  # "line": color_dict[i],
                "alpha": 0.5,
            },  # side trajectory 4
        }
        y[key_name + str(i)] = altitude_dict
        i += 1

    return y


def plot_altitude(trajectory_dict, output_dir, separator, language):
    """Iterate through trajectory dict. For each key, generate altitude plots for all origins/altitude levels.

    Args:
        trajectory_dict (dict): [description]
        output_dir (str): Path to output directory (where the subdirectories containing all plots should be saved)
        separator (str): Separator string to identify main- and side-trajectories (default: '~')
        language (str): Language for plot annotations.

    """
    for key in trajectory_dict:  # iterate through the trajectory dict
        # print(f"--- defining altitude plot properties for {key}")

        y = create_y_dict(
            altitude_levels=trajectory_dict[key]["altitude_levels"].loc[0]
        )

        trajectory_df = trajectory_dict[key]  # extract df for given key
        number_of_times = trajectory_df["block_length"].iloc[
            0
        ]  # block length is constant, because it depends on the runtime of the model and the timestep, which both are constant for a given traj file
        number_of_trajectories = trajectory_df["#trajectories"].iloc[
            0
        ]  # this parameter, corresponds to the number of rows, present in the start file, thus also constant
        x = trajectory_df["datetime"].iloc[
            0:number_of_times
        ]  # shared x-axis is the time axis, which is constant for a given traj file

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
                    y["altitude_" + str(alt_index)]["origin"] = origin
                    # print(trajectory_df['hsurf'][lower_row:upper_row])
                    y["altitude_" + str(alt_index)]["y_surf"] = trajectory_df["hsurf"][
                        lower_row:upper_row
                    ]
                    y["altitude_" + str(alt_index)]["subplot_index"] = subplot_index
                    y["altitude_" + str(alt_index)][
                        "max_start_altitude"
                    ] = max_start_altitude

                y["altitude_" + str(alt_index)]["y_type"] = trajectory_df["z_type"][
                    lower_row
                ]
                y["altitude_" + str(alt_index)]["alt_level"] = trajectory_df[
                    "start_altitude"
                ][lower_row]
                y["altitude_" + str(alt_index)]["y" + str(traj_index)][
                    "z"
                ] = trajectory_df["z"][lower_row:upper_row]
                y["altitude_" + str(alt_index)]["y" + str(traj_index)][
                    "z_type"
                ] = trajectory_df["z_type"][lower_row]

                row_index += 1
                traj_index += 1

                if traj_index == 5:
                    traj_index = 0
                    alt_index += 1

                if alt_index > altitude_levels:
                    alt_index = 1
                    generate_altitude_plot(
                        x=x,
                        y=y,
                        key=key,
                        side_traj=side_traj,
                        output_dir=output_dir,
                        altitude_levels=altitude_levels,
                        language=language,
                        max_start_altitude=max_start_altitude,
                    )

            else:
                # print(f'row_index = {row_index} corresponds to origin {origin}')
                y["altitude_" + str(alt_index)]["origin"] = origin
                y["altitude_" + str(alt_index)]["subplot_index"] = subplot_index
                y["altitude_" + str(alt_index)][
                    "max_start_altitude"
                ] = max_start_altitude
                y["altitude_" + str(alt_index)]["y_surf"] = trajectory_df["hsurf"][
                    lower_row:upper_row
                ]
                y["altitude_" + str(alt_index)]["y_type"] = trajectory_df["z_type"][
                    lower_row
                ]
                y["altitude_" + str(alt_index)]["alt_level"] = trajectory_df[
                    "start_altitude"
                ][lower_row]
                y["altitude_" + str(alt_index)]["y0"]["z"] = trajectory_df["z"][
                    lower_row:upper_row
                ]
                y["altitude_" + str(alt_index)]["y1"]["z"] = []
                y["altitude_" + str(alt_index)]["y2"]["z"] = []
                y["altitude_" + str(alt_index)]["y3"]["z"] = []
                y["altitude_" + str(alt_index)]["y4"]["z"] = []
                y["altitude_" + str(alt_index)]["y0"]["z_type"] = trajectory_df[
                    "z_type"
                ][lower_row]
                y["altitude_" + str(alt_index)]["y1"]["z_type"] = None
                y["altitude_" + str(alt_index)]["y2"]["z_type"] = None
                y["altitude_" + str(alt_index)]["y3"]["z_type"] = None
                y["altitude_" + str(alt_index)]["y4"]["z_type"] = None

                row_index += 1
                alt_index += 1
                if alt_index > altitude_levels:
                    alt_index = 1
                    generate_altitude_plot(
                        x=x,
                        y=y,
                        key=key,
                        side_traj=side_traj,
                        output_dir=output_dir,
                        altitude_levels=altitude_levels,
                        language=language,
                        max_start_altitude=max_start_altitude,
                    )
    return


def altitude_limits(y, max_start_altitude, altitude_levels):
    """Define the y-axis limits dynamically.

    Args:
        y (dict):                   Dictionary containing the y-axis information (esp. the altitude)
        max_start_altitude (float): The highest start altitude. Check the altitude for this trajectory only, to define the altitude limits
        altitude_levels (int):      #start altitudes

    Returns:
        unit (str):                 [m] or [hPa]
        custom_ylim (tuple):        (lower y-limit, upper y-limit)

    """
    # print('--- defining the altitude limits')
    i = 1
    max_altitude_array = []

    if y["altitude_1"]["y0"]["z_type"] == "hpa":
        unit = "hPa"  # unit for the HRES case

        while i <= altitude_levels:
            max_altitude_array.append(np.min(y["altitude_" + str(i)]["y0"]["z"]))
            i += 1

        # (lower y-limit, upper y-limit); lower y-limit > upper y-limit to invert the y-axis for pressure altitude

        if np.min(max_altitude_array) >= max_start_altitude:
            custom_ylim = (1020, 0.8 * max_start_altitude)  # 20% margin on top
        else:
            custom_ylim = (1020, 0.8 * np.min(max_altitude_array))

    else:
        unit = "m"
        while i <= altitude_levels:
            max_altitude_array.append(np.max(y["altitude_" + str(i)]["y0"]["z"]))
            i += 1

        if np.max(max_altitude_array) <= max_start_altitude:
            custom_ylim = (0, max_start_altitude + 1000)
        else:
            custom_ylim = (0, np.max(max_altitude_array) + 500)

    return unit, custom_ylim


def generate_altitude_plot(
    x, y, key, side_traj, output_dir, altitude_levels, language, max_start_altitude
):
    """Iterate through y-dict, generate & save plot.

    Args:
        x (df): Pandas Dataframe containing the datetime column of the trajectory dataframe (x-axis information)
        y (dict): Dictionary, containig the y-axis information for all subplots
        key (str): Key string necessary for creating an output folder for each start/trajectory file pair
        side_traj (int): 0/1 --> Necessary, for choosing the correct loop in the plotting pipeline
        output_dir (str): Path to output directory
        altitude_levels (int): #altitude levels = #subplots
        language (str): language for plot annotations
        max_start_altitude (float): maximum start altitude

    """
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

    converter = mdates.ConciseDateConverter()
    munits.registry[np.datetime64] = converter
    munits.registry[datetime.date] = converter
    munits.registry[datetime.datetime] = converter

    origin = y["altitude_1"]["origin"]

    print(f"--- {key}\t{origin}\t plot altitude")

    unit, custom_ylim = altitude_limits(
        y=y, max_start_altitude=max_start_altitude, altitude_levels=altitude_levels
    )

    if altitude_levels == 1:
        # figsize=(width, height)
        fig, axs = plt.subplots(
            altitude_levels,
            1,
            tight_layout=True,
            figsize=(9, 6),
            dpi=150,
        )

    else:
        fig, axs = plt.subplots(
            altitude_levels, 1, tight_layout=True, sharex=True, figsize=(6, 8), dpi=150
        )

    # Setting the values for all y-axes.
    plt.setp(axs, ylim=custom_ylim)

    if language == "en":
        plt.setp(axs, ylabel="Altitude [" + str(unit) + "]")
    if language == "de":
        plt.setp(axs, ylabel="Höhe [" + str(unit) + "]")

    if altitude_levels > 1:
        i = 1
        while i <= altitude_levels:
            alt_level = y["altitude_" + str(i)]["alt_level"]
            sub_index = int(y["altitude_" + str(i)]["subplot_index"])
            # print(f'altitude_{i} = {alt_level} --> subplot {sub_index} (have {altitude_levels} alt levels/subplots)')

            if side_traj:
                traj_index = [0, 1, 2, 3, 4]

                axs[sub_index].grid(color="grey", linestyle="--", linewidth=1)

                y_surf = y["altitude_" + str(i)]["y_surf"]

                lower_boundary = [custom_ylim[0]] * len(x)
                upper_boundary = y_surf

                axs[sub_index].fill_between(
                    x,
                    lower_boundary,
                    upper_boundary,
                    color="brown",
                    alpha=0.5,
                )

                for traj in traj_index:

                    textstr = (
                        str(y["altitude_" + str(i)]["alt_level"])
                        + " "
                        + unit
                        + " ("
                        + y["altitude_" + str(i)]["y" + str(traj)]["z_type"]
                        + ")"
                    )

                    yaxis = y["altitude_" + str(i)]["y" + str(traj)]["z"]
                    ystart = yaxis.iloc[0]
                    xstart = x[0]

                    linestyle = subplot_properties_dict[sub_index]
                    # print(f'linestyle for subplot {sub_index}: {linestyle}')
                    alpha = y["altitude_" + str(i)]["y" + str(traj)]["alpha"]

                    axs[sub_index].plot(
                        x,  # define x-axis
                        yaxis,  # define y-axis
                        linestyle,  # define linestyle
                        alpha=alpha,  # define line opacity
                        label=textstr,
                    )

                    if (
                        y["altitude_" + str(i)]["y" + str(traj)]["alpha"] == 1
                    ):  # only add legend & startpoint for the main trajectories
                        axs[sub_index].plot(
                            xstart,
                            ystart,
                            marker="^",
                            markersize=10,
                            markeredgecolor="red",
                            markerfacecolor="white",
                        )
                        axs[sub_index].legend()

            else:  # no side traj
                axs[sub_index].grid(color="grey", linestyle="--", linewidth=1)

                if key[-1] == "B":
                    y_surf = np.flip(y["altitude_" + str(i)]["y_surf"])
                else:
                    y_surf = y["altitude_" + str(i)]["y_surf"]

                lower_boundary = [custom_ylim[0]] * len(x)
                upper_boundary = y_surf

                axs[sub_index].fill_between(
                    x,
                    lower_boundary,
                    upper_boundary,
                    color="brown",
                    alpha=0.5,
                )

                textstr = (
                    str(y["altitude_" + str(i)]["alt_level"])
                    + " "
                    + unit
                    + " ("
                    + y["altitude_" + str(i)]["y0"]["z_type"]
                    + ")"
                )

                yaxis = y["altitude_" + str(i)]["y0"]["z"]
                ystart = yaxis.iloc[0]
                xstart = x[0]

                linestyle = subplot_properties_dict[sub_index]
                alpha = y["altitude_" + str(i)]["y0"]["alpha"]

                # plot altitude profile
                axs[sub_index].plot(
                    x,  # define x-axis
                    yaxis,  # define y-axis
                    linestyle,  # define linestyle
                    alpha=alpha,  # define line opacity
                    label=textstr,
                )

                # plot starting marker
                axs[sub_index].plot(
                    xstart,
                    ystart,
                    marker="^",
                    markersize=10,
                    markeredgecolor="red",
                    markerfacecolor="white",
                )
                axs[sub_index].legend()

            i += 1

    else:  # only one subplot

        axs.grid(color="grey", linestyle="--", linewidth=1)

        y_surf = y["altitude_1"]["y_surf"]

        lower_boundary = [custom_ylim[0]] * len(x)
        upper_boundary = y_surf

        axs.fill_between(
            x,
            lower_boundary,
            upper_boundary,
            color="brown",
            alpha=0.5,
        )

        if y["altitude_1"]["y1"]["z_type"] is not None:
            traj_index = [0, 1, 2, 3, 4]
            for traj in traj_index:
                textstr = (
                    str(y["altitude_1"]["alt_level"])
                    + " "
                    + unit
                    + " ("
                    + y["altitude_1"]["y" + str(traj)]["z_type"]
                    + ")"
                )

                yaxis = y["altitude_1"]["y" + str(traj)]["z"]
                ystart = yaxis.iloc[0]
                xstart = x[0]

                linestyle = subplot_properties_dict[0]
                # print(f'linestyle for subplot {sub_index}: {linestyle}')
                alpha = y["altitude_1"]["y" + str(traj)]["alpha"]

                axs.plot(
                    x,  # define x-axis
                    yaxis,  # define y-axis
                    linestyle,  # define linestyle
                    alpha=alpha,  # define line opacity
                    label=textstr,
                )

                if (
                    y["altitude_1"]["y" + str(traj)]["alpha"] == 1
                ):  # only add legend & startpoint for the main trajectories
                    axs.plot(
                        xstart,
                        ystart,
                        marker="^",
                        markersize=10,
                        markeredgecolor="red",
                        markerfacecolor="white",
                    )
                    axs.legend()

        else:
            textstr = (
                str(y["altitude_1"]["alt_level"])
                + " "
                + unit
                + " ("
                + y["altitude_1"]["y0"]["z_type"]
                + ")"
            )

            yaxis = y["altitude_1"]["y0"]["z"]
            ystart = yaxis.iloc[0]
            xstart = x[0]

            linestyle = subplot_properties_dict[0]
            alpha = y["altitude_1"]["y0"]["alpha"]

            # plot altitude profile
            axs.plot(
                x,  # define x-axis
                yaxis,  # define y-axis
                linestyle,  # define linestyle
                alpha=alpha,  # define line opacity
                label=textstr,
            )

            # plot starting marker
            axs.plot(
                xstart,
                ystart,
                marker="^",
                markersize=10,
                markeredgecolor="red",
                markerfacecolor="white",
            )
            axs.legend()

    outpath = os.getcwd() + "/" + output_dir + "/plots/" + key + "/"

    os.makedirs(
        outpath, exist_ok=True
    )  # create plot folder if it doesn't already exist

    if language == "en":
        locale.setlocale(locale.LC_ALL, "en_GB")
        fig.suptitle("Altitude Plot for " + origin)
    if language == "de":
        fig.suptitle("Höhenplot für " + origin)

    # print(f'AltPlt: fig={fig}, type(fig)={type(fig)}, ax={axs}, type(ax)={type(axs)}')

    plt.savefig(outpath + origin + "_altitude.png")
    plt.close(fig)
    # print('Saved plot: ', outpath + origin + '.png') # prints location of saved figure for further processing

    return
