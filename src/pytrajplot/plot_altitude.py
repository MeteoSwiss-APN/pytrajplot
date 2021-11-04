"""Generate Altitude Figure."""
# Standard library
import locale
import os

# Third-party
# import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.units as munits

plt.rcParams["axes.grid"] = True
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = [6, 8]


# Standard library
import datetime

# Third-party
import numpy as np


def create_y_dict(altitude_levels):
    assert (
        altitude_levels <= 10
    ), "It is not possible, to generate altitude plots for more than 10 different starting altitudes."

    y = {}

    key_name = "altitude_"

    i = 1
    # I chose 10 to be the maximum number of altitude plots
    # color_dict = {
    #     1: "r-",
    #     2: "b-",
    #     3: "g-",
    #     4: "k-",
    #     5: "c-",
    #     6: "m-",
    #     7: "y-",
    #     8: "deepskyblue-",
    #     9: "crimson-",
    #     10: "lightgreen-",
    # }

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
    for key in trajectory_dict:  # iterate through the trajectory dict
        print(f"--- defining plot properties for {key}")

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


def generate_altitude_plot(
    x, y, key, side_traj, output_dir, altitude_levels, language, max_start_altitude
):

    # # TEMPORARY SOLUTION for the single subplot problem
    # if altitude_levels == 1:
    #     altitude_levels = 2
    #     y["altitude_2"] = y["altitude_1"]

    subplot_properties_dict = {
        0: "r-",
        1: "b-",
        2: "g-",
        3: "k-",
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

    print(f"--- generating altitude plot for {origin}")

    if y["altitude_1"]["y0"]["z_type"] == "hpa":
        unit = "hPa"  # unit for the HRES case
        # (lower y-limit, upper y-limit); lower y-limit > upper y-limit to invert the y-axis for pressure altitude
        custom_ylim = (1000, 0.8 * max_start_altitude)  # 20% margin on top
    else:
        unit = "m"
        custom_ylim = (0, max_start_altitude + 1000)  # 20% margin on top

    if altitude_levels == 1:
        # figsize=(width, height)
        fig, axs = plt.subplots(altitude_levels, 1, tight_layout=True, figsize=(9, 6))

    else:
        fig, axs = plt.subplots(altitude_levels, 1, tight_layout=True, sharex=True)

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
            # if unit == "hPa":
            #     axs[sub_index].invert_yaxis()

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

        # print(y['altitude_1'][])

        axs.grid(color="grey", linestyle="--", linewidth=1)

        # if key[-1] == "B":
        #     y_surf = np.flip(y["altitude_1"]["y_surf"])
        # else:
        #     y_surf = y["altitude_1"]["y_surf"]

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

    plt.savefig(outpath + origin + ".png")
    plt.close(fig)
    # print('Saved plot: ', outpath + origin + '.png') # prints location of saved figure for further processing

    return
