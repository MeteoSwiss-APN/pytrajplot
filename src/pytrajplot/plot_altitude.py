"""Generate Altitude Figure."""


# Standard library
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


def plot_altitude(trajectory_dict, output_dir):
    # print(trajectory_dict)
    for key in trajectory_dict:  # iterate through the trajectory dict
        print(f"--- defining plot properties for {key}")

        ### CASE: WITH SIDE TRAJECTORIES
        y = {
            "altitude_1": {
                "origin": None,
                "y_surf": None,
                "y_type": None,
                "alt_level": None,
                "y0": {
                    "z": [],
                    "z_type": None,
                    "line": "-r",
                    "alpha": 1,
                },  # main trajectory
                "y1": {
                    "z": [],
                    "z_type": None,
                    "line": "-r",
                    "alpha": 0.5,
                },  # side trajectory 1
                "y2": {
                    "z": [],
                    "z_type": None,
                    "line": "-r",
                    "alpha": 0.5,
                },  # side trajectory 2
                "y3": {
                    "z": [],
                    "z_type": None,
                    "line": "-r",
                    "alpha": 0.5,
                },  # side trajectory 3
                "y4": {
                    "z": [],
                    "z_type": None,
                    "line": "-r",
                    "alpha": 0.5,
                },  # side trajectory 4
            },
            "altitude_2": {
                "origin": None,
                "y_surf": None,
                "y_type": None,
                "alt_level": None,
                "y0": {
                    "z": [],
                    "z_type": None,
                    "line": "b-",
                    "alpha": 1,
                },  # main trajectory
                "y1": {
                    "z": [],
                    "z_type": None,
                    "line": "b-",
                    "alpha": 0.5,
                },  # side trajectory 1
                "y2": {
                    "z": [],
                    "z_type": None,
                    "line": "b-",
                    "alpha": 0.5,
                },  # side trajectory 2
                "y3": {
                    "z": [],
                    "z_type": None,
                    "line": "b-",
                    "alpha": 0.5,
                },  # side trajectory 3
                "y4": {
                    "z": [],
                    "z_type": None,
                    "line": "b-",
                    "alpha": 0.5,
                },  # side trajectory 4
            },
            "altitude_3": {
                "origin": None,
                "y_surf": None,
                "y_type": None,
                "alt_level": None,
                "y0": {
                    "z": [],
                    "z_type": None,
                    "line": "g-",
                    "alpha": 1,
                },  # main trajectory
                "y1": {
                    "z": [],
                    "z_type": None,
                    "line": "g-",
                    "alpha": 0.5,
                },  # side trajectory 1
                "y2": {
                    "z": [],
                    "z_type": None,
                    "line": "g-",
                    "alpha": 0.5,
                },  # side trajectory 2
                "y3": {
                    "z": [],
                    "z_type": None,
                    "line": "g-",
                    "alpha": 0.5,
                },  # side trajectory 3
                "y4": {
                    "z": [],
                    "z_type": None,
                    "line": "g-",
                    "alpha": 0.5,
                },  # side trajectory 4
            },
            "altitude_4": {
                "origin": None,
                "y_surf": None,
                "y_type": None,
                "alt_level": None,
                "y0": {
                    "z": [],
                    "z_type": None,
                    "line": "k-",
                    "alpha": 1,
                },  # main trajectory
                "y1": {
                    "z": [],
                    "z_type": None,
                    "line": "k-",
                    "alpha": 0.5,
                },  # side trajectory 1
                "y2": {
                    "z": [],
                    "z_type": None,
                    "line": "k-",
                    "alpha": 0.5,
                },  # side trajectory 2
                "y3": {
                    "z": [],
                    "z_type": None,
                    "line": "k-",
                    "alpha": 0.5,
                },  # side trajectory 3
                "y4": {
                    "z": [],
                    "z_type": None,
                    "line": "k-",
                    "alpha": 0.5,
                },  # side trajectory 4
            },
        }

        trajectory_df = trajectory_dict[key]  # extract df for given key
        trajectory_df.to_csv("trajectory_" + key + ".csv", index=False)
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
        alt_index = 1  # altitude 1,2,3,4
        traj_index = 0  # 0=main, 1=east, 2=north, 3=west, 4=south

        while row_index < number_of_trajectories:

            # print(f'row: {row_index}, altitude: {alt_index}, trajectory: {traj_index}')
            lower_row = row_index * number_of_times
            upper_row = row_index * number_of_times + number_of_times
            origin = trajectory_df["origin"].loc[lower_row]
            altitude_levels = trajectory_df["altitude_levels"].loc[lower_row]

            # print(origin)
            # print(trajectory_df['side_traj'].loc[lower_row])

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
                    "~" not in origin
                ):  # the y_surf information, is only taken from the main trajectories (not side trajectories)
                    # print(f'row_index = {row_index} corresponds to origin {origin}')
                    y["altitude_" + str(alt_index)]["origin"] = origin
                    # print(trajectory_df['hsurf'][lower_row:upper_row])
                    y["altitude_" + str(alt_index)]["y_surf"] = trajectory_df["hsurf"][
                        lower_row:upper_row
                    ]

                y["altitude_" + str(alt_index)]["y_type"] = trajectory_df["z_type"][
                    lower_row
                ]
                y["altitude_" + str(alt_index)]["alt_level"] = trajectory_df["z"][
                    lower_row
                ]
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
                    )

            else:
                # print(f'row_index = {row_index} corresponds to origin {origin}')
                y["altitude_" + str(alt_index)]["origin"] = origin
                y["altitude_" + str(alt_index)]["y_surf"] = trajectory_df["hsurf"][
                    lower_row:upper_row
                ]
                y["altitude_" + str(alt_index)]["y_type"] = trajectory_df["z_type"][
                    lower_row
                ]
                y["altitude_" + str(alt_index)]["alt_level"] = trajectory_df["z"][
                    lower_row
                ]
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
                    )
    return


def generate_altitude_plot(x, y, key, side_traj, output_dir, altitude_levels):

    output_dir = "src/pytrajplot/plots/"

    converter = mdates.ConciseDateConverter()
    munits.registry[np.datetime64] = converter
    munits.registry[datetime.date] = converter
    munits.registry[datetime.datetime] = converter

    origin = y["altitude_1"]["origin"]

    print(f"--- generating altitude plot for {origin}")

    if y["altitude_1"]["y0"]["z_type"] == "hpa":
        unit = "hPa"  # unit for the HRES case
        custom_ylim = (500, 1000)
    else:
        unit = "m"

    # to cases need to be distinguished:
    # regular altitude plots:   4 subplots
    # pollen altitude plots:    2 subplots

    # map the altitudes to the subplots via alt_dict (enables variable number of subplots, created dynamically)
    alt_levels_tmp = altitude_levels
    alt_dict = {}
    tmp = 0
    while tmp < altitude_levels:
        alt_dict[tmp] = alt_levels_tmp
        tmp += 1
        alt_levels_tmp -= 1

    fig, axs = plt.subplots(altitude_levels, 1, tight_layout=True, sharex=True)
    # Setting the values for all y-axes.
    plt.setp(axs, ylim=custom_ylim)
    plt.setp(axs, ylabel="Altitude [" + str(unit) + "]")
    props = dict(boxstyle="square", facecolor="white", alpha=0.3)

    if side_traj:
        # alt_level = [1,2,3,4]
        traj_index = [0, 1, 2, 3, 4]

        for nn, ax in enumerate(axs):
            alt = alt_dict[nn]
            if unit == "hPa":
                ax.invert_yaxis()
            ax.grid(color="grey", linestyle="--", linewidth=1)
            ax.fill_between(
                x,
                [custom_ylim[1]] * len(x),
                y["altitude_" + str(alt)]["y_surf"],
                color="brown",
                alpha=0.5,
            )
            for traj in traj_index:
                textstr = str(y["altitude_" + str(alt)]["alt_level"]) + " " + unit
                # print(f'plotting trajectory {traj} on altitude level {alt} in subplot {nn}')
                ax.plot(
                    x,  # define x-axis
                    y["altitude_" + str(alt)]["y" + str(traj)]["z"],  # define y-axis
                    y["altitude_" + str(alt)]["y" + str(traj)][
                        "line"
                    ],  # define linestyle
                    alpha=y["altitude_" + str(alt)]["y" + str(traj)][
                        "alpha"
                    ],  # define line opacity
                )

                # ax.fill_between(x,[custom_ylim[1]]*len(x), y['altitude_' + str(alt)]['y_surf'], color='brown', alpha=.1)
                ax.text(
                    0.8,
                    0.9,
                    textstr,
                    transform=ax.transAxes,  # fontsize=10,
                    verticalalignment="top",
                    bbox=props,
                )

    else:  # no side traj
        for nn, ax in enumerate(axs):
            alt = alt_dict[nn]
            if unit == "hPa":
                ax.invert_yaxis()

            ax.grid(color="grey", linestyle="--", linewidth=1)
            textstr = str(y["altitude_" + str(alt)]["alt_level"]) + " " + unit
            # print(f'plotting main trajectory on altitude level {alt} in subplot {nn}')
            ax.plot(
                x,  # define x-axis
                y["altitude_" + str(alt)]["y0"]["z"],  # define y-axis
                y["altitude_" + str(alt)]["y0"]["line"],  # define linestyle
                alpha=y["altitude_" + str(alt)]["y0"]["alpha"],  # define line opacity
            )
            ax.fill_between(
                x,
                [custom_ylim[1]] * len(x),
                y["altitude_" + str(alt)]["y_surf"],
                color="brown",
                alpha=0.5,
            )
            ax.text(
                0.8,
                0.9,
                textstr,
                transform=ax.transAxes,  # fontsize=10,
                verticalalignment="top",
                bbox=props,
            )

    outpath = os.getcwd() + "/" + output_dir + "/" + key + "/"

    os.makedirs(
        outpath, exist_ok=True
    )  # create plot folder if it doesn't already exist

    fig.suptitle("Altitude Plot for " + origin)
    plt.savefig(outpath + origin + ".png")
    # print('Saved plot: ', outpath + origin + '.png') # prints location of saved figure for further processing

    return
