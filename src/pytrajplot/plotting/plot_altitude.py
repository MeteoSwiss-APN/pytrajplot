"""Generate Altitude Figure."""

# Standard library
import datetime
import locale

# Third-party
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.units as munits
import numpy as np

# Local
from .plot_utils import subplot_properties_dict


def altitude_limits(plot_dict, max_start_altitude, altitude_levels):
    """Define the y-axis limits dynamically.

    Args:
        plot_dict
                            dict       Dictionary containing the y-axis information (esp. the altitude)
        max_start_altitude
                            float      Highest start altitude - derive upper limit from these altitude-values.
        altitude_levels
                            int        # start altitudes

    Returns:
        unit
                            str        [m] or [hPa]
        custom_ylim
                            tuple      (lower y-limit, upper y-limit)

    """
    i = 1
    max_altitude_array = []

    if plot_dict["altitude_1"]["traj_0"]["z_type"] == "hpa":
        unit = "hPa"  # unit for the HRES case

        while i <= altitude_levels:
            max_altitude_array.append(
                np.min(plot_dict["altitude_" + str(i)]["traj_0"]["z"])
            )
            i += 1

        if np.min(max_altitude_array) >= max_start_altitude:
            custom_ylim = (1020, 0.8 * max_start_altitude)  # 20% margin on top
        else:
            custom_ylim = (1020, 0.8 * np.min(max_altitude_array))

    else:
        unit = "m"
        while i <= altitude_levels:
            max_altitude_array.append(
                np.max(plot_dict["altitude_" + str(i)]["traj_0"]["z"])
            )
            i += 1

        if np.max(max_altitude_array) <= max_start_altitude:
            custom_ylim = (0, max_start_altitude + 1000)
        else:
            custom_ylim = (0, np.max(max_altitude_array) + 500)

    return unit, custom_ylim


def generate_altitude_plot(
    x,
    plot_dict,
    key,
    side_traj,
    altitude_levels,
    language,
    max_start_altitude,
    alt_index,
    sub_index,
    ax=None,
):
    """Generate the altitude plots.

    Args:
        x                   df         Pandas Dataframe containing the datetime column of the trajectory dataframe (x-axis information)
        plot_dict           dict       Dictionary, containig the y-axis information for all subplots
        key                 str        Key string necessary for creating an output folder for each start/trajectory file pair
        side_traj           int        0/1 --> Necessary, for choosing the correct loop in the plotting pipeline
        altitude_levels     int        #altitude levels = #subplots
        language            str        language for plot annotations
        max_start_altitude  float      maximum start altitude
        alt_index           int        index of current altitude (in dict)
        sub_index           int        index of corresponding subplot

        ax (Axes): Axes to plot the altitude on. Defaults to None.

    Returns:
        ax (Axes): Axes w/ altitude plot.


    """
    ax = ax or plt.gca()

    if language == "en":
        locale.setlocale(locale.LC_ALL, "en_GB")
    else:  # language == "de"
        locale.setlocale(locale.LC_ALL, "de_DE")

    if sub_index != (altitude_levels - 1):
        ax.set_xticklabels([])

    plt.tick_params(axis="both", labelsize=8)

    converter = mdates.ConciseDateConverter()
    munits.registry[np.datetime64] = converter
    munits.registry[datetime.date] = converter
    munits.registry[datetime.datetime] = converter

    # print(f"--- {key} > plot altitude \t{origin}")

    unit, custom_ylim = altitude_limits(
        plot_dict=plot_dict,
        max_start_altitude=max_start_altitude,
        altitude_levels=altitude_levels,
    )

    # Setting the values for all y-axes.
    plt.setp(ax, ylim=custom_ylim)
    ax.grid(color="grey", linestyle="--", linewidth=1)

    if language == "en":
        plt.setp(ax, ylabel="Altitude [" + str(unit) + "]")
    if language == "de":
        plt.setp(ax, ylabel="Höhe [" + str(unit) + "]")

    if side_traj:
        traj_index = [0, 1, 2, 3, 4]

        y_surf = plot_dict["altitude_" + str(alt_index)]["y_surf"]

        lower_boundary = [custom_ylim[0]] * len(x)
        upper_boundary = y_surf

        ax.fill_between(
            x,
            lower_boundary,
            upper_boundary,
            color="brown",
            alpha=0.5,
            rasterized=True,
        )

        for traj in traj_index:
            textstr = (
                str(plot_dict["altitude_" + str(alt_index)]["alt_level"])
                + " "
                + unit
                + " ("
                + plot_dict["altitude_" + str(alt_index)]["traj_" + str(traj)]["z_type"]
                + ")"
            )

            yaxis = plot_dict["altitude_" + str(alt_index)]["traj_" + str(traj)]["z"]
            yaxis_masked = np.ma.masked_where(np.isnan(yaxis.values), yaxis.values)

            ystart = yaxis.iloc[0]
            xstart = x[0]

            linestyle = subplot_properties_dict[sub_index]
            alpha = plot_dict["altitude_" + str(alt_index)]["traj_" + str(traj)][
                "alpha"
            ]

            ax.plot(
                x,
                yaxis_masked,
                linestyle,
                alpha=alpha,
                label=textstr,
                rasterized=True,
            )

            # this if statement differentiates between main & side trajectories
            # alternatively: if traj==0 should also work.
            if (
                plot_dict["altitude_" + str(alt_index)]["traj_" + str(traj)]["alpha"]
                == 1
            ):  # only add legend & startpoint for the main trajectories
                ax.plot(
                    xstart,
                    ystart,
                    marker="D",
                    markersize=10,
                    markeredgecolor="red",
                    markerfacecolor="white",
                    rasterized=True,
                )
                ax.legend(fontsize=8)

    else:  # no side traj
        if key[-1] == "B":
            y_surf = np.flip(plot_dict["altitude_" + str(alt_index)]["y_surf"])
        else:
            y_surf = plot_dict["altitude_" + str(alt_index)]["y_surf"]

        lower_boundary = [custom_ylim[0]] * len(x)
        upper_boundary = y_surf

        ax.fill_between(
            x,
            lower_boundary,
            upper_boundary,
            color="brown",
            alpha=0.5,
            rasterized=True,
        )

        textstr = (
            str(plot_dict["altitude_" + str(alt_index)]["alt_level"])
            + " "
            + unit
            + " ("
            + plot_dict["altitude_" + str(alt_index)]["traj_0"]["z_type"]
            + ")"
        )

        yaxis = plot_dict["altitude_" + str(alt_index)]["traj_0"]["z"]
        ystart = yaxis.iloc[0]
        xstart = x[0]

        linestyle = subplot_properties_dict[sub_index]
        alpha = plot_dict["altitude_" + str(alt_index)]["traj_0"]["alpha"]

        # plot altitude profile
        ax.plot(
            x,
            yaxis,
            linestyle,
            alpha=alpha,
            label=textstr,
            rasterized=True,
        )

        # plot starting marker
        ax.plot(
            xstart,
            ystart,
            marker="D",
            markersize=10,
            markeredgecolor="red",
            markerfacecolor="white",
            rasterized=True,
        )
        ax.legend(fontsize=8)

    if key[-1] == "F":
        ax.set_xlim(left=x.iloc[0], right=x.iloc[-1])
    if key[-1] == "B":
        ax.set_xlim(left=x.iloc[-1], right=x.iloc[0])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return ax
