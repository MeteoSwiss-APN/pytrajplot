"""Function, to assemble the plot info header."""

# Third-party
import matplotlib.pyplot as plt


def generate_info_header(language, plot_dict, ax=None):
    """Generate info header.

    Args:
        language            str        language for plot annotations
        plot_dict           dict       Dict w/ information for info header (taken from start/traj files)

        ax (Axes): Axes to plot the info header on. Defaults to None.

    Returns:
        ax (Axes): Axes w/ info header.

    """
    ax = ax or plt.gca()
    ax.axis("off")

    origin = plot_dict["altitude_1"]["origin"]
    y_type = plot_dict["altitude_1"][
        "y_type"
    ]  # anaologous: plot_data["altitude_1"]["traj_0"]["z_type"]
    lon_0 = format(round(plot_dict["altitude_1"]["lon_precise"], 3), ".3f")
    lat_0 = format(round(plot_dict["altitude_1"]["lat_precise"], 3), ".3f")
    trajectory_direction = plot_dict["altitude_1"]["trajectory_direction"]
    elevation = ""
    start_time = str(plot_dict["altitude_1"]["start_time"])[
        :-3
    ]  # remove the seconds from the time in the header

    for key in plot_dict:
        if plot_dict[key]["subplot_index"] == (len(plot_dict.keys()) - 1):
            elevation = int(plot_dict[key]["y_surf"].iloc[0])

    # if not elevation:
    #     if language == "en":
    #         elevation = "not available"
    #     else:
    #         elevation = "nicht verfügbar"

    if y_type == "hpa":
        unit = "hPa"
    else:
        unit = "m"

    if language == "en":
        if trajectory_direction == "F":
            title = f"Forward Trajectory from {origin} departing on: {start_time} UTC"
            site = r"$\it{Release\;Site:}$ "
        if trajectory_direction == "B":
            title = f"Backward Trajectory from {origin} arriving on: {start_time} UTC"
            site = r"$\it{Receptor\;Site:}$ "
        info = (
            site
            + f"{origin}"
            + "  |  "
            + r"$\it{Coordinates:}$ "
            + f" {lat_0}"
            + "°N, "
            + f"{lon_0}°E  |  "
            + r"$\it{Elevation:}$ "
            + f" {elevation} {unit}"
        )

    else:
        if trajectory_direction == "F":
            title = f"Vorwärts-Trajektorie von {origin} gestartet am: {start_time} UTC"
            site = r"$\it{Ursprungsort:}$ "
        if trajectory_direction == "B":
            title = f"Rückwärts-Trajektorie von {origin} gestartet am: {start_time} UTC"
            site = r"$\it{Ankunftsort:}$ "
        info = (
            site
            + f"{origin}"
            + "  |  "
            + r"$\it{Koordinaten:}$"
            + f" {lat_0}"
            + "°N, "
            + f"{lon_0}°E  |  "
            + r"$\it{Elevation:}$"
            + f" {elevation} {unit}"
        )

    # title
    ax.text(
        x=0.5,
        y=0.6,
        s=title,
        transform=ax.transAxes,
        va="center",
        ha="center",
        weight="bold",
        fontdict={
            "fontsize": 25,
            "color": "black",
            # "verticalalignment": "baseline",
        },
    )

    props = dict(
        boxstyle="square",  # 'round',
        facecolor="#FFFAF0",
        alpha=1,
    )
    # description
    ax.text(
        x=0.5,
        y=0.15,
        s=info,
        transform=ax.transAxes,
        va="center",
        ha="center",
        fontdict={
            "fontsize": 15,
            "color": "black",
        },
        bbox=props,
    )
    return ax
