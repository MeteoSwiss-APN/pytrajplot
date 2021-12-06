"""Function, to assemble the plot info header."""

# Third-party
import matplotlib.pyplot as plt


def generate_info_header(language, plot_info, plot_data, domain, ax=None):
    """Summary - First line should end with a period.

    Args:
        language            str        language for plot annotations
        plot_info           dict       Dict w/ information for info header (taken from info file)
        plot_data           dict       Dict w/ information for info header (taken from start/traj files)
        domain              str        domain of current plot

        ax (Axes): Axes to plot the info header on. Defaults to None.

    Returns:
        ax (Axes): Axes w/ info header.

    """
    ax = ax or plt.gca()
    ax.axis("off")

    origin = plot_data["altitude_1"]["origin"]
    y_type = plot_data["altitude_1"]["y_type"]
    lon_0 = plot_data["altitude_1"]["traj_0"]["lon"].iloc[0]
    lat_0 = plot_data["altitude_1"]["traj_0"]["lat"].iloc[0]
    model_name = plot_info["model_name"]
    trajectory_direction = plot_data["altitude_1"]["trajectory_direction"]
    elevation = ""

    for key in plot_data:
        if plot_data[key]["subplot_index"] == (len(plot_data.keys()) - 1):
            elevation = plot_data[key]["y_surf"].iloc[0]

    if not elevation:
        if language == "en":
            elevation = "not available"
        else:
            elevation = "nicht verfügbar"

    if y_type == "hpa":
        unit = "hPA"
    else:
        unit = "m"

    start_time = plot_info["mbt"]

    if language == "en":
        if trajectory_direction == "F":
            title = f"Forward Trajectory from {origin} departing on: {start_time}"
            site = "$\it{Release}$ $\it{Site}:} $"
        if trajectory_direction == "B":
            title = f"Backward Trajectory from {origin} arriving on: {start_time}"
            site = "$\it{Receptor}$ $\it{Site}:} $"
    else:
        if trajectory_direction == "F":
            title = f"Vorwärts Trajektorie von {origin} gestartet am: {start_time}"
            site = "$\it{Ursprungsort}$: $"
        if trajectory_direction == "B":
            title = f"Rückwärts Trajektorie von {origin} gestartet am: {start_time}"
            site = "$\it{Ankunftsort}$: "

    ax.set_title(
        title,
        fontdict={
            "fontsize": 20,
            "color": "grey",
            "verticalalignment": "baseline",
        },
    )
    if language == "en":
        info = (
            site
            + f"{origin}"
            + "  |  "
            + "$\it{Coordinates:}$"
            + f" {lat_0}"
            + "°N, "
            + f"{lon_0}°E  |  "
            + "$\it{Domain:}$"
            + f" {domain}  |  "
            + "$\it{Model:}$"
            + f" {model_name}  |  "
            + "$\it{Elevation: }$"
            + f" {elevation} {unit} ({y_type})"
        )
    else:
        info = (
            site
            + f"{origin}"
            + "  |  "
            + "$\it{Koordinaten:}$"
            + f" {lat_0}"
            + "°N, "
            + f"{lon_0}°E  |  "
            + "$\it{Domäne:}$"
            + f" {domain}  |  "
            + "$\it{Model:}$"
            + f" {model_name}  |  "
            + "$\it{Elevation: }$"
            + f" {elevation} {unit} ({y_type})"
        )

    ax.text(
        x=0.5,
        y=0.5,
        s=info,
        transform=ax.transAxes,
        va="center",
        ha="center",
    )
    return ax
