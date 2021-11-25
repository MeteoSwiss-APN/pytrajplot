"""Function, to assemble the plot info header."""

# Third-party
import matplotlib.pyplot as plt


def generate_info_header(language, plot_info, plot_data, domain, ax=None):
    """Summary - First line should end with a period.

    Args:
        language
                            str        language for plot annotations
        plot_info
                            dict       Dict w/ information for info header (taken from info file)
        plot_data
                            dict       Dict w/ information for info header (taken from start/traj files)
        domain
                            str        domain of current plot

        ax ([Axes], optional): Axes to plot the info header on. Defaults to None.

    Returns:
        ax ([Axes], optional): Axes w/ info header.

    """
    ax = ax or plt.gca()
    ax.axis("off")

    # TODO: extract relevant parameters from plot_data for the header
    origin = plot_data["altitude_1"]["origin"]
    y_type = plot_data["altitude_1"]["y_type"]
    lon_0 = plot_data["altitude_1"]["traj_0"]["lon"].iloc[0]
    lat_0 = plot_data["altitude_1"]["traj_0"]["lat"].iloc[0]
    model_name = plot_info["model_name"]
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
        title = f"TRAJECTORY from {origin} on: {start_time}"
    else:
        title = f"TAJEKTORIE von {origin} am: {start_time}"

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
            "$\it{Release}$ $\it{Site}:}$ "
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
            "$\it{Ursprungsort}$: "
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
        # bbox=dict(facecolor='red', alpha=0.5),
        # fontsize='xx-large'
    )
    return ax
