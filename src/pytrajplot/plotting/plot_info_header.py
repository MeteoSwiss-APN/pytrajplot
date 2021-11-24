"""Function, to assemble the plot info header."""

# Third-party
import matplotlib.pyplot as plt


def generate_info_header(plot_info, plot_data, domain, ax=None):
    ax = ax or plt.gca()
    ax.axis("off")

    # TODO: extract relevant parameters from plot_data for the header
    origin = plot_data["altitude_1"]["origin"]
    y_type = plot_data["altitude_1"]["y_type"]
    elevation = plot_data["altitude_1"]["traj_0"]["z"].iloc[0]
    lon_0 = plot_data["altitude_1"]["traj_0"]["lon"].iloc[0]
    lat_0 = plot_data["altitude_1"]["traj_0"]["lat"].iloc[0]
    model_name = plot_info["model_name"]

    if y_type == "hpa":
        unit = "hPA"
    else:
        unit = "m"

    start_time = plot_info["mbt"]

    title = f"TRAJECTORY from {origin} on: {start_time}"

    ax.set_title(
        title,
        fontdict={
            "fontsize": 20,
            "color": "grey",
            "verticalalignment": "baseline",
        },
    )
    # for pos in ['right', 'top', 'bottom', 'left']:
    #     ax.spines[pos].set_visible(False)
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)

    info = (
        "$\it{Release}$ $\it{Site}:}$ "
        + f"{origin}"
        + "  |  "
        + "$\it{Coordinates:}$"
        + " ("
        + f"{lat_0}"
        + "°N, "
        + f"{lon_0}"
        + "°E)  |  "
        + "$\it{Domain:}$"
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
