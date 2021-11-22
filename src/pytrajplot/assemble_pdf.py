"""Assemble figures in PDF."""
# Third-party
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt


def combine_dicts(alt_plot_dict, map_plot_dict) -> dict:
    """Combine dicts, containing the axes of the altitude and the map plot in one dict.

    Args:
        alt_plot_dict (dict): Dict, containing vor each location the altitude plot axes (& figure)
        map_plot_dict (dict): Dict, containing for each domain/start location the map plot axes instances (& figure)

    Returns:
        axes (dict:) Dict, containing the combined content of the aforementioned dicts. For each key (i.e. start location) there are all necessary axes instances to create the output pdf

    """
    axes_dict = {}
    for key in alt_plot_dict:
        axes_dict[key] = {}
        for map_key in map_plot_dict:
            if map_key.startswith(key):
                axes_dict[key].update(map_plot_dict[map_key])
                axes_dict[key].update(alt_plot_dict[key])
    return axes_dict


def get_info_axes():
    fig = plt.figure(figsize=(12, 8), constrained_layout=False)
    ax = plt.axes()
    ax.title.set_text("Info Dummy")
    # print(f'AltPlt: fig={fig}, type(fig)={type(fig)}, ax={ax}, type(ax)={type(ax)}')
    # plt.savefig('src/pytrajplot/scratch/map_dummy.png')
    return ax


def iterate_axes_dict(axes_dict, domains):
    info_ax = get_info_axes()
    for origin in axes_dict:
        alt_ax = axes_dict[origin]["altitude_axes"]["alt_axes"]
        # alt_ax = alt_ax[0]
        for domain in domains:
            map_ax = axes_dict[origin][domain]["map_axes"]
            map_fig = axes_dict[origin][domain]["map_fig"]
            _assemble_pdf(
                info_ax=info_ax,
                map_ax=map_ax,
                alt_ax=alt_ax,
                domain=domain,
                origin=origin,
                map_fig=map_fig,
            )

    return


# 4. somehow arrange all plots in one pdf
def _assemble_pdf(info_ax, map_ax, alt_ax, domain, origin, map_fig):
    print(f"--- assembling PDF for {origin}/{domain}")
    print(f"map ax: {map_ax}, alt_ax = {alt_ax}, info_ax = {info_ax}")

    fig = plt.figure()
    fig2 = map_fig

    # create grid spec oject
    grid_specification = gs.GridSpec(nrows=4, ncols=2)

    # axis handle for plot 1 (map)
    info_ax = plt.subplot(grid_specification[0, 0])
    # info_ax.text(x=0.5, y=0.5, s='info', va='center', ha='center')

    # axis handle for plot 1 (map)
    map_ax = plt.subplot(grid_specification[1:, 0])
    # map_ax.text(x=0.5, y=0.5, s='map', va='center', ha='center')

    # axis handle for alt plot 1
    for tmp, tmp_ax in enumerate(
        alt_ax
    ):  # tmp -> index of subplot/ax; tmp_ax -> AxesSubplot
        tmp_ax = plt.subplot(grid_specification[tmp, 1])
        # tmp_ax.text(x=0.5, y=0.5, s='alt'+str(tmp), va='center', ha='center')

    # plt.suptitle("Test Gridspec + Subplot")
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig("src/pytrajplot/scratch/pdfs/" + domain + "_" + origin + ".pdf")
    plt.close(fig)
    return


def assemble_pdf(altitude_axes, map_axes, domains):
    print("--- assembling PDF")
    # 1. combine dicts
    axes_dict = combine_dicts(alt_plot_dict=altitude_axes, map_plot_dict=map_axes)
    iterate_axes_dict(axes_dict=axes_dict, domains=domains)
