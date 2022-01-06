"""Compute dynamic domain & dynamic central longitude."""


# interesting link: https://stackoverflow.com/questions/13856123/setting-up-a-map-which-crosses-the-dateline-in-cartopy
# Third-party
# IMPORTS
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.function_base import median


# FUNCTIONS
def _get_traj_dict(data, number_of_trajectories=None, traj_length=None) -> dict:
    # number_of_rows = len(data)
    if traj_length is None:
        number_of_trajectories = data["#trajectories"].iloc[0]
        traj_length = data["block_length"].iloc[0]

    traj_dict = {}
    max_lons, min_lons = [], []

    for traj_index in range(number_of_trajectories):
        bottom_index = traj_index * traj_length
        top_index = traj_index * traj_length + traj_length
        max_lons.append(np.max(data["lon"].iloc[bottom_index:top_index]))
        min_lons.append(np.min(data["lon"].iloc[bottom_index:top_index]))
        traj_dict["traj_" + str(traj_index)] = {
            "lon": data["lon"].iloc[bottom_index:top_index],
            "lat": data["lat"].iloc[bottom_index:top_index],
        }

    # if the max longitude of a trajectory is greater than 175° (east), this trajectory is assumed to cross the dateline
    # (could be cross checked, by looking whether the same values occur if np.where(np.array(min_lons)<-175) contains the same indexes,
    # which at the same time correspond to the trajectory index)
    dateline_crossing_trajectories = np.where(np.array(max_lons) > 175)[0]

    # since there is no cyclical behaviour with the latitude, these expansion boundaries can be derived from the
    # single latitude column, by looking at the min/max values
    # latitude_expansion = (lower_boundary, upper_boundary)
    latitude_expansion = [np.min(data["lat"]), np.max(data["lat"])]

    # since all MeteoSwiss trajectories start in the eastern hemisphere - it is handy to have all eastern longitude values
    # in one compact list. i.e. to figure out how far the travel when figuring out the longitude expanse
    eastern_longitudes = pd.Series(data["lon"].where(data["lon"] > 0))

    return (
        traj_dict,
        dateline_crossing_trajectories,
        latitude_expansion,
        eastern_longitudes,
    )


def _check_dateline_crossing(
    lon: pd.Series,
) -> bool:
    """Check, wheter dateline gets crossed by *any* trajectory departing from this location.

    Args:
        lon (pandas series): one list, containing the longitude values for all trajectories

    Returns:
        cross_dateline (bool): true, if dateline gets crossed by any trajectory

    """
    max_lon, min_lon = np.max(lon), np.min(lon)
    far_east = 175 <= max_lon <= 180
    far_west = -175 >= min_lon >= -180
    cross_dateline = False

    if far_east and far_west:
        cross_dateline = True
        min_lon, max_lon = None, None

    return cross_dateline, [min_lon, max_lon]


def _get_central_longitude(
    traj_dict, dateline_crossing_trajectories, eastern_longitudes, verbose=False
):
    """Compute central longitude for dateline-crossing trajectories.

    Args:
        traj_dict (dict): Dictionary, containig all trajectories
        dateline_crossing_trajectories (list): List, containing the trajectory indexes, of dateline-crossing trajectories.

    Returns:
        # TODO
        [type]: [description]

    """
    # IF there are several trajectories crossing the dateline in the traj_dict - collect their longitude expansions in
    # these two lists. In the end the respective min/max values result in the overall longitude expansion.
    min_lons, max_lons = [], []

    # iterate through dateline-crossing trajcetories
    for traj_index in dateline_crossing_trajectories:
        if verbose:
            print(f"Trajectory {traj_index} crosses the dateline.")
        # 1) extract longitude pandas series from date-line-crossing trajectory from trajectory dict
        lon = (traj_dict["traj_" + str(traj_index)]["lon"]).values

        # 2) check at which indexes, the signs of the longitude values flip
        sign_flip_indexes = np.where(np.sign(lon[:-1]) != np.sign(lon[1:]))[0] + 1
        # sign_flip_indexes = np.where(np.diff(np.sign(lon)) != 0)[0] + 1
        relevant_flip_indexes = []

        # 3) check, which flips refer to east-west flip @ 0° and @ ±180°.
        # the first dateline crossing could refer to crossing the 0° line, following indexes refer to crossing
        # the dateline. only the first dateline crossing is necessary, thus continue after the 2nd flip index.
        tmp = False
        for i, flip_index in enumerate(sign_flip_indexes):
            if i < 2:
                if verbose:
                    print(
                        f"@ index {flip_index} the trajectory flips from: {lon[flip_index-1]} to {lon[flip_index]}"
                    )

                if (lon[flip_index] > 0) and (lon[flip_index] > 175):
                    if tmp is True:
                        if verbose:
                            print(f"tmp is True; -360 was not added")
                        break
                    # direction = 'west' # the trajectory crossed from the west to the east over the dateline
                    lon[flip_index : len(lon)] -= 360
                    tmp = True

                if (lon[flip_index] < 0) and (lon[flip_index] < -175):
                    if tmp is True:
                        if verbose:
                            print(f"tmp is True; +360 was not added")
                        break
                    # direction = 'east' # the trajectory crossed from the east to the west over the dateline
                    lon[flip_index : len(lon)] += 360
                    tmp = True
            else:
                break

        # 4) append the maximum/minimum longitude to their respective list
        # print(f"min_lon = {np.min(lon)}\nmax_lon = {np.max(lon)}")
        min_lons.append(np.min(lon))
        max_lons.append(np.max(lon))

    # 5) compute central longitude
    central_longitude = round(median([np.max(max_lons), np.min(min_lons)]))
    if verbose:
        print(
            f"min lons:\t\t{min_lons}\nmax lons:\t\t{max_lons}\nCentral Longitude:\t{central_longitude}°"
        )

    # 6) compute longitude expansion:
    # case: trajectory crosses the dateline from the east and ends in the west
    if central_longitude > 0:  # TODO
        left_boundary = np.min(eastern_longitudes)
        right_boundary = np.max(max_lons)
    if central_longitude < 0:
        left_boundary = np.min(min_lons)
        right_boundary = np.max(max_lons)

    longitude_expansion = [left_boundary, right_boundary]
    return central_longitude, longitude_expansion


def _analyse_trajectories(
    traj_dict,
    cross_dateline,
    dateline_crossing_trajectories,
    latitude_expansion,
    longitude_expansion,
    eastern_longitudes,
    verbose=False,
):
    if cross_dateline:
        central_lontigude, longitude_expansion = _get_central_longitude(
            traj_dict,
            dateline_crossing_trajectories,
            eastern_longitudes=eastern_longitudes,
        )

    else:
        central_lontigude = 0
        longitude_expansion = longitude_expansion

    dynamic_domain = longitude_expansion + latitude_expansion
    if verbose:
        print(
            f"latitude_expansion = {latitude_expansion}\nlongitude expansion = {longitude_expansion}"
        )
        print(f"dynamic domain: {dynamic_domain}")

    return central_lontigude, dynamic_domain


def _create_plot(traj_dict, central_longitude, dynamic_domain):

    # REMOVE HARDCODED CENTRAL LONGITUDE LATER!
    # central_longitude = 0

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_longitude))

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels, gl.right_labels = False, False
    ax.coastlines(resolution="110m")

    # ax.set_global()
    offset = 3
    ax.set_extent(
        [
            dynamic_domain[0] - offset,
            dynamic_domain[1] + offset,
            dynamic_domain[2] - offset,
            dynamic_domain[3] + offset,
        ],
        ccrs.PlateCarree(central_longitude=0),
    )

    # https://stackoverflow.com/questions/65086715/longitude-bounds-of-cartopy-crs-geodetic
    # https://duckduckgo.com/?t=ffab&q=numpy.mod()&ia=web
    for i, _ in enumerate(traj_dict):
        lon = traj_dict["traj_" + str(i)]["lon"]
        lat = traj_dict["traj_" + str(i)]["lat"]
        ax.plot(lon, lat, transform=ccrs.Geodetic())

    plt.title(f"plate carree w/ centr. lon: {central_longitude}")
    plt.savefig(
        f"src/pytrajplot/scratch/dateline/test_plots/central_longtitude_{central_longitude}.png"
    )
    plt.close()

    return


def main():
    # 1) load csv --> dataframe
    # data = pd.read_csv('src/pytrajplot/scratch/dateline/144-000B_df.csv') # trajectories cross from the west
    # data = pd.read_csv('src/pytrajplot/scratch/dateline/006-144F_df.csv') # trajectories cross from the east
    data = pd.read_csv(
        "src/pytrajplot/scratch/dateline/000-048F_df.csv"
    )  # trajectories don't cross dateline

    # 2) check if dateline gets crossed;
    # if the dateline does not get crossed, the min_lon value is the left boundary and the max_lon the right boundary
    cross_dateline, longitude_expansion = _check_dateline_crossing(lon=data["lon"])

    # 3) split lon/lat lists into corresponding trajectories. for each trajectory one key should be assigned.
    (
        traj_dict,
        dateline_crossing_trajectories,
        latitude_expansion,
        eastern_longitudes,
    ) = _get_traj_dict(data=data)

    # 4) compute central longitude dynamic domain if dateline does not get crossed
    central_longitude, dynamic_domain = _analyse_trajectories(
        traj_dict=traj_dict,
        cross_dateline=cross_dateline,
        dateline_crossing_trajectories=dateline_crossing_trajectories,
        latitude_expansion=latitude_expansion,
        longitude_expansion=longitude_expansion,
        eastern_longitudes=eastern_longitudes,
    )

    # 5) plot trajectories & save plot
    _create_plot(traj_dict, central_longitude, dynamic_domain)

    print(f"--- done.")


if __name__ == "__main__":
    main()
