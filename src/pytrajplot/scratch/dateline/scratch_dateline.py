"""Compute dynamic domain & dynamic central longitude."""


# interesting link: https://stackoverflow.com/questions/13856123/setting-up-a-map-which-crosses-the-dateline-in-cartopy
# Standard library
# IMPORTS
import csv

# Third-party
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.function_base import flip
from numpy.lib.function_base import median


# FUNCTIONS
def _get_traj_dict(data, number_of_trajectories=None, traj_length=None) -> dict:
    # the variables which are pre-defined as 'None' are for compatibility purposes. this script can be used
    # as a stand-alone, or in the overall pytrajplot package

    if traj_length is None:
        number_of_trajectories = data["#trajectories"].iloc[0]
        traj_length = data["block_length"].iloc[0]

    traj_dict = {}
    max_lons, min_lons = [], []
    dateline_crossing_trajectories = []

    for traj_index in range(number_of_trajectories):
        bottom_index = traj_index * traj_length
        top_index = traj_index * traj_length + traj_length

        lon = data["lon"].iloc[bottom_index:top_index]

        max_lon = np.max(lon)
        min_lon = np.min(lon)

        max_lons.append(max_lon)
        min_lons.append(min_lon)

        if (max_lon > 0) and (min_lon < 0):
            lon = lon.values
            # check wheter this trajectory is dateline_crossing:
            # 1) check if the sign gets flipped
            sign_flip_indexes = np.where(np.sign(lon[:-1]) != np.sign(lon[1:]))[0] + 1

            # 2) check if the any of the sign_flips are relevant (i.e. not crossing the 0° longitude line)
            relevant_flip_indexes = (
                []
            )  # here, only flip indexes which correspond to dateline crossings, not 0°-line crossings
            for flip_index in sign_flip_indexes:
                if not -20 < lon[flip_index] < 20:
                    relevant_flip_indexes.append(flip_index)

            # if this list is not empty --> dateline got crossed
            if relevant_flip_indexes is not None:
                dateline_crossing_trajectories.append(traj_index)

        traj_dict["traj_" + str(traj_index)] = {
            "lon": data["lon"].iloc[bottom_index:top_index],
            "lat": data["lat"].iloc[bottom_index:top_index],
        }

    # since there is no cyclical behaviour with the latitude, these expansion boundaries can be derived from the
    # single latitude column, by looking at the min/max values
    latitude_expansion = [
        np.min(data["lat"]),
        np.max(data["lat"]),
    ]  # latitude_expansion = (lower_boundary, upper_boundary)

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
):
    """Check, wheter dateline gets crossed by *any* trajectory departing from this location.

    Args:
        lon (pandas series): one list, containing the longitude values for all trajectories

    Returns:
        cross_dateline (bool): true, if dateline gets crossed by any trajectory

    """
    max_lon, min_lon = np.max(lon), np.min(lon)
    cross_dateline = False

    lon = lon.values  # convert pandas series to array.
    # 1) check if the sign gets flipped
    sign_flip_indexes = np.where(np.sign(lon[:-1]) != np.sign(lon[1:]))[0] + 1

    if len(sign_flip_indexes) == 0:
        return cross_dateline, [min_lon, max_lon]

    else:
        # 2) check if the any of the sign_flips are relevant (i.e. not crossing the 0° longitude line)
        for flip_index in sign_flip_indexes:
            # print(f"@flip index {flip_index} the longitude is: {lon[flip_index]}")
            if not np.isnan(lon[flip_index]) and not np.isnan(lon[flip_index - 1]):
                if not (
                    -20 < lon[flip_index] < 20
                ):  # the dateline must have been crossed because the longitude value after the sign flip is not in the -20° - 20° longitude range.
                    return True, [None, None]

    return False, [min_lon, max_lon]


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
            print(f"Trajectory {traj_index} crosses the dateline or prime meridian.")
        # 1) extract longitude pandas series from date-line-crossing trajectory from trajectory dict
        lon = (traj_dict["traj_" + str(traj_index)]["lon"]).values

        # 2.1) check at which indexes, the signs of the longitude values flip
        sign_flip_indexes = np.where(np.sign(lon[:-1]) != np.sign(lon[1:]))[0] + 1
        # sign_flip_indexes = np.where(np.diff(np.sign(lon)) != 0)[0] + 1

        relevant_flip_indexes = (
            []
        )  # here, only flip indexes which correspond to dateline crossings, not 0°-line crossings

        # 3) check, which sign changes refer to east-west flip @ 0° and @ ±180°. the relevant_flip_indexes are the ones, that
        # refer to crossing of ±180°, NOT the 0°-lon line.
        for i, flip_index in enumerate(sign_flip_indexes):
            if not -20 < lon[flip_index] < 20:
                relevant_flip_indexes.append(flip_index)

        # ~~~~~~~~~~~~~~~~~~~~~~ NEW ~~~~~~~~~~~~~~~~~~~~~~ #
        for flip_index_position, flip_index in enumerate(relevant_flip_indexes):

            if verbose:
                print(
                    f"@ index {flip_index} the trajectory flips from: {lon[flip_index-1]} to {lon[flip_index]} (flip_index_position: {flip_index_position})"
                )

            value_after_flip = lon[flip_index]

            if (
                value_after_flip > 0
            ):  # the trajectory crossed from the west to the east over the dateline

                # if there is only one crossing of the dateline, all values after the crossing need to be adjusted
                if len(relevant_flip_indexes) == 1:
                    lon[flip_index : len(lon)] -= 360
                    break

                # if there are several crossings of the dateline (i.e. back&forth), only the values between consecutive crossings need to be adjusted
                # > if there are several crossings, and the loop has arrived at the last crossing, adapt the remaining values
                # > else, fill the values until the next dateline crossing
                elif flip_index_position % 2 == 0:
                    if flip_index_position == (len(relevant_flip_indexes) - 1):
                        lon[flip_index : len(lon)] -= 360
                        break

                    else:
                        lon[
                            flip_index : (
                                relevant_flip_indexes[flip_index_position + 1]
                            )
                        ] -= 360

            if (
                value_after_flip < 0
            ):  # the trajectory crossed from the east to the west over the dateline

                # if there is only one crossing of the dateline, all values after the crossing need to be adjusted
                if len(relevant_flip_indexes) == 1:
                    lon[flip_index : len(lon)] += 360
                    break

                # if there are several crossings of the dateline (i.e. back&forth), only the values between consecutive crossings need to be adjusted
                # > if there are several crossings, and the loop has arrived at the last crossing, adapt the remaining values
                # > else, fill the values until the next dateline crossing
                elif flip_index_position % 2 == 0:
                    if flip_index_position == (len(relevant_flip_indexes) - 1):
                        lon[flip_index : len(lon)] += 360
                        break

                    else:
                        lon[
                            flip_index : (
                                relevant_flip_indexes[flip_index_position + 1]
                            )
                        ] += 360
        # 4) append the maximum/minimum longitude to their respective list
        min_lons.append(np.min(lon))
        max_lons.append(np.max(lon))

    # 5) compute central longitude
    central_longitude = round(median([np.max(max_lons), np.min(min_lons)]))

    # 6) compute longitude expansion:

    # case: trajectory crosses the dateline from the west and ends in the east
    # europe is on the right side of the map --> thus, the right boundary should
    # be a lower eastern longitude value --> here, take the trajectories that did not
    # cross the dateline into consideration
    if central_longitude < 0:
        for traj_index, _ in enumerate(traj_dict):
            if traj_index not in dateline_crossing_trajectories:
                max_lons.append(np.max(traj_dict["traj_" + str(traj_index)]["lon"]))

        left_boundary = np.min(min_lons)
        right_boundary = np.max(max_lons)

    # case: trajectory crosses the dateline from the east and ends in the west
    if central_longitude > 0:  # TODO
        left_boundary = np.min(eastern_longitudes)
        right_boundary = np.max(max_lons)

    longitude_expansion = [left_boundary, right_boundary]

    if verbose:
        print(
            f"min lons:\t\t{min_lons}\nmax lons:\t\t{max_lons}\nCentral Longitude:\t{central_longitude}°\nlongitude expansion:\t\t{longitude_expansion}"
        )

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
    csv_files = [
        "src/pytrajplot/scratch/dateline/test_files/144-000B_df.csv",  # [0] trajectories cross from the west
        "src/pytrajplot/scratch/dateline/test_files/006-144F_df.csv",  # [1] trajectories cross from the east
        "src/pytrajplot/scratch/dateline/test_files/000-048F_df.csv",  # [2] trajectories don't cross dateline
        "src/pytrajplot/scratch/dateline/test_files/punggyeri_df.csv",  # [3] trajectories cross dateline several times
        "src/pytrajplot/scratch/dateline/test_files/basel_df.csv",  # [4] fix computation of dynamic domain
        "src/pytrajplot/scratch/dateline/test_files/zurich_df.csv",  # [5] fix computation of dynamic domain
        "src/pytrajplot/scratch/dateline/test_files/bagdad_df.csv",  # [6] fix computation of dynamic domain
        "src/pytrajplot/scratch/dateline/test_files/punggyeri_000-144F_df.csv",  # [7] trajectories cross dateline several times
        "src/pytrajplot/scratch/dateline/test_files/punggyeri_012-144F_df.csv",  # [8] trajectories cross dateline several times
        "src/pytrajplot/scratch/dateline/test_files/punggyeri_018-144F_df.csv",  # [9] trajectories cross dateline several times
    ]

    for csv_file in csv_files:
        data = pd.read_csv(csv_file)

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
