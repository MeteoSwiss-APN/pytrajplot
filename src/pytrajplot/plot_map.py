"""Generate Map Plot Figure."""
# follwing this guide: https://earth-env-data-science.github.io/lectures/mapping_cartopy.html
# Standard library
import locale
import os

# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import ipdb


def create_coord_dict(altitude_levels):
    """Create dict of dicts to be filled with information for all trajectories to be plottet.

    Args:
        altitude_levels (int): Number of trajectory dicts (for each alt. level one dict)

    Returns:
        coord_dict (dict): Dict of dicts. For each altitude level, one dict is present in this dict. Each of those 'altitude dicts' contains the relevant information to plot the corresponding trajectory.

    """
    assert (
        altitude_levels <= 10
    ), "It is not possible, to generate altitude plots for more than 10 different starting altitudes."

    coord_dict = {}

    key_name = "altitude_"

    i = 1

    while i < altitude_levels + 1:
        altitude_dict = {
            "origin": None,
            "y_type": None,
            "alt_level": None,
            "subplot_index": None,
            "traj_0": {
                "lon": [],
                "lat": [],
                "time": [],
                "z_type": None,
                "alpha": 1,
            },  # main trajectory
            "traj_1": {
                "lon": [],
                "lat": [],
                "time": [],
                "z_type": None,
                "alpha": 0.5,
            },  # side trajectory 1
            "traj_2": {
                "lon": [],
                "lat": [],
                "time": [],
                "z_type": None,
                "alpha": 0.5,
            },  # side trajectory 2
            "traj_3": {
                "lon": [],
                "lat": [],
                "time": [],
                "z_type": None,
                "alpha": 0.5,
            },  # side trajectory 3
            "traj_4": {
                "lon": [],
                "lat": [],
                "time": [],
                "z_type": None,
                "alpha": 0.5,
            },  # side trajectory 4
        }
        coord_dict[key_name + str(i)] = altitude_dict
        i += 1

    return coord_dict


def add_features(ax):
    """Add grid/coastlines/borders/oceans/lakes and rivers to current map (ax).

    Args:
        ax (cartopy.mpl.geoaxes.GeoAxesSubplot): adding features to current map (axes instance)

    """
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="k",
        alpha=0.3,
        linestyle="-.",
    )  # define grid line properties
    gl.top_labels = False
    gl.right_labels = False

    ax.coastlines(resolution="10m")
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle="--")
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.RIVERS)
    # additional lakes & rivers on a smaller scale
    rivers_10m = cfeature.NaturalEarthFeature(
        "physical", "rivers_lake_centerlines", "10m"
    )
    ax.add_feature(rivers_10m, facecolor="None", edgecolor="lightblue", alpha=0.5)

    # ax.add_feature(cfeature.STATES)
    return


def crop_map(ax, domain, custom_domain_boundaries):
    """Crop map to given domain (i.e. centraleurope).

    Args:
        ax (cartopy.mpl.geoaxes.GeoAxesSubplot):    cropping current map (axes instance)
        domain (str):                               key for the domain_dict to read correct domain boundaries
        custom_domain_boundaries (list):            list, containing the domain for these specifc trajectories (created dynamically)

    Returns:
        domain_boundaries (list):                   [lat0,lat1,lon0,lon1]

    """
    # got these pre-defined domain boundaries from: https://github.com/MeteoSwiss-APN/oprtools/blob/master/dispersion/lib/get_domain.pro

    padding = 5  # padding on each side, for the dynamically created plots

    domain_dict = {
        "centraleurope": {"domain": [2, 18, 42, 52]},
        "ch": {"domain": [5.8, 10.6, 45.4, 48.2]},
        "alps": {"domain": [2, 14, 43, 50]},
        "europe": {"domain": [-10, 47, 35, 65]},
        "ch_hd": {"domain": [3.5, 12.6, 44.1, 49.4]},
        "dynamic": {
            "domain": [
                round(custom_domain_boundaries[0]) - padding,
                round(custom_domain_boundaries[1]) + padding,
                round(custom_domain_boundaries[2]) - padding,
                round(custom_domain_boundaries[3]) + padding,
            ]
        },
    }

    domain_boundaries = domain_dict[domain]["domain"]
    ax.set_extent(domain_boundaries, crs=ccrs.PlateCarree())

    return domain_boundaries


def is_visible(lat, lon, domain_boundaries, cross_dateline) -> bool:
    """Check if a point (city) is inside the domain.

    Args:
        lat (float):                latitude of city
        lon (float):                longitue of city
        domain_boundaries (list):   latitude & longitude range of domain

    Returns:
        bool:                       True if city is within domain boundaries, else false.

    """
    if cross_dateline:
        if lon < 0:
            lon = 360 - abs(lon)

    in_domain = (
        domain_boundaries[0] <= lon <= domain_boundaries[1]
        and domain_boundaries[2] <= lat <= domain_boundaries[3]
    )

    # print(f'lon/lat of city: ({lon}/{lat}) --> in domain: {in_domain}')

    if in_domain:
        return True
    else:
        return False


def is_of_interest(name, capital_type, population, domain, lon) -> bool:
    """Check if a city fulfils certain importance criteria.

    Args:
        name (str):         Name of city
        capital_type (str): primary -> country's capital, admin -> 1st level admin capital, minor -> lower-level admin capital
        population (int):   Population of city (used for filtering out smaller cities)
        domain (str):       Domain for map

    Returns:
        bool:               True if city is of interest, else false

    """
    # print(f'checking whether {name} is of interest')
    # is_capital = (capital_type == "primary") or (capital_type == "admin")
    is_capital = capital_type == "primary"

    # list of cities, that should be excluded for aesthetic reasons
    excluded_cities = [
        # East-, West-Europe and Asia
        "Incheon",
        "Duisburg",
        "Essen",
        "Dortmund",
        "San Marino",
        "Skopje",
        "Bratislava",
        "Pristina",
        "Bursa",
        "Yerevan",
        "Gaziantep",
        "Athens",
        "The Hague",
        "Tallinn",
        "Podgorica",
        "Ljubljana",
        "Voronezh",
        "Tunceli",
        "Sanliurfa",
        "Keren",
        "Massawa",
        "Elazig",
        "Adiyaman",
        "Erzincan",
        "Giresun",
        "Gumushane",
        "Ryanzan",
        "Luhansk",
        "New Delhi",
        "Manama",
        "Osaka",
        "Nagoya",
        "Tongshan",
        "Tianjin",
        "Shijiazhuang",
        "Heze",
        "Guangzhou",
        "Kolkata",
        "Thimphu",
        # United States & South America
        "Carson City",
        "Helena",
        "St. Paul",
        "Des Moines",
        "Salt Lake City",
        "Mexicali",
        "Hermosillo",
        "Little Rock",
        "Oklahoma City",
        "Jefferson City",
        "Boise",
        "Cheyenne",
        "Topeka",
        "Culiacan",
        "Ciudad Victoria",
        "Saltillo",
        "Durango",
        "Zacatecas",
        "San Luis Potosi",
        "Aguascalientes",
        "Guanajuato",
    ]

    is_excluded = name in excluded_cities

    if domain == "dynamic":
        is_capital = (capital_type == "primary") or (capital_type == "admin")

        if 40 <= lon <= 180:
            is_capital = (
                capital_type == "primary"
            )  # there are too many cities that meet this requirement in Asia.

        is_large = population > 8000000  # dynamic range often is global (or in Asia)

        return (is_capital or is_large) and not is_excluded

    if domain == "europe":
        is_large = population > 2000000
        return (is_capital or is_large) and not is_excluded

    else:
        is_large = population > 400000

    return (is_capital or is_large) and not is_excluded


def add_cities(ax, domain_boundaries, domain, cross_dateline):
    """Add cities to map.

    Args:
        ax (cartopy.mpl.geoaxes.GeoAxesSubplot):    adding cities to current map (axes instance)
        domain_boundaries (list):                   latitude & longitude range of domain
        domain (str):                               Domain for map
        cross_dateline (bool):                      Bool, to pass on to the is_visible function.

    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # IMPORTING POPULATED ARES FROM https://simplemaps.com/data/world-cities INSTEAD OF NATURAL EARTH
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    cities_df = pd.read_csv(
        "src/pytrajplot/cities/worldcities.csv"
    )  # len(cities_df) = 41001
    cities_df = (
        cities_df.dropna()
    )  # remove less important cities to reduce size of dataframe --> # len(cities_df) = 8695

    for i, row in cities_df.iterrows():
        city = row["city_ascii"]
        lon = row["lng"]
        lat = row["lat"]
        capital_type = row["capital"]
        population = row["population"]

        if is_visible(
            lat=lat,
            lon=lon,
            domain_boundaries=domain_boundaries,
            cross_dateline=cross_dateline,
        ) and is_of_interest(
            name=city,
            capital_type=capital_type,
            population=population,
            domain=domain,
            lon=lon,
        ):

            plt.scatter(
                x=lon,
                y=lat,
                marker="o",
                color="black",
                transform=ccrs.PlateCarree(),
            )

            ax.text(x=lon + 0.05, y=lat + 0.05, s=city, transform=ccrs.PlateCarree())


def add_time_interval_points(coord_dict, ax, i, linestyle):
    """Add time interval points to trajectories (6h interval).

    Args:
        coord_dict (dict):                          Dictionary containing the lan/lot data & other plot properties
        ax (cartopy.mpl.geoaxes.GeoAxesSubplot):    adding time marks to trajectories on current map
        i (int):                                    Altitude index. Only want to add legend for altitude one.
        linestyle (str):                            Defines colour of interval poins. (Same as corresponding trajectory)

    """
    lon_important, lat_important = retrieve_interval_points(
        coord_dict, altitude_index=i
    )

    # marker styles: https://matplotlib.org/stable/api/markers_api.html
    # add 6 hour interval points
    if i == 1:
        ax.scatter(
            lon_important,
            lat_important,
            marker="d",
            color=linestyle[:-1],
            label="6,12,...h",
            transform=ccrs.PlateCarree(),
        )
    else:
        ax.scatter(
            lon_important,
            lat_important,
            marker="d",
            color=linestyle[:-1],
            transform=ccrs.PlateCarree(),
        )


def retrieve_interval_points(coord_dict, altitude_index):
    """Extract only the interval points from the coord_dict and plot them.

    Args:
        coord_dict (dict):  Dictionary containing the lan/lot data & other plot properties
        i (int):            Altitude index.

    Returns:
        lon_important (pandas.core.series.Series): pandas list w/ interval point longitude values
        lat_important (pandas.core.series.Series): pandas list w/ interval point latitude values

    """
    lat_df_tmp = pd.DataFrame(
        coord_dict["altitude_" + str(altitude_index)]["traj_0"]["lat"].items()
    )
    lon_df_tmp = pd.DataFrame(
        coord_dict["altitude_" + str(altitude_index)]["traj_0"]["lon"].items()
    )
    time_df_tmp = pd.DataFrame(
        coord_dict["altitude_" + str(altitude_index)]["traj_0"]["time"].items()
    )

    # delete index columns
    del lon_df_tmp[0]
    del lat_df_tmp[0]
    del time_df_tmp[0]
    # rename remaining column
    lon_df = lon_df_tmp.rename(columns={1: "lon"})
    lat_df = lat_df_tmp.rename(columns={1: "lat"})
    time_df = time_df_tmp.rename(columns={1: "time"})
    # combine columns to one df
    comb_df = pd.concat([lat_df, lon_df, time_df], axis=1, join="inner")
    comb_df["time"] = comb_df["time"].astype(
        float
    )  # ensure correct type for time column (difference w/ HRES & COSMO)

    # extract position every 6 hours into important_points dataframe
    important_points_tmp = comb_df[comb_df["time"] % 6 == 0]
    important_points = important_points_tmp.iloc[
        1:
    ]  # remove start point --> want triangle there
    lon_important = important_points["lon"]
    lat_important = important_points["lat"]

    return lon_important, lat_important


def add_trajectories(
    coord_dict, side_traj, altitude_levels, ax, subplot_properties_dict
):
    """Add trajectories to map.

    Args:
        coord_dict (dict):                          Dictionary containing the lan/lot data & other plot properties
        side_traj (int):                            0/1 --> necessary for choosing the correct loop
        altitude_levels (int):                      # altitude levels
        ax (cartopy.mpl.geoaxes.GeoAxesSubplot):    adding trajectories to current map (axes instance)
        subplot_properties_dict (dict):             Dictionary containing the mapping between the altitude levels and colours. Uniform w/ altitude plots.

    """
    i = 1
    while i <= altitude_levels:
        alt_level = coord_dict["altitude_" + str(i)]["alt_level"]
        sub_index = int(coord_dict["altitude_" + str(i)]["subplot_index"])
        textstr = (
            str(coord_dict["altitude_" + str(i)]["alt_level"])
            + " "
            + coord_dict["altitude_" + str(i)]["y_type"]
        )

        # print(f'altitude_{i} = {alt_level} --> subplot {sub_index} (have {altitude_levels} alt levels/subplots)')

        if side_traj:
            traj_index = [0, 1, 2, 3, 4]

            for traj in traj_index:
                latitude = coord_dict["altitude_" + str(i)]["traj_" + str(traj)]["lat"]
                longitude = coord_dict["altitude_" + str(i)]["traj_" + str(traj)]["lon"]

                ystart = latitude.iloc[0]
                xstart = longitude.iloc[0]

                linestyle = subplot_properties_dict[sub_index]
                # print(f'linestyle for subplot {sub_index}: {linestyle}')
                alpha = coord_dict["altitude_" + str(i)]["traj_" + str(traj)]["alpha"]

                if (
                    coord_dict["altitude_" + str(i)]["traj_" + str(traj)]["alpha"] == 1
                ):  # only add legend & startpoint for the main trajectories
                    # plot main trajectory
                    ax.plot(
                        longitude,  # define x-axis
                        latitude,  # define y-axis
                        linestyle,  # define linestyle
                        alpha=alpha,  # define line opacity
                        label=textstr,
                        transform=ccrs.PlateCarree(),
                    )

                    add_time_interval_points(coord_dict, ax, i, linestyle)

                    # add start point triangle
                    ax.plot(
                        xstart,
                        ystart,
                        marker="^",
                        markersize=10,
                        markeredgecolor="red",
                        markerfacecolor="white",
                        transform=ccrs.PlateCarree(),
                    )

                else:
                    ax.plot(
                        longitude,  # define x-axis
                        latitude,  # define y-axis
                        linestyle,  # define linestyle
                        alpha=alpha,  # define line opacity
                        transform=ccrs.PlateCarree(),
                    )

        else:  # no side traj
            latitude = coord_dict["altitude_" + str(i)]["traj_0"]["lat"]
            longitude = coord_dict["altitude_" + str(i)]["traj_0"]["lon"]

            ystart = latitude.iloc[0]
            xstart = longitude.iloc[0]

            linestyle = subplot_properties_dict[sub_index]
            # print(f'linestyle for subplot {sub_index}: {linestyle}')
            alpha = coord_dict["altitude_" + str(i)]["traj_0"]["alpha"]

            # print(f"longitude = {longitude}")
            # print(f"latitude = {latitude}")

            # plot main trajectory
            ax.plot(
                longitude,  # define x-axis
                latitude,  # define y-axis
                linestyle,  # define linestyle
                alpha=alpha,  # define line opacity
                label=textstr,
                transform=ccrs.PlateCarree(),
            )

            add_time_interval_points(
                coord_dict=coord_dict, ax=ax, i=i, linestyle=linestyle
            )

            # add start point triangle
            ax.plot(
                xstart,
                ystart,
                marker="^",
                markersize=10,
                markeredgecolor="red",
                markerfacecolor="white",
                transform=ccrs.PlateCarree(),
            )

        i += 1
    return


def get_dynamic_domain(coord_dict, altitude_levels, side_traj):
    """Check wheter dateline is crossed or not and return dynamic domain boundaries.

    Args:
        trajectory_df (df):         Dataframe containing the lon/lat values of all trajectories

    Returns:
        central_longitude (float):  0° or 180°. If dateline is crossed, the central longitude is shifted (as well as all lon values)
        domain_boundaries (list):   [lon0,lon1,lat0,lat1]
        lon_df            (df):     single column dataframe containing the (shifted/unchanged) longitude values
        cross_dateline    (bool):   bool, to remember if the dateline was crossed for a given trajectory

    """
    coord_dict_tmp = coord_dict.copy()

    if side_traj:
        max_traj_index = 4
    else:
        max_traj_index = 0

    alt_index = 1

    lat_mins = (
        []
    )  #  latitude minimum values for all trajectories starting at this origin (bottom/south)
    lat_maxs = (
        []
    )  #  latitude maximum values for all trajectories starting at this origin (top/north)
    lon_mins = (
        []
    )  # longitude minimum values for all trajectories starting at this origin (left/west)
    lon_maxs = (
        []
    )  # longitude maximum values for all trajectories starting at this origin (right/east)

    while alt_index <= altitude_levels:
        traj_index = 0
        while traj_index <= max_traj_index:
            lat_mins.append(
                np.min(
                    coord_dict_tmp["altitude_" + str(alt_index)][
                        "traj_" + str(traj_index)
                    ]["lat"]
                )
            )
            lat_maxs.append(
                np.max(
                    coord_dict_tmp["altitude_" + str(alt_index)][
                        "traj_" + str(traj_index)
                    ]["lat"]
                )
            )
            lon_mins.append(
                np.min(
                    coord_dict_tmp["altitude_" + str(alt_index)][
                        "traj_" + str(traj_index)
                    ]["lon"]
                )
            )
            lon_maxs.append(
                np.max(
                    coord_dict_tmp["altitude_" + str(alt_index)][
                        "traj_" + str(traj_index)
                    ]["lon"]
                )
            )
            traj_index += 1
        alt_index += 1

    lower_boundary = np.min(lat_mins)
    upper_boundary = np.max(lat_maxs)

    # print(f'lon_min: {np.min(lon_mins)}')
    # print(f'lon_max: {np.max(lon_maxs)}')
    # print(f'far east: {150 <= np.max(lon_maxs) <= 180}')
    # print(f'far west: {-180 <= np.min(lon_mins) <= -150}')

    # check dateline crossing
    far_east = 150 <= np.max(lon_maxs) <= 180
    far_west = -180 <= np.min(lon_mins) <= -150

    if far_east and far_west:  # trajectory *must* cross dateline somehow
        central_longitude = 180

        # reset altitude index
        alt_index = 1
        while alt_index <= altitude_levels:
            traj_index = 0
            while traj_index <= max_traj_index:
                lon = coord_dict_tmp["altitude_" + str(alt_index)][
                    "traj_" + str(traj_index)
                ]["lon"]
                # extract western & eastern points
                lon_west = np.where(lon < 0, lon, np.NaN)
                lon_east = np.where(lon >= 0, lon, np.NaN)

                # remove NaN values (not actually necessary)
                lon_west = lon_west[np.logical_not(np.isnan(lon_west))]
                lon_east = lon_east[np.logical_not(np.isnan(lon_east))]

                left_boundary = np.min(lon_east)  # least eastern point

                if lon_west.size == 0:
                    right_boundary = np.max(lon_east)  # most eastern point

                else:
                    right_boundary = central_longitude + (
                        180 - abs(np.max(lon_west))
                    )  # least western point

                # print(f'minimum occuring in the lon array for this trajectory ({alt_index}, {traj_index}):\t{np.min(lon)}\t(maximum: {np.max(lon)})')
                # the lon-data must be adapted to the shifted projection
                # lon is a dataframe !
                if np.min(lon) < 0:
                    i = 0
                    while i < len(lon):
                        if lon.iloc[i] < 0:
                            lon.iloc[i] = central_longitude + (180 - abs(lon.iloc[i]))
                            # make the lon data compatible with the shifted projection
                        i += 1

                traj_index += 1
            alt_index += 1

        domain_boundaries = [
            left_boundary,
            right_boundary,
            lower_boundary,
            upper_boundary,
        ]
        cross_dateline = True
        return central_longitude, domain_boundaries, coord_dict_tmp, cross_dateline

    else:
        right_boundary = np.max(lon_maxs)  # most eastern point
        left_boundary = np.min(lon_mins)  # most western point
        central_longitude = 0
        domain_boundaries = [
            left_boundary,
            right_boundary,
            lower_boundary,
            upper_boundary,
        ]
        cross_dateline = False
        return central_longitude, domain_boundaries, coord_dict_tmp, cross_dateline

    # print(f'Latitude (y) Range: \t{lower_boundary}°\t\t-\t{upper_boundary}° ')
    # print(f'Longitude (x) Range:\t{left_boundary}°\t\t-\t{right_boundary}° ')


def plot_map(trajectory_dict, separator, output_dir, domains, language):
    """Iterate through the trajectory dict and, for each key, generate map plot.

    Args:
        trajectory_dict (dict): Trajectory dictionary. Each key contains a dataframe with all relevant information for the plots.
        separator (str):        Separator string between main- & side-trajectories.
        output_dir (str):       Path to directory, where plots should be saved.
        domains (list)):        List of possible domains. Iterates through this list and automatically plots all domains.
        language (str):         Language in plots (en/ge).

    """
    for key in trajectory_dict:  # iterate through the trajectory dict
        # print(f"--- defining trajectory plot properties for {key}")

        coord_dict = create_coord_dict(
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

        row_index = 0  # corresponds to row from start file
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
                    coord_dict["altitude_" + str(alt_index)]["origin"] = origin
                    coord_dict["altitude_" + str(alt_index)][
                        "subplot_index"
                    ] = subplot_index
                    # print(trajectory_df['hsurf'][lower_row:upper_row])

                coord_dict["altitude_" + str(alt_index)]["y_type"] = trajectory_df[
                    "z_type"
                ][lower_row]
                coord_dict["altitude_" + str(alt_index)]["alt_level"] = trajectory_df[
                    "start_altitude"
                ][lower_row]

                coord_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                    "lon"
                ] = trajectory_df["lon"][lower_row:upper_row]

                # coord_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                #     "lon"
                # ] = lon_df[lower_row:upper_row]

                coord_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                    "lat"
                ] = trajectory_df["lat"][lower_row:upper_row]

                coord_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                    "time"
                ] = trajectory_df["time"][lower_row:upper_row]

                coord_dict["altitude_" + str(alt_index)]["traj_" + str(traj_index)][
                    "z_type"
                ] = trajectory_df["z_type"][lower_row]

                row_index += 1
                traj_index += 1

                if traj_index == 5:
                    traj_index = 0
                    alt_index += 1

                if alt_index > altitude_levels:
                    alt_index = 1

                    for domain in domains:
                        generate_map_plot(
                            coord_dict=coord_dict,
                            side_traj=side_traj,
                            altitude_levels=altitude_levels,
                            domain=domain,
                            output_dir=output_dir,
                            language=language,
                            key=key,
                        )

            # complete the non-side-trajectory case, after having completed the side trajectory case
            else:
                # print(f'row_index = {row_index} corresponds to origin {origin}')
                coord_dict["altitude_" + str(alt_index)]["origin"] = origin
                coord_dict["altitude_" + str(alt_index)][
                    "subplot_index"
                ] = subplot_index
                coord_dict["altitude_" + str(alt_index)]["y_type"] = trajectory_df[
                    "z_type"
                ][lower_row]
                coord_dict["altitude_" + str(alt_index)]["alt_level"] = trajectory_df[
                    "start_altitude"
                ][lower_row]

                coord_dict["altitude_" + str(alt_index)]["traj_0"][
                    "lon"
                ] = trajectory_df["lon"][lower_row:upper_row]

                # fill the longitude column in the coord_dict using the lon_df
                # coord_dict["altitude_" + str(alt_index)]["traj_0"]["lon"] = lon_df[
                #     lower_row:upper_row
                # ]

                coord_dict["altitude_" + str(alt_index)]["traj_0"][
                    "lat"
                ] = trajectory_df["lat"][lower_row:upper_row]

                coord_dict["altitude_" + str(alt_index)]["traj_0"][
                    "time"
                ] = trajectory_df["time"][lower_row:upper_row]

                # since there are no side-traj, these keys remain empty
                coord_dict["altitude_" + str(alt_index)]["traj_1"]["lon"] = []
                coord_dict["altitude_" + str(alt_index)]["traj_1"]["lat"] = []
                coord_dict["altitude_" + str(alt_index)]["traj_2"]["lon"] = []
                coord_dict["altitude_" + str(alt_index)]["traj_2"]["lat"] = []
                coord_dict["altitude_" + str(alt_index)]["traj_3"]["lon"] = []
                coord_dict["altitude_" + str(alt_index)]["traj_3"]["lat"] = []
                coord_dict["altitude_" + str(alt_index)]["traj_4"]["lon"] = []
                coord_dict["altitude_" + str(alt_index)]["traj_4"]["lat"] = []

                coord_dict["altitude_" + str(alt_index)]["traj_0"][
                    "z_type"
                ] = trajectory_df["z_type"][lower_row]
                coord_dict["altitude_" + str(alt_index)]["traj_1"]["z_type"] = None
                coord_dict["altitude_" + str(alt_index)]["traj_2"]["z_type"] = None
                coord_dict["altitude_" + str(alt_index)]["traj_3"]["z_type"] = None
                coord_dict["altitude_" + str(alt_index)]["traj_4"]["z_type"] = None

                row_index += 1
                alt_index += 1
                if alt_index > altitude_levels:
                    alt_index = 1
                    for domain in domains:
                        generate_map_plot(
                            coord_dict=coord_dict,
                            side_traj=side_traj,
                            altitude_levels=altitude_levels,
                            domain=domain,
                            output_dir=output_dir,
                            language=language,
                            key=key,
                        )
    return


def generate_map_plot(
    coord_dict,
    side_traj,
    altitude_levels,
    domain,
    output_dir,
    language,
    key,
):
    """Generate map plot. fig & ax are defined here, as well as the plots are being saved.

    Args:
        coord_dict                  (dict):   Dictionary containing the lan/lot data & other plot properties
        side_traj                   (int):    0/1 --> necessary for choosing the correct loop
        altitude_levels             (int):    # altitude levels
        domain                      (str):    Domain for map
        output_dir                  (str):    Path to directory where the plots should be saved.
        language                    (str):    Language in plots (en/ge).
        key                         (str):    Key of start- & trajectory file. Necessary to create a corresponding directory in the output directory.
        central_longitude           (float):  0° or 180°. If dateline is crossed, the central longitude is shifted (as well as all lon values)
        custom_domain_boundaries    (list):   [lon0,lon1,lat0,lat1]

    """
    origin = coord_dict["altitude_1"]["origin"]

    print(f"--- {key} / {origin}\t --> {domain}")

    if domain == "dynamic":
        (
            central_longitude,
            custom_domain_boundaries,
            coord_dict_new,
            cross_dateline,
        ) = get_dynamic_domain(
            coord_dict, altitude_levels=altitude_levels, side_traj=side_traj
        )
        coord_dict = coord_dict_new

    else:
        central_longitude = 0
        custom_domain_boundaries = [0, 0, 0, 0]
        cross_dateline = False

    if coord_dict["altitude_1"]["y_type"] == "hpa":
        case = "HRES"
        projection = ccrs.PlateCarree(central_longitude=central_longitude)
    else:
        case = "COSMO"
        projection = ccrs.RotatedPole(
            pole_longitude=-170, pole_latitude=43
        )  # define rotation of COSMO model

    fig = plt.figure(figsize=(12, 8), constrained_layout=False)
    ax = plt.axes(projection=projection, frameon=False)
    ax.set_aspect(
        "auto"
    )  # skaliert die karte s.d. dass Bildformat von fig & axes übereinstimmen
    add_features(ax=ax)

    domain_boundaries = crop_map(
        ax=ax, domain=domain, custom_domain_boundaries=custom_domain_boundaries
    )  # sets extent of map

    # if the start point of the trajectories is not within the domain boundaries (i.e. Teheran is certainly not in Switzerland or even Europe), this plot can be skipped
    if not is_visible(
        lat=coord_dict["altitude_1"]["traj_0"]["lat"].iloc[0],
        lon=coord_dict["altitude_1"]["traj_0"]["lon"].iloc[0],
        domain_boundaries=domain_boundaries,
        cross_dateline=False,
    ):
        return

    add_cities(
        ax=ax,
        domain_boundaries=domain_boundaries,
        domain=domain,
        cross_dateline=cross_dateline,
    )

    subplot_properties_dict = {
        0: "k-",
        1: "g-",
        2: "b-",
        3: "r-",
        4: "c-",
        5: "m-",
        6: "y-",
        7: "deepskyblue-",
        8: "crimson-",
        9: "lightgreen-",
    }

    add_trajectories(
        coord_dict=coord_dict,
        side_traj=side_traj,
        altitude_levels=altitude_levels,
        ax=ax,
        subplot_properties_dict=subplot_properties_dict,
    )

    ax.legend()  # add legend

    title = True
    if title:
        if language == "en":
            locale.setlocale(locale.LC_ALL, "en_GB")
            fig.suptitle("Air trajectories originating from " + origin)
        if language == "de":
            fig.suptitle("Luft-Trajektorien Karte für " + origin)

    outpath = os.getcwd() + "/" + output_dir + "/plots/" + key + "/"
    os.makedirs(
        outpath, exist_ok=True
    )  # create plot folder if it doesn't already exist
    plt.savefig(outpath + origin + "_" + domain + ".png", dpi=150)
    plt.close(fig)
    return
