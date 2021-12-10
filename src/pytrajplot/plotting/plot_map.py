"""Generate Map Plot Figure."""

# Standard library
from pathlib import Path

# Third-party
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_coord_dict(altitude_levels):
    """Create dict of dicts to be filled with information for all trajectories to be plottet.

    Args:
        altitude_levels     int        Number of trajectory dicts (for each alt. level one dict)

    Returns:
        coord_dict          dict       Dict of dicts. For each altitude level, one dict is present in this dict. Each of those 'altitude dicts' contains the relevant information to plot the corresponding trajectory.

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
        ax:                 Axes       Current map to add features to

    """
    # point cartopy to the folder containing the shapefiles for the features on the map
    earth_data_path = Path(Path.home(), "pytrajplot/src/pytrajplot/resources/")
    assert (
        earth_data_path.exists()
    ), f"The natural earth data could not be found at {earth_data_path}"
    # earth_data_path = str(earth_data_path)
    cartopy.config["pre_existing_data_dir"] = earth_data_path
    cartopy.config["data_dir"] = earth_data_path

    # add grid & labels to map
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="k",
        alpha=0.3,
        linestyle="-.",
        # rasterized=True,
    )  # define grid line properties
    gl.top_labels = False
    gl.right_labels = False

    # THESE FEATURES SHOULD BE GENERATED USING FILES WHICH HAVE BEEN DOWNLOADED RATHER THAN DOWNLOAD THEM EACH TIME
    ax.add_feature(cfeature.LAND, rasterized=True, color="#FFFAF0")
    ax.add_feature(cfeature.COASTLINE, alpha=0.5, rasterized=True)
    ax.add_feature(cfeature.BORDERS, linestyle="--", alpha=0.5, rasterized=True)
    ax.add_feature(cfeature.OCEAN, rasterized=True)
    ax.add_feature(cfeature.LAKES, rasterized=True)
    ax.add_feature(cfeature.RIVERS, rasterized=True)
    ax.add_feature(
        cartopy.feature.NaturalEarthFeature(
            category="physical",
            name="lakes_europe",
            scale="10m",
            rasterized=True,
        ),
        rasterized=True,
        color="#97b6e1",
    )

    return


def crop_map(ax, domain, custom_domain_boundaries):
    """Crop map to given domain (i.e. centraleurope).

    Args:
        ax                  Axes       current map to crop
        domain              str        key for the domain_dict to retrieve correct domain boundaries
        custom_domain_boundaries
                            list       list, containing the domain for these specifc trajectories (created dynamically)

    Returns:
        domain_boundaries   list       [lat0,lat1,lon0,lon1]

    """
    padding = 5  # padding on each side, for the dynamically created plots

    domain_dict = {
        "centraleurope": {"domain": [2, 18, 42.5, 51.5]},  # optimised boundaries
        # "centraleurope": {"domain": [2, 18, 42, 52]},     # original boundaries
        "ch": {"domain": [5.8, 10.6, 45.4, 48.2]},  # optimised boundaries
        "alps": {"domain": [0.7, 16.5, 42.3, 50]},  # optimised boundaries
        # "europe": {"domain": [-12.5, 50.5, 35, 65]},      # optimised boundaries
        "europe": {"domain": [-10, 47, 35, 65]},  # original boundaries
        "ch_hd": {"domain": [2.8, 13.2, 44.1, 49.4]},  # optimised boundaries
        # "ch_hd": {"domain": [3.5, 12.6, 44.1, 49.4]},     # original boundaries
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
        lat                 float      latitude of city
        lon                 float      longitude of city
        domain_boundaries   list       lon/lat range of domain
        cross_dateline      bool       if cross_dateline --> western lon values need to be shifted

    Returns:
                            bool       True if city is within domain boundaries, else false.

    """
    if cross_dateline:
        if lon < 0:
            lon = 360 - abs(lon)

    in_domain = (
        domain_boundaries[0] <= float(lon) <= domain_boundaries[1]
        and domain_boundaries[2] <= float(lat) <= domain_boundaries[3]
    )

    if in_domain:
        return True
    else:
        return False


def is_of_interest(name, capital_type, population, domain, lon) -> bool:
    """Check if a city fulfils certain importance criteria.

    Args:
        name                str        Name of city
        capital_type        str        primary/admin/minor (label for "relevance" of city)
        population          int        Population of city (there are conditions based on population)
        domain              str        Map domain. Different domains have different conditions to determine interest.
        lon                 float      Longitude of city  (there are conditions based on longitude)

    Returns:
                            bool       True if city is of interest, else false

    """
    if domain == "dynamic":
        if 0 <= lon <= 40:  # 0°E - 40°E (mainly Europe)
            is_capital = capital_type == "primary"
            if capital_type == "admin":
                if population > 5000000:
                    is_capital = True
            is_large = population > 10000000

        if 40 <= lon <= 180:  # 40°E - 180°E (mainly Asia)
            is_capital = capital_type == "primary"
            if capital_type == "admin":
                if population > 10000000:
                    is_capital = True
            is_large = population > 12000000

        if -40 <= lon < 0:  # 40° W to 0° E/W (mainly Atlantic)
            is_capital = capital_type == "primary"
            if capital_type == "admin":
                if population > 2500000:
                    is_capital = True
            is_large = population > 3000000

        if -180 <= lon < -40:  # 180° W to 40° W (mainly American Continent)
            is_capital = capital_type == "primary"
            if capital_type == "admin":
                if population > 800000:
                    is_capital = True
            is_large = population > 1100000

        excluded_cities = [
            "Casablanca",
            "Fes",
            "Hartford",
            "Providence",
            "Andorra La Vella",
            "Indiana",
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
            "Leon de los Aldama",
            "Wroclaw",
            "Rotterdam",
            "Indianapolis",
            "Raleigh",
        ]
        is_excluded = name in excluded_cities
        return (is_capital or is_large) and not is_excluded

    is_capital = capital_type == "primary"

    if domain == "europe":
        is_large = population > 2000000
        excluded_cities = [
            "Tbilisi",
            "Yerevan",
            "Ljubljana",
            "Bratislava",
            "San Marino",
            "Podgorica",
            "Pristina",
            "Helsinki",
            "The Hague",
            "Andorra la Vella",
            "Athens",
            "Gaziantep",
            "Tallinn",
            "Bursa",
            "Adana",
            "Skopje",
            "Vaduz",
            "Konya",
        ]
        is_excluded = name in excluded_cities
        return (is_capital or is_large) and not is_excluded

    if domain == "centraleurope":
        is_large = population > 500000
        excluded_cities = [
            "Bratislava",
            "San Marino",
            "Nice",
            "Wroclaw",
            "Essen",
            "Dortmund",
            "Rotterdam",
        ]
        is_excluded = name in excluded_cities
        return (is_capital or is_large) and not is_excluded

    if domain == "ch":
        is_large = population > 200000
        if capital_type == "admin":
            is_capital = True
        excluded_cities = [
            "Neuchatel",
            "Stans",
            "Sarnen",
            "Schwyz",
            "Triesenberg",
            "Schaan",
            "Delemont",
            "Liestal",
            "Planken",
            "Ruggell",
            "Bregenz",
            "Eschen",
            "Schellenberg",
            "Gamprin",
            "Mauren",
        ]
        is_excluded = name in excluded_cities
        return (is_capital or is_large) and not is_excluded

    if domain == "ch_hd":
        is_large = population > 200000
        if capital_type == "admin":
            if population > 50000:
                is_capital = True
        excluded_cities = [
            "Metz",
            "Freiburg im Breisgau",
            "Augsburg",
            "Stuttgart",
            "Sankt Gallen",
            "Venice",
            "Mulhouse",
            "Karlsruhe",
            "Padova",
        ]
        is_excluded = name in excluded_cities
        return (is_capital or is_large) and not is_excluded

    if domain == "alps":
        is_large = population > 200000
        if capital_type == "admin":
            if population > 50000:
                is_capital = True
        excluded_cities = [
            "Metz",
            "Freiburg im Breisgau",
            "Augsburg",
            "Stuttgart",
            "Sankt Gallen",
            "Venice",
            "Mulhouse",
            "Karlsruhe",
            "Padova",
            "Plzen",
            "Mainz",
            "Mannheim",
            "Salzburg",
            "Trieste",
            "Ancona",
            "San Marino",
            "Nice",
            "Lucerne",
            "Maribor",
            "Kranj",
            "L'Aquila",
        ]
        is_excluded = name in excluded_cities
        return (is_capital or is_large) and not is_excluded


def add_cities(ax, domain_boundaries, domain, cross_dateline):
    """Add cities to map.

    Args:
        ax:                 Axes       current map to crop
        domain_boundaries:  list       lon/lat range of domain
        domain:             str        Map domain. Different domains have different conditions to determine interest.
        cross_dateline:     bool       if cross_dateline --> western lon values need to be shifted

    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # IMPORTING POPULATED ARES FROM https://simplemaps.com/data/world-cities INSTEAD OF NATURAL EARTH
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    cities_data_path = Path(Path.home(), "pytrajplot/src/pytrajplot/resources/cities/")
    assert (
        cities_data_path.exists()
    ), f"The natural earth data could not be found at {cities_data_path}"
    cities_df = pd.read_csv(Path(cities_data_path, "worldcities.csv"))

    # remove less important cities to reduce size of dataframe (from 41001 rows to 8695)
    cities_df = cities_df.dropna()

    add_w_town = True

    if add_w_town:
        if not cross_dateline:
            # add Weesen to maps
            ax.scatter(
                x=9.108376385221725,
                y=47.1361694653364,
                marker="1",
                color="grey",
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )
            ax.text(
                x=9.108376385221725 + 0.01,
                y=47.1361694653364 + 0.01,
                s="W-Town",
                color="grey",
                fontsize=5,
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )

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

            if cross_dateline:
                if lon < 0:
                    lon = 360 - abs(lon)

            ax.scatter(
                x=lon,
                y=lat,
                s=2,
                marker="o",
                facecolors="k",
                edgecolors="k",
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )

            text_shift = 0.05

            if domain == "ch":
                text_shift = 0.01

            ax.text(
                x=lon + text_shift,
                y=lat + text_shift,
                s=city,
                fontsize=8,
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )


def add_time_interval_points(coord_dict, ax, i, linestyle):
    """Summary - First line should end with a period.

    Args:
        coord_dict:         dict       containing the lan/lot data & other plot properties
        ax:                 Axes       current map to crop
        i:                  int        Altitude index. Only want to add legend for altitude one.
        linestyle:          str        Defines colour of interval poins (same as corresponding trajectory)

    """
    lon_important, lat_important = retrieve_interval_points(
        coord_dict, altitude_index=i
    )

    # add 6 hour interval points
    if i == 1:
        ax.scatter(
            lon_important,
            lat_important,
            marker="d",
            color=linestyle[:-1],
            label="6,12,...h",
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )
    else:
        ax.scatter(
            lon_important,
            lat_important,
            marker="d",
            color=linestyle[:-1],
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )


def retrieve_interval_points(coord_dict, altitude_index):
    """Extract the interval points from coord_dict add them to ax.

    Args:
        coord_dict          dict       containing the lan/lot data & other plot properties
        altitude_index      int        Altitude index. Only want to add legend for altitude one.

    Returns:
        lon_important       series     pandas list w/ interval point longitude values
        lat_important       series     pandas list w/ interval point latitude values

    """
    # create temporary dataframes
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
        coord_dict:                    dict       containing the lan/lot data & other plot properties
        side_traj:                     int        0/1 --> necessary for choosing the correct loop
        altitude_levels:               int        # altitude levels
        ax:                            Axes       current map to crop
        subplot_properties_dict:       dict       Dictionary containing the mapping between the altitude levels and colours. Uniform w/ altitude plots.

    """
    i = 1
    while i <= altitude_levels:
        sub_index = int(coord_dict["altitude_" + str(i)]["subplot_index"])
        textstr = (
            str(coord_dict["altitude_" + str(i)]["alt_level"])
            + " "
            + coord_dict["altitude_" + str(i)]["y_type"]
        )

        if side_traj:
            traj_index = [0, 1, 2, 3, 4]

            for traj in traj_index:
                latitude = coord_dict["altitude_" + str(i)]["traj_" + str(traj)]["lat"]
                longitude = coord_dict["altitude_" + str(i)]["traj_" + str(traj)]["lon"]

                ystart = latitude.iloc[0]
                xstart = longitude.iloc[0]

                linestyle = subplot_properties_dict[sub_index]
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
                        rasterized=True,
                    )

                    # add time interval points to main trajectory
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
                        rasterized=True,
                    )

                else:
                    # plot sidetrajectories
                    ax.plot(
                        longitude,  # define x-axis
                        latitude,  # define y-axis
                        linestyle,  # define linestyle
                        alpha=alpha,  # define line opacity
                        transform=ccrs.PlateCarree(),
                        rasterized=True,
                    )

        else:  # no side traj
            latitude = coord_dict["altitude_" + str(i)]["traj_0"]["lat"]
            longitude = coord_dict["altitude_" + str(i)]["traj_0"]["lon"]

            ystart = latitude.iloc[0]
            xstart = longitude.iloc[0]

            linestyle = subplot_properties_dict[sub_index]
            alpha = coord_dict["altitude_" + str(i)]["traj_0"]["alpha"]

            # plot main trajectory
            ax.plot(
                longitude,  # define x-axis
                latitude,  # define y-axis
                linestyle,  # define linestyle
                alpha=alpha,  # define line opacity
                label=textstr,
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )

            # add time interval points to main trajectory
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
                rasterized=True,
            )

        i += 1
    return


def get_dynamic_domain(coord_dict, altitude_levels, side_traj):
    """Check wheter dateline is crossed or not and return dynamic domain boundaries.

    Args:
        coord_dict          dict       containing the lan/lot data & other plot properties
        altitude_levels     int        # altitude levels
        side_traj           bool       True if there are side trajectories, else false.

    Returns:
        central_longitude   float      0° or 180°. If dateline is crossed, the central longitude is shifted (as well as all lon values)
        domain_boundaries   list       [lon0,lon1,lat0,lat1]
        lon_df              df         single column dataframe containing the (shifted/unchanged) longitude values
        cross_dateline      bool       bool, to remember if the dateline was crossed for a given trajectory

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

    # check dateline crossing
    far_east = 150 <= np.max(lon_maxs) <= 180
    far_west = -180 <= np.min(lon_mins) <= -150

    if far_east and far_west:  # if True > trajectory *must* cross dateline somehow
        central_longitude = 180  # shift the central longitude

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


def generate_map_plot(
    cross_dateline,
    coord_dict,
    side_traj,
    altitude_levels,
    domain,
    ax=None,
):
    """Generate Map Plot.

    Args:
        cross_dateline:     bool       Bool, to specify whether the dateline was crossed or not
        coord_dict:         dict       Dictionary containing the lan/lot data & other plot properties
        side_traj:          int        0/1 --> necessary for choosing the correct loop
        altitude_levels:    int        # altitude levels
        domain:             str        Domain for map
        ax:                 Axes       Axes to generate the map on. Defaults to None.

    """
    if domain == "dynamic":
        (_, custom_domain_boundaries, _, _,) = get_dynamic_domain(
            coord_dict, altitude_levels=altitude_levels, side_traj=side_traj
        )

    else:
        custom_domain_boundaries = [0, 0, 0, 0]
        cross_dateline = False

    ax = ax or plt.gca()

    ax.set_aspect(
        "auto"
    )  # scales the map, so that the aspeact ratio of fig & axes match

    domain_boundaries = crop_map(
        ax=ax, domain=domain, custom_domain_boundaries=custom_domain_boundaries
    )  # sets extent of map

    # if the start point of the trajectories is not within the domain boundaries (i.e. Teheran is certainly not in Switzerland or even Europe), this plot can be skipped
    lat = pd.DataFrame(coord_dict["altitude_1"]["traj_0"]["lat"], columns=["lat"])
    lon = pd.DataFrame(coord_dict["altitude_1"]["traj_0"]["lon"], columns=["lon"])
    if not is_visible(
        lat=lat.iloc[0],
        lon=lon.iloc[0],
        domain_boundaries=domain_boundaries,
        cross_dateline=False,
    ):
        return ax.text(
            0.5,
            0.5,
            "The start point is not within the domain.",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="center",
            horizontalalignment="center",
            rasterized=True,
        )

    add_features(ax=ax)

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

    ax.legend(fontsize=8)
    return ax
