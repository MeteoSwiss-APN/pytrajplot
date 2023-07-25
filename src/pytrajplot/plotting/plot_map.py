"""Generate Map Plot Figure."""

# Standard library
from pathlib import Path
from itertools import groupby
from typing import List, Dict, Union, Tuple, Any

# Third-party
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes as geoaxes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local
from ..__init__ import cities_data_path
from .plot_utils import alps_cities_list
from .plot_utils import centraleurope_cities_list
from .plot_utils import ch_cities_list
from .plot_utils import europe_cities_list
from .plot_utils import subplot_properties_dict

# from ..__init__ import earth_data_path


def add_features(ax: geoaxes.GeoAxesSubplot) -> None:
    """Add grid/coastlines/borders/oceans/lakes and rivers to current map (ax).

    Args:
        ax:                 Axes       Current map to add features to

    """
    # point cartopy to the folder containing the shapefiles for the features on the map
    # earth_data_path = Path("src/pytrajplot/resources/")
    # assert (
    #     earth_data_path.exists()
    # ), f"The natural earth data could not be found at {earth_data_path}"
    # # earth_data_path = str(earth_data_path)
    # cartopy.config["pre_existing_data_dir"] = earth_data_path
    # cartopy.config["data_dir"] = earth_data_path
    # add grid & labels to map
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="k",
        alpha=0.3,
        linestyle="-.",
        rasterized=True,
    )  # define grid line properties
    gl.top_labels = False
    gl.right_labels = False

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


def crop_map(
    ax: geoaxes.GeoAxesSubplot,
    domain: str,
    custom_domain_boundaries: List[float],
    origin_coordinates: Dict[str, Union[float, str]],
) -> List[float]:
    """Crop map to given domain (i.e. centraleurope).

    Args:
        ax                  Axes       current map to crop
        domain              str        key for the domain_dict to retrieve correct domain boundaries
        custom_domain_boundaries
                            list       list, containing the domain for these specifc trajectories (created dynamically)
        origin_coordinates  dict       dict, containing the lon/lat values of the origin

    Returns:
        domain_boundaries   list       [lat0,lat1,lon0,lon1]

    """
    left_boundary, right_boundary, lower_boundary, upper_boundary = 0, 0, 0, 0
    if domain == "dynamic_zoom":
        (
            left_boundary,
            right_boundary,
            lower_boundary,
            upper_boundary,
        ) = get_dynamic_zoom_boundary(custom_domain_boundaries, origin_coordinates)

    padding = 1  # padding on each side, for the dynamically created plots
    domain_dict = {
        "centraleurope": {
            "domain": [1, 20, 42.5, 51.5]
        },  # added two degrees to the east
        "ch": {"domain": [5.3, 11.2, 45.4, 48.2]},  # optimised boundaries
        "alps": {"domain": [0.7, 16.5, 42.3, 50]},  # optimised boundaries
        "europe": {"domain": [-10, 47, 35, 65]},  # original boundaries
        "dynamic": {
            "domain": [
                round(custom_domain_boundaries[0]) - padding,
                round(custom_domain_boundaries[1]) + padding,
                round(custom_domain_boundaries[2]) - padding,
                round(custom_domain_boundaries[3]) + padding,
            ]
        },
        "dynamic_zoom": {
            "domain": [
                round(left_boundary),
                round(right_boundary),
                round(lower_boundary),
                round(upper_boundary),
            ]
        },
    }
    domain_boundaries = domain_dict[domain]["domain"]
    ax.set_extent(domain_boundaries, crs=ccrs.PlateCarree(central_longitude=0))
    lon_limits = ax.get_xlim()
    if np.isclose(np.abs(lon_limits[0] - lon_limits[1]), 360, atol=1):
        return []
    else:
        return domain_boundaries


def get_dynamic_zoom_boundary(
    custom_domain_boundaries: List[float], origin_coordinates: Dict[str, float]
) -> Tuple[float, float, float, float]:
    # case 1: trajectory expansion mainly towards the east from origin
    if abs(custom_domain_boundaries[0] - origin_coordinates["lon"]) <= 10:
        left_boundary = origin_coordinates["lon"] - 2
        right_boundary = origin_coordinates["lon"] + 18
    # case 2: trajectory expansion mainly towards the west from origin
    if abs(custom_domain_boundaries[1] - origin_coordinates["lon"]) <= 10:
        left_boundary = origin_coordinates["lon"] - 18
        right_boundary = origin_coordinates["lon"] + 2
    # case 3: trajectory expansion to both east&west from origin
    if (abs(custom_domain_boundaries[0] - origin_coordinates["lon"]) > 10) and (
        abs(custom_domain_boundaries[1] - origin_coordinates["lon"]) > 10
    ):
        left_boundary = origin_coordinates["lon"] - 10
        right_boundary = origin_coordinates["lon"] + 10

    # case 1: trajectory expansion mainly towards the north from origin
    if abs(custom_domain_boundaries[2] - origin_coordinates["lat"]) <= 10:
        lower_boundary = origin_coordinates["lat"] - 3
        upper_boundary = origin_coordinates["lat"] + 12
    # case 2: trajectory expansion mainly towards the south from origin
    if abs(custom_domain_boundaries[3] - origin_coordinates["lat"]) <= 10:
        lower_boundary = origin_coordinates["lat"] - 12
        upper_boundary = origin_coordinates["lat"] + 3
    # case 3: trajectory expansion to both south&north from origin
    if (abs(custom_domain_boundaries[2] - origin_coordinates["lat"]) > 10) and (
        abs(custom_domain_boundaries[3] - origin_coordinates["lat"]) > 10
    ):
        lower_boundary = origin_coordinates["lat"] - 7.5
        upper_boundary = origin_coordinates["lat"] + 7.5
    return left_boundary, right_boundary, lower_boundary, upper_boundary


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

    is_in_domain = (
        domain_boundaries[0] <= float(lon) <= domain_boundaries[1]
        and domain_boundaries[2] <= float(lat) <= domain_boundaries[3]
    )
    return is_in_domain


def is_of_interest(name: str, capital_type: str, population: int, lon: float) -> bool:
    """Check if a city fulfils certain importance criteria.

    Args:
        name                str        Name of city
        capital_type        str        primary/admin/minor (label for "relevance" of city)
        population          int        Population of city (there are conditions based on population)
        lon                 float      Longitude of city  (there are conditions based on longitude)

    Returns:
                            bool       True if city is of interest, else false

    """
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


def add_cities(
    ax: geoaxes.GeoAxesSubplot,
    domain_boundaries: List[float],
    domain: str,
    cross_dateline: bool,
) -> None:
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
    # Easter Egg - always add Weesen to plots, to make sure its safe.
    add_w_town = False
    if add_w_town:
        if domain in ["ch", "alps", "europe", "centraleurope"]:
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

    if "dynamic" in domain:
        # cities_data_path = Path("src/pytrajplot/resources/cities/")
        # assert (
        #     cities_data_path.exists()
        # ), f"The cities data could not be found at {cities_data_path}"
        cities_df = pd.read_csv(Path(cities_data_path, "worldcities.csv"))

        # remove less important cities to reduce size of dataframe (from 41001 rows to 8695)
        cities_df = cities_df.dropna()

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
                lon=lon,
            ):
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

                ax.text(
                    x=lon + 0.05,
                    y=lat + 0.05,
                    s=city,
                    fontsize=8,
                    transform=ccrs.Geodetic(),
                    rasterized=True,
                )

    else:
        text_shift = 0.05

        if domain == "ch":
            text_shift = 0.01
            cities_list = ch_cities_list

        if domain == "alps":
            cities_list = alps_cities_list

        if domain == "centraleurope":
            cities_list = centraleurope_cities_list

        if domain == "europe":
            cities_list = europe_cities_list

        for city in cities_list:
            lon = cities_list[city]["lon"]
            lat = cities_list[city]["lat"]
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
            ax.text(
                x=lon + text_shift,
                y=lat + text_shift,
                s=city,
                fontsize=8,
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )


def add_time_interval_points(
    plot_dict: Dict[str, Union[List[float], str]],
    ax: geoaxes.GeoAxesSubplot,
    i: int,
    linestyle: str,
) -> None:
    """Add time interval points to map..

    Args:
        plot_dict:         dict       containing the lan/lot data & other plot properties
        ax:                 Axes       current map to crop
        i:                  int        Altitude index. Only want to add legend for altitude one.
        linestyle:          str        Defines colour of interval poins (same as corresponding trajectory)

    """
    lon_important, lat_important = retrieve_interval_points(plot_dict, altitude_index=i)

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


def filter_by_domain(
    longitude: pd.Series,
    latitude: pd.Series,
    domain_boundaries: List[float],
    cross_dateline: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filters points out which are not within fiven domain boundaries.

    Args:
        latitude:            list of the points' latitude
        longitude:           list of the points' longitude
        domain_boundaries:   lon/lat range of domain
        cross_dateline       if other coordinates cross the dateline
    """
    filter_mask = [
        is_visible(lat, lon, domain_boundaries, cross_dateline)
        for lon, lat in zip(longitude.to_numpy(), latitude.to_numpy())
    ]
    return longitude.to_numpy()[filter_mask], latitude.to_numpy()[filter_mask]


def add_time_interval_points_within_domain(
    plot_dict: dict,
    ax: Any,
    i: int,
    linestyle: str,
    domain_boundaries: List[float],
    cross_dateline: bool,
) -> None:
    """Add time interval points to map..

    Args:
        plot_dict:         dict       containing the lan/lot data & other plot properties
        ax:                 Axes       current map to crop
        i:                  int        Altitude index. Only want to add legend for altitude one.
        linestyle:          str        Defines colour of interval poins (same as corresponding trajectory)

    """

    lon_important, lat_important = retrieve_interval_points(plot_dict, altitude_index=i)
    lon_important, lat_important = filter_by_domain(
        lon_important, lat_important, domain_boundaries, cross_dateline
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


def retrieve_interval_points(
    plot_dict: dict, altitude_index: int
) -> Tuple[pd.Series, pd.Series]:
    """Extract the interval points from plot_dict add them to ax.

    Args:
        plot_dict          dict       containing the lan/lot data & other plot properties
        altitude_index      int        Altitude index. Only want to add legend for altitude one.

    Returns:
        lon_important       series     pandas list w/ interval point longitude values
        lat_important       series     pandas list w/ interval point latitude values

    """
    # create temporary dataframes
    lat_df_tmp = plot_dict["altitude_" + str(altitude_index)]["traj_0"]["lat"].values
    lon_df_tmp = plot_dict["altitude_" + str(altitude_index)]["traj_0"]["lon"].values
    time_df_tmp = plot_dict["altitude_" + str(altitude_index)]["traj_0"]["time"].values

    comb_df = pd.DataFrame(
        data={"time": time_df_tmp, "lon": lon_df_tmp, "lat": lat_df_tmp},
        dtype=np.float64,
    )

    shift = comb_df["time"].values[0]
    if shift > 0:
        # The time column of COSMO trajectories starts @ the lead time. I.e. the first entry for 003-033F would be 3.00.
        # The time column of HRES  trajectories starts w/ 0.00, ALWAYS. Thus when computing the modulo of the time column
        # of HRES trajectories, the 'interval' points get computet correctly. This shift in the COSMO data must be accounted
        # for by subtraction the time-shift from the time column, before applying %6
        comb_df["time"] -= shift

    # extract position every 6 hours into important_points dataframe
    important_points_tmp = comb_df[comb_df["time"] % 6 == 0]
    important_points = important_points_tmp.iloc[
        1:
    ]  # remove start point --> want triangle there
    lon_important = important_points["lon"]
    lat_important = important_points["lat"]

    return lon_important, lat_important


def get_projected_point_on_domain_boundaries(
    lat: float, lon: float, domain_boundaries: Tuple[float, float, float, float]
) -> np.ndarray:
    """Calculates the projection of a given point onto a given boundary with the shortest distance.
    Args:
        lat:                  Latitude of the given point
        lon:                  Longitude of the given point
        domain_boundaries:    Domain consisting of latitudes and longitudes
    """
    min_dist = np.inf
    closest_points = None
    position = np.array([lon, lat])
    lon_min, lon_max, lat_min, lat_max = domain_boundaries
    boundary_points = np.array(
        [[lon_min, lat_min], [lon_min, lat_max], [lon_max, lat_max], [lon_max, lat_min]]
    )
    for i in range(len(boundary_points)):
        p = np.array(boundary_points[i])
        q = np.array(boundary_points[(i + 1) % len(boundary_points)])
        if np.dot(q - p, position - p) > 0 and np.dot(p - q, position - q) > 0:
            projected_point = p + (q - p) * np.dot(
                q - p, position - p
            ) / np.linalg.norm(q - p)
            dist = np.linalg.norm(position - projected_point)
            if dist < min_dist:
                min_dist = dist
                closest_points = projected_point
        else:
            dist_p = np.linalg.norm(position - p)
            if dist_p < min_dist:
                min_dist = dist_p
                closest_points = p

            dist_q = np.linalg.norm(position - q)
            if dist_q < min_dist:
                min_dist = dist_q
                closest_points = q
    return closest_points


def get_intersection_point_on_domain_boundaries(
    lat_in: float,
    lon_in: float,
    lat_out: float,
    lon_out: float,
    domain_boundaries: List[float],
) -> np.ndarray:
    """Calculates the intersection point between two points and a given boundary.
    Args:
        lat_in:                   Latitude of the point within the domain
        lon_in:                   Longitude of the point within the domain
        lat_out:                  Latitude of the point out of the domain
        lon_out:                  Longitude of the point out of the domain
        domain_boundaries:        Domain consisting of latitudes and longitudes
    """
    min_dist = np.inf
    closest_point_out = None
    closest_point_in = None
    point_out = np.array([lon_out, lat_out])
    point_in = np.array([lon_in, lat_in])
    lon_min, lon_max, lat_min, lat_max = domain_boundaries
    boundary_points = np.array(
        [[lon_min, lat_min], [lon_min, lat_max], [lon_max, lat_max], [lon_max, lat_min]]
    )
    for i in range(len(boundary_points)):
        p = np.array(boundary_points[i])
        q = np.array(boundary_points[(i + 1) % len(boundary_points)])
        if np.dot(q - p, point_out - p) > 0 and np.dot(p - q, point_out - q) > 0:
            line_vector = q - p
            point_vector = point_out - p
            scalar = np.dot(point_vector, line_vector) / np.dot(
                line_vector, line_vector
            )
            projected_point_out = p + scalar * line_vector
            dist = np.linalg.norm(point_out - projected_point_out)
            if dist < min_dist:
                min_dist = dist
                closest_point_out = projected_point_out
                closest_point_in = projected_point_out = p + (q - p) * np.dot(
                    q - p, point_in - p
                ) / np.linalg.norm(q - p)

    distance_to_projected_out = np.linalg.norm(closest_point_out - point_out)
    distance_to_projected_in = np.linalg.norm(closest_point_in - point_in)
    stretch_factor = 1 / (distance_to_projected_in / distance_to_projected_out + 1)
    intersection_point = closest_point_out + stretch_factor * (
        closest_point_out - closest_point_in
    )
    intersection_point = intersection_point + 0.001 * (point_in - intersection_point)
    return intersection_point


def add_trajectories_within_domain(
    plot_dict: dict,
    side_traj: int,
    altitude_levels: List[int],
    ax: geoaxes.GeoAxesSubplot,
    domain_boundaries: List[float],
    cross_dateline: bool,
) -> None:
    """Add trajectories to map.

    Args:
        plot_dict:                    dict       containing the lan/lot data & other plot properties
        side_traj:                     int        0/1 --> necessary for choosing the correct loop
        altitude_levels:               int        # altitude levels
        ax:                            Axes       current map to crop
        domain_boundaries:             list      lon/lat range of domain
        cross_dateline:                bool      if the trajectories cross the dateline

    """
    i = 1
    while i <= altitude_levels:
        sub_index = int(plot_dict["altitude_" + str(i)]["subplot_index"])
        textstr = (
            str(plot_dict["altitude_" + str(i)]["alt_level"])
            + " "
            + plot_dict["altitude_" + str(i)]["y_type"]
        )
        if side_traj:
            for traj in range(5):
                latitude = plot_dict["altitude_" + str(i)]["traj_" + str(traj)]["lat"]
                longitude = plot_dict["altitude_" + str(i)]["traj_" + str(traj)]["lon"]
                if cross_dateline:
                    longitude = longitude.apply(
                        lambda lon: 360 - np.abs(lon) if lon < 0 else lon
                    )
                ystart = latitude.iloc[0]
                xstart = longitude.iloc[0]
                linestyle = subplot_properties_dict[sub_index]
                alpha = plot_dict["altitude_" + str(i)]["traj_" + str(traj)]["alpha"]
                in_domain = [
                    is_visible(lat, lon, domain_boundaries, False)
                    for lat, lon in zip(latitude, longitude)
                ]
                true_indices = np.where(in_domain)[0]
                diff = np.diff(true_indices)
                if len(diff) != 0:
                    start_indices = np.r_[
                        true_indices[0], true_indices[np.where(diff != 1)[0] + 1]
                    ]
                    end_indices = np.r_[
                        true_indices[np.where(diff != 1)[0]], true_indices[-1]
                    ]
                    intervals = [
                        (start, end) for start, end in zip(start_indices, end_indices)
                    ]
                    data_length = len(latitude)
                    is_main_trajectory = (
                        plot_dict["altitude_" + str(i)]["traj_" + str(traj)]["alpha"]
                        == 1
                    )

                    for intv in intervals:
                        plot_longitude = longitude[intv[0] : intv[1]]
                        plot_latitude = latitude[intv[0] : intv[1]]
                        if intv[0] > 0:
                            plot_longitude = longitude[intv[0] - 1 : intv[1]]
                            plot_latitude = latitude[intv[0] - 1 : intv[1]]

                        if intv[1] < data_length - 1:
                            plot_longitude = plot_longitude.append(
                                pd.Series(longitude.iloc[intv[1] + 1])
                            )
                            plot_latitude = plot_latitude.append(
                                pd.Series(latitude.iloc[intv[1] + 1])
                            )

                        ax.plot(
                            plot_longitude,  # define x-axis
                            plot_latitude,  # define y-axis
                            linestyle,  # define linestyle
                            alpha=alpha,  # define line opacity
                            label=textstr
                            if is_main_trajectory
                            else None,  # only provide labels for main trajectories
                            transform=ccrs.Geodetic(),
                            rasterized=True,
                        )
                    if is_main_trajectory:
                        # add time interval points to main trajectory
                        add_time_interval_points_within_domain(
                            plot_dict,
                            ax,
                            i,
                            linestyle,
                            domain_boundaries,
                            cross_dateline,
                        )
        else:  # no side traj
            latitude = plot_dict["altitude_" + str(i)]["traj_0"]["lat"]
            longitude = plot_dict["altitude_" + str(i)]["traj_0"]["lon"]

            ystart = latitude.iloc[0]
            xstart = longitude.iloc[0]

            linestyle = subplot_properties_dict[sub_index]
            alpha = plot_dict["altitude_" + str(i)]["traj_0"]["alpha"]

            # plot main trajectory
            ax.plot(
                longitude,  # define x-axis
                latitude,  # define y-axis
                linestyle,  # define linestyle
                alpha=alpha,  # define line opacity
                label=textstr,
                transform=ccrs.Geodetic(),
                rasterized=True,
            )

            # add time interval points to main trajectory
            add_time_interval_points(
                plot_dict=plot_dict, ax=ax, i=i, linestyle=linestyle
            )

            # add start point triangle
            ax.plot(
                xstart,
                ystart,
                marker="^",
                markersize=10,
                markeredgecolor="red",
                markerfacecolor="white",
                transform=ccrs.Geodetic(),
                rasterized=True,
            )
        i += 1
    return


def generate_map_plot(
    cross_dateline: bool,
    plot_dict: Dict,
    side_traj: int,
    altitude_levels: List[int],
    domain: str,
    trajectory_expansion: List[float],  # this is the dynamic domain
    ax: geoaxes.GeoAxesSubplot = None,
) -> geoaxes.GeoAxesSubplot:
    """Generate Map Plot.

    Args:
        cross_dateline:             bool       Bool, to specify whether the dateline was crossed or not
        plot_dict:                  dict       Dictionary containing the lan/lot data & other plot properties
        side_traj:                  int        0/1 --> necessary for choosing the correct loop
        altitude_levels:            int        # altitude levels
        domain:                     str        Domain for map
        trajectory_expansion:       array      array w/ dynamic domain boundaries
        ax:                         Axes       Axes to generate the map on. Defaults to None.

    """
    ax = ax or plt.gca()
    ax.set_aspect(
        "auto"
    )  # scales the map, so that the aspeact ratio of fig & axes match
    origin_coordinates = {
        "lon": plot_dict["altitude_1"]["traj_0"]["lon"].iloc[0],
        "lat": plot_dict["altitude_1"]["traj_0"]["lat"].iloc[0],
    }
    domain_boundaries = crop_map(
        ax=ax,
        domain=domain,
        custom_domain_boundaries=trajectory_expansion,
        origin_coordinates=origin_coordinates,
    )  # sets extent of map
    if not domain_boundaries:
        return ax.text(
            0.5,
            0.5,
            "Trajectories are far away from displayed domain.",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="center",
            horizontalalignment="center",
            rasterized=True,
        )

    # print(f"Cropping map took:\t\t{end-start} seconds")
    # if the start point of the trajectories is not within the domain boundaries (i.e. Teheran is certainly not in Switzerland or even Europe), this plot can be skipped
    lat = pd.DataFrame(plot_dict["altitude_1"]["traj_0"]["lat"], columns=["lat"])
    lon = pd.DataFrame(plot_dict["altitude_1"]["traj_0"]["lon"], columns=["lon"])

    add_features(ax=ax)
    add_cities(
        ax=ax,
        domain_boundaries=domain_boundaries,
        domain=domain,
        cross_dateline=cross_dateline,
    )
    add_trajectories_within_domain(
        plot_dict=plot_dict,
        side_traj=side_traj,
        altitude_levels=altitude_levels,
        ax=ax,
        domain_boundaries=domain_boundaries,
        cross_dateline=cross_dateline,
    )
    if is_visible(
        lat=lat.iloc[0],
        lon=lon.iloc[0],
        domain_boundaries=domain_boundaries,
        cross_dateline=False,
    ):
        ax.plot(
            lon.iloc[0],
            lat.iloc[0],
            marker="D",
            markersize=10,
            markeredgecolor="red",
            markerfacecolor="white",
            transform=ccrs.Geodetic(),
            rasterized=True,
        )
        ax.legend(fontsize=8)
    return ax
