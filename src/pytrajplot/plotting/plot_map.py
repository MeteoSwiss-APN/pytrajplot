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

# Local
from .plot_utils import alps_cities_list
from .plot_utils import centraleurope_cities_list
from .plot_utils import ch_cities_list
from .plot_utils import europe_cities_list
from .plot_utils import subplot_properties_dict


def add_features(ax):
    """Add grid/coastlines/borders/oceans/lakes and rivers to current map (ax).

    Args:
        ax:                 Axes       Current map to add features to

    """
    # point cartopy to the folder containing the shapefiles for the features on the map
    earth_data_path = Path("src/pytrajplot/resources/")
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

    return


def crop_map(ax, domain, custom_domain_boundaries, origin_coordinates):
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

    return domain_boundaries


def get_dynamic_zoom_boundary(custom_domain_boundaries, origin_coordinates):
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

    in_domain = (
        domain_boundaries[0] <= float(lon) <= domain_boundaries[1]
        and domain_boundaries[2] <= float(lat) <= domain_boundaries[3]
    )

    if in_domain:
        return True
    else:
        return False


def is_of_interest(name, capital_type, population, lon) -> bool:
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
        cities_data_path = Path("src/pytrajplot/resources/cities/")
        assert (
            cities_data_path.exists()
        ), f"The cities data could not be found at {cities_data_path}"
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


def add_time_interval_points(plot_dict, ax, i, linestyle):
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


def retrieve_interval_points(plot_dict, altitude_index):
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


def add_trajectories(
    plot_dict,
    side_traj,
    altitude_levels,
    ax,
):
    """Add trajectories to map.

    Args:
        plot_dict:                    dict       containing the lan/lot data & other plot properties
        side_traj:                     int        0/1 --> necessary for choosing the correct loop
        altitude_levels:               int        # altitude levels
        ax:                            Axes       current map to crop

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
            traj_index = [0, 1, 2, 3, 4]

            for traj in traj_index:
                latitude = plot_dict["altitude_" + str(i)]["traj_" + str(traj)]["lat"]
                longitude = plot_dict["altitude_" + str(i)]["traj_" + str(traj)]["lon"]

                ystart = latitude.iloc[0]
                xstart = longitude.iloc[0]
                linestyle = subplot_properties_dict[sub_index]
                alpha = plot_dict["altitude_" + str(i)]["traj_" + str(traj)]["alpha"]

                if (
                    plot_dict["altitude_" + str(i)]["traj_" + str(traj)]["alpha"] == 1
                ):  # only add legend & startpoint for the main trajectories

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
                    add_time_interval_points(plot_dict, ax, i, linestyle)

                    # add start point triangle
                    ax.plot(
                        xstart,
                        ystart,
                        marker="D",
                        markersize=10,
                        markeredgecolor="red",
                        markerfacecolor="white",
                        transform=ccrs.Geodetic(),
                        rasterized=True,
                    )

                else:
                    # plot sidetrajectories
                    ax.plot(
                        longitude,  # define x-axis
                        latitude,  # define y-axis
                        linestyle,  # define linestyle
                        alpha=alpha,  # define line opacity
                        transform=ccrs.Geodetic(),
                        rasterized=True,
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
    cross_dateline,
    plot_dict,
    side_traj,
    altitude_levels,
    domain,
    trajectory_expansion,  # this is the dynamic domain
    ax=None,
):
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
    # print(f"Cropping map took:\t\t{end-start} seconds")

    # if the start point of the trajectories is not within the domain boundaries (i.e. Teheran is certainly not in Switzerland or even Europe), this plot can be skipped
    lat = pd.DataFrame(plot_dict["altitude_1"]["traj_0"]["lat"], columns=["lat"])
    lon = pd.DataFrame(plot_dict["altitude_1"]["traj_0"]["lon"], columns=["lon"])
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

    add_trajectories(
        plot_dict=plot_dict,
        side_traj=side_traj,
        altitude_levels=altitude_levels,
        ax=ax,
    )
    ax.legend(fontsize=8)
    return ax
