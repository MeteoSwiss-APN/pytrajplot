"""Generate Map Plot Figure."""

# Standard library
import time
from pathlib import Path

# Third-party
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
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
        "centraleurope": {"domain": [2, 18, 42.5, 51.5]},  # optimised boundaries
        # "centraleurope": {"domain": [2, 18, 42, 52]},     # original boundaries
        "ch": {"domain": [5.8, 10.6, 45.4, 48.2]},  # optimised boundaries
        "alps": {"domain": [0.7, 16.5, 42.3, 50]},  # optimised boundaries
        # "europe": {"domain": [-12.5, 50.5, 35, 65]},      # optimised boundaries
        "europe": {"domain": [-10, 47, 35, 65]},  # original boundaries
        # the domain ch_hd has been removed
        # "ch_hd": {"domain": [2.8, 13.2, 44.1, 49.4]},  # optimised boundaries
        # "ch_hd": {"domain": [3.5, 12.6, 44.1, 49.4]},     # original boundaries
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
    add_w_town = True
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
            cities_list = {
                "Milan": {"lon": 9.19, "lat": 45.4669},
                "Bern": {"lon": 7.4474, "lat": 46.948},
                "Vaduz": {"lon": 9.5215, "lat": 47.1415},
                "Zurich": {"lon": 8.54, "lat": 47.3786},
                "Mulhouse": {"lon": 7.34, "lat": 47.75},
                "Freiburg im Breisgau": {"lon": 7.8497, "lat": 47.9947},
                "Geneva": {"lon": 6.15, "lat": 46.2},
                "Basel": {"lon": 7.5906, "lat": 47.5606},
                "Lausanne": {"lon": 6.6333, "lat": 46.5333},
                "Lucerne": {"lon": 8.3059, "lat": 47.0523},
                "Sankt Gallen": {"lon": 9.3772, "lat": 47.4233},
                "Fribourg": {"lon": 7.15, "lat": 46.8},
                "Schaffhausen": {"lon": 8.6339, "lat": 47.6965},
                "Chur": {"lon": 9.5297, "lat": 46.8521},
                "Aosta": {"lon": 7.3206, "lat": 45.7372},
                "Zug": {"lon": 8.5169, "lat": 47.1681},
                "Aarau": {"lon": 8.0446, "lat": 47.3923},
                "Solothurn": {"lon": 7.5375, "lat": 47.2081},
                "Altdorf": {"lon": 8.6394, "lat": 46.8806},
                "Glarus": {"lon": 9.0667, "lat": 47.0333},
                "Appenzell": {"lon": 9.4086, "lat": 47.3306},
            }

        if domain == "alps":
            cities_list = {
                "Paris": {"lon": 2.3522, "lat": 48.8566},
                "Vienna": {"lon": 16.3731, "lat": 48.2083},
                "Munich": {"lon": 11.5755, "lat": 48.1372},
                "Milan": {"lon": 9.19, "lat": 45.4669},
                "Zagreb": {"lon": 15.95, "lat": 45.8},
                "Ljubljana": {"lon": 14.5167, "lat": 46.05},
                "Bern": {"lon": 7.4474, "lat": 46.948},
                "Luxembourg": {"lon": 6.1328, "lat": 49.6106},
                "Vaduz": {"lon": 9.5215, "lat": 47.1415},
                "Andorra la Vella": {"lon": 1.5, "lat": 42.5},
                "Toulouse": {"lon": 1.444, "lat": 43.6045},
                "Turin": {"lon": 7.7, "lat": 45.0667},
                "Marseille": {"lon": 5.37, "lat": 43.2964},
                "Grenoble": {"lon": 5.7224, "lat": 45.1715},
                "Montpellier": {"lon": 3.8772, "lat": 43.6119},
                "Genoa": {"lon": 8.934, "lat": 44.4072},
                "Lyon": {"lon": 4.84, "lat": 45.76},
                "Nuremberg": {"lon": 11.0775, "lat": 49.4539},
                "Rouen": {"lon": 1.0886, "lat": 49.4428},
                "Strasbourg": {"lon": 7.7458, "lat": 48.5833},
                "Nancy": {"lon": 6.1846, "lat": 48.6936},
                "Zurich": {"lon": 8.54, "lat": 47.3786},
                "Bologna": {"lon": 11.3428, "lat": 44.4939},
                "Florence": {"lon": 11.2542, "lat": 43.7714},
                "Orleans": {"lon": 1.909, "lat": 47.9025},
                "Limoges": {"lon": 1.2625, "lat": 45.8353},
                "Graz": {"lon": 15.4409, "lat": 47.0749},
                "Verona": {"lon": 10.9928, "lat": 45.4386},
                "Geneva": {"lon": 6.15, "lat": 46.2},
                "Linz": {"lon": 14.2833, "lat": 48.3},
                "Basel": {"lon": 7.5906, "lat": 47.5606},
                "Saarbrucken": {"lon": 7.0, "lat": 49.2333},
                "Split": {"lon": 16.45, "lat": 43.51},
                "Perugia": {"lon": 12.3888, "lat": 43.1121},
                "Dijon": {"lon": 5.0167, "lat": 47.3167},
                "Lausanne": {"lon": 6.6333, "lat": 46.5333},
                "Innsbruck": {"lon": 11.3933, "lat": 47.2683},
                "Rijeka": {"lon": 14.4411, "lat": 45.3272},
                "Trento": {"lon": 11.1167, "lat": 46.0667},
                "Klagenfurt": {"lon": 14.3, "lat": 46.6167},
                "Ceske Budejovice": {"lon": 14.4747, "lat": 48.9747},
                "Zadar": {"lon": 15.2167, "lat": 44.1167},
                "Jihlava": {"lon": 15.5906, "lat": 49.4003},
            }

        if domain == "centraleurope":
            cities_list = {
                "Paris": {"lon": 2.3522, "lat": 48.8566},
                "Vienna": {"lon": 16.3731, "lat": 48.2083},
                "Brussels": {"lon": 4.3333, "lat": 50.8333},
                "Munich": {"lon": 11.5755, "lat": 48.1372},
                "Milan": {"lon": 9.19, "lat": 45.4669},
                "Prague": {"lon": 14.4167, "lat": 50.0833},
                "Zagreb": {"lon": 15.95, "lat": 45.8},
                "Ljubljana": {"lon": 14.5167, "lat": 46.05},
                "Bern": {"lon": 7.4474, "lat": 46.948},
                "Luxembourg": {"lon": 6.1328, "lat": 49.6106},
                "Vaduz": {"lon": 9.5215, "lat": 47.1415},
                "Antwerp": {"lon": 4.4003, "lat": 51.2206},
                "Turin": {"lon": 7.7, "lat": 45.0667},
                "Marseille": {"lon": 5.37, "lat": 43.2964},
                "Frankfurt": {"lon": 8.6797, "lat": 50.1136},
                "Grenoble": {"lon": 5.7224, "lat": 45.1715},
                "Dusseldorf": {"lon": 6.7724, "lat": 51.2311},
                "Stuttgart": {"lon": 9.1775, "lat": 48.7761},
                "Montpellier": {"lon": 3.8772, "lat": 43.6119},
                "Leipzig": {"lon": 12.3833, "lat": 51.3333},
                "Genoa": {"lon": 8.934, "lat": 44.4072},
                "Dresden": {"lon": 13.74, "lat": 51.05},
                "Lyon": {"lon": 4.84, "lat": 45.76},
                "Nuremberg": {"lon": 11.0775, "lat": 49.4539},
            }

        if domain == "europe":
            cities_list = {
                "Moscow": {"lon": 37.6178, "lat": 55.7558},
                "Istanbul": {"lon": 28.9603, "lat": 41.01},
                "Paris": {"lon": 2.3522, "lat": 48.8566},
                "London": {"lon": -0.1275, "lat": 51.5072},
                "Madrid": {"lon": -3.7167, "lat": 40.4167},
                "Ankara": {"lon": 32.85, "lat": 39.93},
                "Saint Petersburg": {"lon": 30.3167, "lat": 59.95},
                "Barcelona": {"lon": 2.1769, "lat": 41.3825},
                "Izmir": {"lon": 27.1384, "lat": 38.4127},
                "Berlin": {"lon": 13.3833, "lat": 52.5167},
                "Algiers": {"lon": 3.0586, "lat": 36.7764},
                "Kyiv": {"lon": 30.5236, "lat": 50.45},
                "Rome": {"lon": 12.4828, "lat": 41.8931},
                "Antalya": {"lon": 30.6956, "lat": 36.9081},
                "Minsk": {"lon": 27.5618, "lat": 53.9022},
                "Vienna": {"lon": 16.3731, "lat": 48.2083},
                "Bucharest": {"lon": 26.0833, "lat": 44.4},
                "Warsaw": {"lon": 21.0333, "lat": 52.2167},
                "Brussels": {"lon": 4.3333, "lat": 50.8333},
                "Budapest": {"lon": 19.0408, "lat": 47.4983},
                "Belgrade": {"lon": 20.4667, "lat": 44.8167},
                "Sofia": {"lon": 23.3217, "lat": 42.6979},
                "Prague": {"lon": 14.4167, "lat": 50.0833},
                "Dublin": {"lon": -6.2603, "lat": 53.3497},
                "Tunis": {"lon": 10.18, "lat": 36.8008},
                "Stockholm": {"lon": 18.0686, "lat": 59.3294},
                "Amsterdam": {"lon": 4.8833, "lat": 52.3667},
                "Zagreb": {"lon": 15.95, "lat": 45.8},
                "Oslo": {"lon": 10.7528, "lat": 59.9111},
                "Chisinau": {"lon": 28.8353, "lat": 47.0228},
                "Riga": {"lon": 24.1069, "lat": 56.9475},
                "Copenhagen": {"lon": 12.5689, "lat": 55.6761},
                "Vilnius": {"lon": 25.2833, "lat": 54.6833},
                "Lisbon": {"lon": -9.1604, "lat": 38.7452},
                "Tirana": {"lon": 19.82, "lat": 41.33},
                "Nicosia": {"lon": 33.365, "lat": 35.1725},
                "Sarajevo": {"lon": 18.4167, "lat": 43.8667},
                "Bern": {"lon": 7.4474, "lat": 46.948},
                "Luxembourg": {"lon": 6.1328, "lat": 49.6106},
                "Valletta": {"lon": 14.5125, "lat": 35.8978},
            }

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
    coord_dict,
    side_traj,
    altitude_levels,
    ax,
    subplot_properties_dict,
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
                        transform=ccrs.Geodetic(),
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
                transform=ccrs.Geodetic(),
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
                transform=ccrs.Geodetic(),
                rasterized=True,
            )

        i += 1
    return


def generate_map_plot(
    cross_dateline,
    coord_dict,
    side_traj,
    altitude_levels,
    domain,
    trajectory_expansion,  # this is the dynamic domain
    ax=None,
):
    """Generate Map Plot.

    Args:
        cross_dateline:             bool       Bool, to specify whether the dateline was crossed or not
        coord_dict:                 dict       Dictionary containing the lan/lot data & other plot properties
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
        "lon": coord_dict["altitude_1"]["traj_0"]["lon"].iloc[0],
        "lat": coord_dict["altitude_1"]["traj_0"]["lat"].iloc[0],
    }

    start = time.perf_counter()
    domain_boundaries = crop_map(
        ax=ax,
        domain=domain,
        custom_domain_boundaries=trajectory_expansion,
        origin_coordinates=origin_coordinates,
    )  # sets extent of map
    end = time.perf_counter()
    print(f"Cropping map took:\t\t{end-start} seconds")

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

    start = time.perf_counter()
    add_features(ax=ax)
    end = time.perf_counter()
    print(f"Adding features took:\t\t{end-start} seconds")

    start = time.perf_counter()
    add_cities(
        ax=ax,
        domain_boundaries=domain_boundaries,
        domain=domain,
        cross_dateline=cross_dateline,
    )
    end = time.perf_counter()
    print(f"Adding cities took:\t\t{end-start} seconds")

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
    start = time.perf_counter()
    add_trajectories(
        coord_dict=coord_dict,
        side_traj=side_traj,
        altitude_levels=altitude_levels,
        ax=ax,
        subplot_properties_dict=subplot_properties_dict,
    )
    end = time.perf_counter()
    print(f"Adding trajectories took:\t{end-start} seconds")
    ax.legend(fontsize=8)
    return ax
