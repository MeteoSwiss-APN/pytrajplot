"""Generate Altitude Figure."""
# map plotting packages
# Third-party
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import cartopy.io.shapereader as shpreader
import cartopy.mpl.geoaxes
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from vega_datasets import data as vds


def add_features(map):
    map.coastlines()
    map.add_feature(cfeature.LAND)
    map.add_feature(cfeature.COASTLINE)
    map.add_feature(cfeature.BORDERS, linestyle="--")
    map.add_feature(cfeature.OCEAN)
    map.add_feature(cfeature.LAKES)
    map.add_feature(cfeature.RIVERS)
    # map.add_feature(cfeature.STATES)
    return


def crop_map(map):
    # [lat0,lat1,lon0,lon1]
    #  Define the croppinglimits like this: [x0,x1,y0,y1]
    # -130° --> 130° West --> x_0
    #   20° -->  20° Nord --> y_0
    # - 60° -->  60° West --> x_1
    #   55° -->  55° Nord --> y_1

    map.set_extent(
        [5.721748, 11.607745942217694, 45.181756, 48.18092927500776], ccrs.PlateCarree()
    )
    return


def plot_map(outpath):
    print("--- generating map")

    fig = plt.figure(figsize=(10, 5))
    m1 = plt.axes(projection=ccrs.PlateCarree())

    add_features(map=m1)
    crop_map(map=m1)

    # airports = vds.airports()
    # airports = airports.iloc[:10]
    # airports.head()

    # for i in airports.itertuples():
    #     m1.scatter(i.longitude, i.latitude, color='blue', transform=ccrs.PlateCarree())
    #     plt.text(i.longitude, i.latitude, i.name)

    plt.savefig(outpath + "/maptest.png")

    return
