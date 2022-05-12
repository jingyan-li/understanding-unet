# TODO: Road network with category

'''
Download road network as mask from OSM
Source: https://github.com/karbogas/traffic4cast/blob/cec5523a794df26c4a71723c866ad5d1443c2d94/utils/OSM.py#L28
'''

import geopandas as gpd
from shapely.geometry import Polygon
import osmnx as ox
import numpy as np
import pickle
import os


def getMaskFromOSM(city='Berlin', storage=''):
    """
        Creates a mask for the Preprocessing from OSM data
        input:
            city: city for which the mask should be created
            storage: mask storage folder
        output:
            mask: created OSM mask
    """

    # Coordinates of the bounding boxes
    if city == 'Berlin':
        yMin = 52.359
        yMax = 52.854
        xMin = 13.189
        xMax = 13.625
    elif city == 'Moscow':
        yMin = 55.506
        yMax = 55.942
        xMin = 37.357
        xMax = 37.852
    elif city == 'Istanbul':
        yMin = 40.810
        yMax = 41.305
        xMin = 28.794
        xMax = 29.230

    # Load street data
    G = ox.graph_from_bbox(yMax, yMin, xMax, xMin, network_type='drive', truncate_by_edge=True)

    # Convert street data to GeoDataFrame
    nodes, edges = ox.graph_to_gdfs(G)

    # Size of the grid cells
    length = 0.001

    # Bounding boxes of the grid cells
    cols = np.arange(xMin, xMax, length)
    rows = np.arange(yMin, yMax, length)

    polygons = []
    xList = []
    yList = []

    # Build grid cells representing the pixels of the traffic4cast images
    for x in cols:
        for y in rows:
            polygons.append(Polygon([(x, y), (x + length, y), (x + length, y + length), (x, y + length)]))
            xList.append(int(np.round(1000 * (x - xMin))))
            yList.append(int(np.round(1000 * (y - yMin))))

    grid = gpd.GeoDataFrame({'geometry': polygons, 'x': xList, 'y': yList})

    edges.crs = grid.crs

    # intersect road data with grid
    joined = gpd.sjoin(edges, grid, op='intersects')

    mask = np.zeros((495, 436))

    # Build a mask from intersections (+ rotate data to fit desired output)
    if city == 'Moscow':
        for idx, row in joined.iterrows():
            mask[row.x, row.y] = 1
        mask = np.flip(mask)
    else:
        for idx, row in joined.iterrows():
            mask[row.y, row.x] = 1
        mask = np.flip(mask, 0)

    mask = (mask > 0)

    # Save mask
    path = os.path.join(storage, city + '_coarse.mask')
    pickle.dump(mask, open(path, 'wb'))

    return mask


if __name__ == '__main__':

    getMaskFromOSM(city = 'Berlin', storage = r'./')