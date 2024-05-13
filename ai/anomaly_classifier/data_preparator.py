"""
This module contains the implementation of
data preparation for further training.
"""

import argparse
import os
import json

import geopandas as gpd
from osgeo import gdal
from pyproj import CRS, Transformer
import numpy as np
import tifffile as tiff
from PIL import Image
import cv2


def save_image(image, image_path):
    """Save the provided image to the specified image path.

    Args:
        image (PIL.Image.Image): the image to be saved.
        image_path (str): the path where the image should be saved.

    Raises:
        Exception: if an error occurs during the saving process.
    """

    image_file_name = os.path.basename(image_path)
    try:
        image.save(image_path)
        print(f"{image_file_name} is saved.")
    except Exception as e:
        print(f"Cannot save {image_file_name}\nError occurred:\n{e}.")
        pass


def get_tiff_crs_epsg(ds):
    """Gets the EPSG code best matching the CRS (Coordinate Reference System) of the GeoTIF.

    Args:
        ds (gdal.GeoTIFF): GeoTIFF file.

    Returns:
        int: the EPSG code.
    """

    crs_wkt = ds.GetProjection()
    crs = CRS.from_wkt(crs_wkt)
    epsg_code = crs.to_epsg()

    return epsg_code


def convert_polygon_gps_to_tiff_epsg(crs_epsg, polygon):
    """Converts polygon GPS coordinates to TIFF EPSG coordinates.

    Args:
        crs_epsg (int): the EPSG code.
        polygon (list([float, float]): list of GPS polygon coordinates.

    Returns:
        np.array: transformed polygon coordinates regarding to the EPSG code.
    """

    epsg = CRS(crs_epsg)
    gps = CRS('EPSG:4326')

    transformer = Transformer.from_crs(gps, epsg, always_xy=True)
    transformed_polygon = []

    for point in polygon:
        transformed_point = transformer.transform(point[0], point[1])
        transformed_polygon.append(transformed_point)

    transformed_polygon = np.array(transformed_polygon)

    return transformed_polygon


def epsg_to_pixel(geo_matrix, epsg_coord):
    """Transform GeoJSON coordinates to pixel coordinates.

    Args:
        geo_matrix (np.array):
        epsg_coord (np.array): list of EPSG polygon coordinates.

    Returns:
        np.array: polygon pixel coordinates.
    """

    pixel_x = ((epsg_coord[:, 0] - geo_matrix[0]) / geo_matrix[1]).astype(int)
    pixel_y = ((epsg_coord[:, 1] - geo_matrix[3]) / geo_matrix[5]).astype(int)

    pixel_coords = np.stack((pixel_x, pixel_y), axis=1)

    return pixel_coords


def get_extent_polygon_coords(polygon_coords):
    """Gets Polygon extent pixel coordinates.

    Args:
        polygon_coords (list([float, float]): list of pixel polygon coordinates.

    Returns:
        minx (int): the minimum x-coordinate (the leftmost)
        miny (int): the minimum y-coordinate (the bottommost)
        maxx (int): the maximum x-coordinate (the rightmost)
        maxy (int): the maximum y-coordinate (the topmost)
    """

    minx = np.min(polygon_coords[:, 0])
    miny = np.min(polygon_coords[:, 1])
    maxx = np.max(polygon_coords[:, 0])
    maxy = np.max(polygon_coords[:, 1])

    return minx, miny, maxx, maxy


def get_tiles_number(height, width, tile_width, tile_height):
    """Calculates total numbers of tiles considering TIFF and desire tile sizes.

    Args:
        height (int): TIFF height.
        width (int): TIFF width.
        tile_width (int): desired tile width.
        tile_height (int): desired tile height.

    Returns:
        num_tiles_x (int): tiles number in one row regarding x-axis.
        num_tiles_y (int): tiles number in one regarding y-axis.
        total_tiles (int): total number of tiles.
    """

    num_tiles_x = width // tile_width
    num_tiles_y = height // tile_height

    remaining_x = width % tile_width
    remaining_y = height % tile_height

    if remaining_x > 0:
        num_tiles_x += 1

    if remaining_y > 0:
        num_tiles_y += 1

    total_tiles = num_tiles_x * num_tiles_y

    return num_tiles_x, num_tiles_y, total_tiles


def darken_outside_polygon(image, polygon_coords):
    """Darken area outside the polygon.

    Args:
        image (np.array): the input image.
        polygon_coords (np.array): polygon pixel coordinates.

    Returns:
        masked (np.array): masked image with darken area outside the polygon.
    """

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_coords], 255)

    masked = cv2.bitwise_and(image, image, mask=mask)
    cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

    return masked


def crop_polygon_on_tiles(polygon_image, polygon_pixel_coords, tile_width, tile_height):
    """Crop polygon on tiles of desired size.

    Args:
        polygon_image (PIL.JpegImage): the polygon image.
        polygon_pixel_coords (np.array): the polygon pixel coordinates.
        tile_width (int): desired tile width.
        tile_height (int): desired tile_height.

    Returns:
        polygon_tiles: list of polygon tiles.
        polygon_tiles_coords: the left upper coordinate of each tile.
    """

    polygon_width, polygon_height = polygon_image.size[:2]
    minx, miny, maxx, maxy = get_extent_polygon_coords(polygon_pixel_coords)

    num_tiles_x, num_tiles_y, total_tiles = get_tiles_number(polygon_height, polygon_width, tile_width, tile_height)

    polygon_tiles = []
    polygon_tiles_coords = []

    for i in range(num_tiles_x):
        for j in range(num_tiles_y):

            left = min(i * tile_width, maxx - tile_width)
            upper = min(j * tile_height, maxy - tile_height)
            right = min((i + 1) * tile_width, maxx)
            lower = min((j + 1) * tile_height, maxy)

            tile = polygon_image.crop((left, upper, right, lower))

            tile_data = tile.getdata()
            is_fully_black = all(pixel == (0, 0, 0) for pixel in tile_data)

            if is_fully_black:
                print("The whole image is black.")
            else:
                polygon_tiles.append(tile)
                polygon_tiles_coords.append([int(left), int(upper)])

    return polygon_tiles, polygon_tiles_coords


def crop_polygons(geojson_path, tiff_path, tile_width, tile_height, output_dir):
    """Prepare the data by implementing the polygons outlining and cropping on tiles of desired size.

    Args:
        geojson_path (str): path to GeoJSON file.
        tiff_path (str): path to TIFF file.
        tile_width (int): desired tile width.
        tile_height (int): desired tile_height.
        output_dir (str): directory to save cropped tiles.
    """

    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(f"Error occurred: \n{e}.")
            return

    try:
        gdf = gpd.read_file(geojson_path)
        print(f"\nRead the GeoJSON file at {geojson_path}.")
    except Exception as e:
        print(f"Cannot read read GeoJSON file {geojson_path}.\nError occurred:\n{e}.")
        return

    try:
        data_source = gdal.Open(tiff_path)
        large_tiff = tiff.imread(tiff_path)
        print(f"Read the TIFF file at {tiff_path}.")
    except Exception as e:
        print(f"Cannot open TIF file {tiff_path}.\nError occurred:\n{e}.")
        return

    output_dir_polygons = output_dir + "/polygons"
    output_dir_tiles = output_dir + "/tiles"

    if not os.path.exists(output_dir_polygons):
        os.mkdir(output_dir_polygons)

    if not os.path.exists(output_dir_tiles):
        os.mkdir(output_dir_tiles)

    tiff_file = os.path.basename(tiff_path)
    tiff_file_name = os.path.splitext(tiff_file)[0]
    geojson_file = os.path.basename(geojson_path)

    data = {
        "tif": tiff_file,
        "geojson": geojson_file,
    }
    polygons_data = []

    geo_transform = data_source.GetGeoTransform()
    crs_epsg = get_tiff_crs_epsg(data_source)

    for p, polygon in enumerate(gdf.geometry):

        polygon_gps = [list(x.exterior.coords) for x in polygon.geoms][0]
        polygon_epsg = convert_polygon_gps_to_tiff_epsg(crs_epsg, polygon_gps)
        polygon_pixel_coords = epsg_to_pixel(geo_transform, polygon_epsg)

        minx, miny, maxx, maxy = get_extent_polygon_coords(polygon_pixel_coords)

        polygon_tile = large_tiff[miny:maxy, minx:maxx, :3]
        polygon_image = Image.fromarray(polygon_tile).convert('RGB')

        saving_path = f"{output_dir_polygons}/{tiff_file_name}_polygon_{p}.jpg"
        save_image(polygon_image, saving_path)

        scaled_polygon_pixel_coords = polygon_pixel_coords - np.array([minx, miny])

        polygon_tile = darken_outside_polygon(polygon_tile, scaled_polygon_pixel_coords)
        polygon_image = Image.fromarray(polygon_tile).convert("RGB")

        saving_path = f"{output_dir_polygons}/{tiff_file_name}_polygon_darken_{p}.jpg"
        save_image(polygon_image, saving_path)

        polygon_tiles, polygon_tiles_coords = crop_polygon_on_tiles(polygon_image, scaled_polygon_pixel_coords,
                                                                    tile_width, tile_height)

        output_dir_polygon_tiles = output_dir_tiles + f"/{tiff_file_name}_polygon_{p}"
        if not os.path.exists(output_dir_polygon_tiles):
            os.mkdir(output_dir_polygon_tiles)

        tiles_data = []
        for i_tile, tile in enumerate(polygon_tiles):

            saving_path = f"{output_dir_polygon_tiles}/{tiff_file_name}_polygon_{p}_tile_{i_tile}.jpg"
            save_image(tile, saving_path)

            tile_data = {
                "tile_name": f"{tiff_file_name}_polygon_{p}_tile_{i_tile}.jpg",
                "tile_pixel_coords": polygon_tiles_coords[i_tile]
            }
            tiles_data.append(tile_data)

        polygon_width, polygon_height = polygon_image.size[:2]
        polygon_data = {
            "polygon_name": f"{tiff_file_name}_polygon_{p}.jpg",
            "polygon_size": [int(polygon_width), int(polygon_height)],
            "polygon_tiff_pixel_coordinates": [int(miny), int(minx)],
            "polygon_pixel_coordinates": scaled_polygon_pixel_coords.tolist(),
            "tiles": tiles_data
        }
        polygons_data.append(polygon_data)

    data["polygons"] = polygons_data

    json_file_path = f"{output_dir}/log_{tiff_file_name}.json"
    with open(json_file_path, 'w', encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)
        print(f"\nlog_{tiff_file_name}.json is saved.")


def crop_polygons_handler(args):
    crop_polygons(args.geojson_path,
                  args.tiff_path,
                  args.tile_width,
                  args.tile_height,
                  args.savedir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--geojson_path",
                        help="Path to GeoJSON file",
                        required=True)
    parser.add_argument("--tiff_path",
                        help="Path to TIFF file",
                        required=True)
    parser.add_argument("--tile_width",
                        help="Tile width.",
                        type=int,
                        default=256)
    parser.add_argument("--tile_height",
                        help="Tile height.",
                        type=int,
                        default=256)
    parser.add_argument("--savedir",
                        help="Where to save cropped tiles",
                        required=True)

    parser.set_defaults(func=crop_polygons_handler)

    args = parser.parse_args()
    args.func(args)
