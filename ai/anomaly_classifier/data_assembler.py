"""
This module contains functions to assemble polygons
using cropped tiles based on provided JSON data.
"""

import argparse
import os
import json

import cv2
import numpy as np
from PIL import Image


def make_transparent_outside_polygon(image, polygon_coords):
    """Make transparent area outside the polygon.

    Parameters:
    - image (PIL.Image): the input image.
    - polygon_coords (np.array): polygon pixel coordinates.

    Returns:
    - masked (np.array): masked image with transparent area outside the polygon.
    """

    image = np.array(image)
    polygon_coords = np.array(polygon_coords)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_coords], 255)

    masked = cv2.bitwise_and(image, image, mask=mask)
    cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

    transparent_masked = Image.fromarray(masked).convert('RGBA')

    return transparent_masked


def assemble_whole_polygon(polygon_data, tiles_dir_path):
    """Assemble a whole polygon image by combining tiles based on the provided polygon data.

    Parameters:
    - polygon_data (dict): A dictionary containing information about the polygon, including:
        - "polygon_name" (str): The name of the polygon file.
        - "polygon_size" (tuple): The width and height of the polygon in pixels.
        - "polygon_pixel_coordinates" (list of tuples): Pixel coordinates of the polygon vertices.
        - "tiles" (list): List of dictionaries, each containing information about a tile, including:
            - "tile_name" (str): The name of the tile file.
            - "tile_pixel_coords" (tuple): Pixel coordinates where the tile should be placed in the polygon.

    - tiles_dir_path (str): The directory path where the tiles are stored.

    Returns:
    - whole_polygon (PIL.Image.Image): The assembled whole polygon image with transparency outside the polygon.
    """

    polygon_file_name = polygon_data["polygon_name"]
    polygon_name = os.path.splitext(polygon_file_name)[0]
    polygon_width, polygon_height = polygon_data["polygon_size"]
    polygon_pixel_coords = polygon_data["polygon_pixel_coordinates"]
    tiles = polygon_data["tiles"]

    background_image = np.zeros((polygon_height, polygon_width, 3), dtype=np.uint8)
    background_image = Image.fromarray(background_image).convert('RGBA')

    for tile in tiles:
        tile_name = tile["tile_name"]
        tile_pixel_coords = tile["tile_pixel_coords"]

        tile_dir = tiles_dir_path + "/" + polygon_name
        tile_path = tile_dir + "/" + tile_name

        tile_image = Image.open(tile_path)
        background_image.paste(tile_image, tuple(tile_pixel_coords))

    whole_polygon = make_transparent_outside_polygon(background_image, polygon_pixel_coords)

    return whole_polygon


def assemble_polygons(json_path, polygons_dir_path, tiles_dir_path, output_dir):
    """Assemble polygons based on the provided JSON data.

    Parameters:
    - json_path (str): The path to the JSON file containing information about the polygons.
    - polygons_dir_path (str): The directory path where the original polygon images are stored.
    - tiles_dir_path (str): The directory path where the tiles used for assembly are stored.
    - output_dir (str): The directory path where the assembled images will be saved.
    """

    if not os.path.exists(polygons_dir_path):
        raise FileNotFoundError(f"The directory '{polygons_dir_path}' does not exist.")

    if not os.path.exists(tiles_dir_path):
        raise FileNotFoundError(f"The directory '{tiles_dir_path}' does not exist.")

    with open(json_path, 'r') as file:
        json_data = json.load(file)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    transparent_polygons_dir_path = output_dir + '/transparent_polygons'
    whole_polygons_dir_path = output_dir + '/whole_polygons'

    if not os.path.exists(transparent_polygons_dir_path):
        os.mkdir(transparent_polygons_dir_path)

    if not os.path.exists(whole_polygons_dir_path):
        os.mkdir(whole_polygons_dir_path)

    for polygon in json_data["polygons"]:

        polygon_file_name = polygon["polygon_name"]
        polygon_name = os.path.splitext(polygon_file_name)[0]

        whole_polygon = assemble_whole_polygon(polygon, tiles_dir_path)

        try:
            whole_polygon.save(f'{transparent_polygons_dir_path}/{polygon_name}.png')
            print(f'{transparent_polygons_dir_path}/{polygon_name}.png is saved')
        except Exception as e:
            print(f"Cannot save {transparent_polygons_dir_path}/{polygon_name}.png \nError occurred:\n{e}.")
            continue

        polygons_path = polygons_dir_path + "/" + polygon["polygon_name"]
        polygon_image = Image.open(polygons_path)

        polygon_image.paste(whole_polygon, (0, 0), whole_polygon)

        try:
            polygon_image.save(f'{whole_polygons_dir_path}/{polygon_name}.jpg')
            print(f'{whole_polygons_dir_path}/{polygon_name}.jpg is saved\n')
        except Exception as e:
            print(f"Cannot save {whole_polygons_dir_path}/{polygon_name}.jpg \nError occurred:\n{e}.")
            continue


def assemble_polygons_handler(args):
    assemble_polygons(args.json_path,
                      args.polygons_path,
                      args.tiles_path,
                      args.savedir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path",
                        help="Path to directory with log file",
                        required=True)
    parser.add_argument("--polygons_path",
                        help="Path to directory with polygon files",
                        required=True)
    parser.add_argument("--tiles_path",
                        help="Path to directory with tiles files",
                        required=True)
    parser.add_argument("--savedir",
                        help="Where to save whole polygons",
                        required=True)

    parser.set_defaults(func=assemble_polygons_handler)

    args = parser.parse_args()
    args.func(args)
