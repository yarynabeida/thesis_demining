import os
import json
import argparse

import cv2
import glob
import numpy as np

from ai.data_processing.tiff import read_tiff, get_tiff_crs_epsg
from ai.data_processing.geojson import get_polygon_gps_coords, convert_polygon_gps_to_epsg, epsg_to_pixel, \
    get_polygon_area


def create_directories(output_directory):
    tiles_savepath = os.path.join(output_directory, "images")
    masks_savepath = os.path.join(output_directory, "masks")

    if not os.path.exists(tiles_savepath):
        os.makedirs(tiles_savepath)
    if not os.path.exists(masks_savepath):
        os.makedirs(masks_savepath)

    return tiles_savepath, masks_savepath


def get_polygons_from_geojson(outlines_data, data_source, crs_epsg):
    px_polygons = []

    for i, line in enumerate(outlines_data):
        feature = json.loads(line)
        try:
            polygon_gps = get_polygon_gps_coords(feature)
            polygon_epsg = convert_polygon_gps_to_epsg(crs_epsg, polygon_gps)
            polygon_px = epsg_to_pixel(data_source.GetGeoTransform(), polygon_epsg)

            px_polygons.append(polygon_px)

        except Exception:
            print("Skipping: bad polygon!")
            continue

    return px_polygons


def get_tiles_number(height, width, tile_width, tile_height):
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


def crop_on_tiles(big_tiff, mask_image, tile_width, tile_height, resize_size):

    tiff_width, tiff_height = big_tiff.shape[:2]
    num_tiles_x, num_tiles_y, total_tiles = get_tiles_number(tiff_height, tiff_width, tile_width, tile_height)

    tiff_tiles = []
    mask_tiles = []
    tiles_coords = []

    for i in range(num_tiles_x):
        for j in range(num_tiles_y):

            left = min(i * tile_width, tiff_width - tile_width)
            upper = min(j * tile_height, tiff_height - tile_height)
            right = min((i + 1) * tile_width, tiff_width)
            lower = min((j + 1) * tile_height, tiff_height)

            tiff_tile = big_tiff[upper:lower, left:right, :3]
            mask_tile = mask_image[upper:lower, left:right]

            resized_tile = cv2.resize(tiff_tile, resize_size)
            resized_mask = cv2.resize(mask_tile, resize_size, interpolation=cv2.INTER_NEAREST)

            tiff_tiles.append(resized_tile)
            mask_tiles.append(resized_mask)
            tiles_coords.append([int(left), int(upper)])

    return tiff_tiles, mask_tiles, tiles_coords


def create_mask(big_tiff, polygons, data_config):
    tiff_w, tiff_h = big_tiff.shape[0], big_tiff.shape[1]
    mask_big = np.zeros((tiff_w, tiff_h), dtype=np.uint8)

    min_area = 0 if data_config["area_constraints"]["min_area_pct"] is None else \
        tiff_w * tiff_h * data_config["area_constraints"]["min_area_pct"] // 100

    for polygon in polygons:
        polygon_area = get_polygon_area(polygon)
        if polygon_area < min_area:
            continue

        cv2.fillPoly(mask_big, [polygon], 155)

    result_mask = mask_big
    padding_w, padding_h = int(tiff_w * data_config["area_constraints"]["padding_pct"] // 100), \
        int(tiff_h * data_config["area_constraints"]["padding_pct"] // 100)

    if not np.any(result_mask[padding_w: tiff_w - padding_w, padding_h: tiff_h - padding_h]):
        raise Exception("All polygons are out of padding")

    return result_mask


def create_tiff_mask(tiff_file, data_config, config, geojsonl_dir, folder_name, tiles_dir, masks_dir):

    tif_name = os.path.splitext(os.path.basename(tiff_file))[0]
    dataset, width, height, big_image = read_tiff(tiff_file)
    crs_epsg = get_tiff_crs_epsg(dataset)

    geojsonl_file = os.path.join(geojsonl_dir, tif_name + ".geojsonl")
    outlines_data = open(geojsonl_file, "rt", encoding="utf-8")

    px_polygons = get_polygons_from_geojson(outlines_data, dataset, crs_epsg)

    mask = create_mask(big_image, px_polygons, data_config)

    maskfilename = f"{folder_name}_{tif_name}.png"
    maskpath = os.path.join(masks_dir, maskfilename)
    cv2.imwrite(maskpath, mask)

    tile_width, tile_height = data_config["image_processing"]["tile_size"]
    resize_size = data_config["image_processing"]["resize_size"]

    tiles, masks, tiles_coords = crop_on_tiles(big_image, mask, tile_width, tile_height, resize_size)

    for idx, tile in enumerate(tiles):
        minx_px, miny_px = tiles_coords[idx]

        tilefilename = f"{folder_name}_{tif_name}_{minx_px}_{miny_px}.png"
        tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        tilepath = os.path.join(tiles_dir, tilefilename)
        cv2.imwrite(tilepath, tile_bgr)

        maskfilename = f"{folder_name}_{tif_name}_{minx_px}_{miny_px}.png"
        maskpath = os.path.join(masks_dir, maskfilename)
        cv2.imwrite(maskpath, masks[idx])
        print(f"Saved Tile: {tilefilename}.")


def tiler_handler(args):
    if args.data_config:
        data_config_path = os.path.normpath(args.data_config)
        if not os.path.isfile(data_config_path):
            print("Data configuration file %s cannot be opened." % data_config_path)
            return
        file = open(data_config_path, 'r')
        data_config = json.load(file)
    else:
        data_config = {}

    if args.tile_size:
        data_config["image_processing"]["tile_size"] = args.tile_size
    if args.resize_size:
        data_config["image_processing"]["resize_size"] = args.resize_size
    if args.overlap_size:
        data_config["image_processing"]["overlap_size"] = args.overlap_size
    if args.merge_kernel_size:
        data_config["morphological_operations"]["merge_kernel_size"] = args.merge_kernel_size
    if args.ext_kernel_size:
        data_config["morphological_operations"]["ext_kernel_size"] = args.ext_kernel_size
    if args.min_area_pct:
        data_config["area_constraints"]["min_area_pct"] = args.min_area_pct
    if args.max_area_pct:
        data_config["area_constraints"]["max_area_pct"] = args.max_area_pct
    if args.padding_pct:
        data_config["area_constraints"]["padding_pct"] = args.padding_pct
    if args.area_splitter_pct:
        data_config["area_constraints"]["area_splitter_pct"] = args.area_splitter_pct
    if args.num_classes:
        data_config["num_classes"] = args.num_classes

    tiff_dir = os.path.normpath(args.tiff_dir)
    if args.geojsonl_dir:
        geojsonl_dir = os.path.normpath(args.geojsonl_dir)
        if not os.path.isdir(geojsonl_dir):
            print("Geojson dir %s cannot be opened." % geojsonl_dir)
    else:
        config = os.path.normpath(args.config)
        if not os.path.isfile(config):
            print("Config file %s cannot be opened." % config)
            return

    output_dir = os.path.normpath(args.output_dir)
    tiles_dir, masks_dir = create_directories(output_dir)

    tiff_files = glob.glob(os.path.join(tiff_dir, "*.tif"))
    folder_name = os.path.basename(tiff_dir)

    for tiff_file in tiff_files:
        create_tiff_mask(tiff_file, data_config, args.config, args.geojsonl_dir, folder_name, tiles_dir, masks_dir)


def visualizations_handler(args):
    data_dir = os.path.normpath(args.data_dir)
    image_folder = os.path.join(data_dir, 'images')
    mask_folder = os.path.join(data_dir, 'masks')

    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.png')]
    mask_files = [os.path.join(mask_folder, file) for file in os.listdir(mask_folder) if file.endswith('.png')]

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for i in range(min(args.num_vis, len(image_files))):
        image = cv2.imread(image_files[i], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_files[i], cv2.IMREAD_UNCHANGED)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.applyColorMap(mask_bgr, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(image, 0.5, mask_bgr, 0.5, 0)

        output_path = os.path.join(args.output_folder, f"visualization_{i}.png")
        cv2.imwrite(output_path, overlay)

        print(f"Saved visualization {i} to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_tiler = subparsers.add_parser("tiler")
    parser_tiler.add_argument("--tiff_dir", help="Directory containing TIFF files to be processed.", required=True)
    parser_tiler.add_argument("--geojsonl_dir", help="Directory containing GeoJSON files.")
    parser_tiler.add_argument("--config", help="Configuration file path for PostGIS database access.")
    parser_tiler.add_argument("--data_config", help="JSON configuration file path for data processing parameters.")
    parser_tiler.add_argument("--output_dir", help="Directory for saving processed tiles and masks.",  required=True)

    parser_tiler.add_argument("--task",
                              help="Task for segmentation. Options: 'areas_segmentation', 'outlines_segmentation'.")
    parser_tiler.add_argument("--num_classes",
                              help="Number of segmentation classes. Options: 2 for binary, 3 for ternary", type=int)
    parser_tiler.add_argument("--tile_size", help="Size of the tiles in pixels as a tuple (height, width).",
                              nargs=2, type=int)
    parser_tiler.add_argument("--resize_size",
                              help="Size to which tiles are resized in pixels as a tuple (height, width).",
                              nargs=2, type=int)
    parser_tiler.add_argument("--overlap_size",
                              help="Size of overlap between tiles in pixels as a tuple (height, width).",
                              nargs=2, type=int)
    parser_tiler.add_argument("--merge_kernel_size",
                              help="Size of the kernel used for merging neighboring outlines, tuple (height, width).",
                              nargs=2, type=int)
    parser_tiler.add_argument("--ext_kernel_size",
                              help="Size of the external kernel used for expanding blobs as a tuple (height, width).",
                              nargs=2, type=int)
    parser_tiler.add_argument("--min_area_pct",
                              help="Minimum area percentage to filter out smaller outlines.", type=float)
    parser_tiler.add_argument("--max_area_pc",
                              help="Maximum area percentage to filter out larger outlines.", type=float)
    parser_tiler.add_argument("--padding_pct",
                              help="Percentage of padding to exclude tiles containing only border outlines.", type=int)

    parser_tiler.set_defaults(func=tiler_handler)

    parser_vis = subparsers.add_parser("vis", description="Generate visualizations for images and their masks.")
    parser_vis.add_argument("--data_dir", required=True, help="Directory containing the 'images' and 'masks' folders.")
    parser_vis.add_argument("--output_folder", required=True, help="Folder to save the visualization results.")
    parser_vis.add_argument("--num_vis", type=int, default=1, help="Number of visualizations to generate.")

    parser_vis.set_defaults(func=visualizations_handler)

    args = parser.parse_args()
    args.func(args)
