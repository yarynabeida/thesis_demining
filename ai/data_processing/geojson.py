import argparse
import csv
import json
import os
import numpy as np
import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, shape


def convert_pixel_to_deg(min_val, max_val, pixel_size, num_pixels):
    deg_per_pixel = (max_val - min_val) / (num_pixels / pixel_size)
    return deg_per_pixel


def get_polygon_gps_coords(feature):
    feature_type = feature["geometry"]["type"]
    if feature_type == "Polygon":
        polygon = np.array(feature["geometry"]["coordinates"][0])
    elif feature_type == "MultiPolygon":
        polygon = np.array(feature["geometry"]["coordinates"][0][0])
    else:
        raise ValueError("Only 'Polygon' and 'MultiPolygon' coordinates are supported!")
    return polygon


def convert_polygon_gps_to_epsg(crs_epsg, polygon):
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
    ul_x = geo_matrix[0]
    ul_y = geo_matrix[3]
    x_dist = geo_matrix[1]
    y_dist = geo_matrix[5]
    pixel_x = ((epsg_coord[:, 0] - ul_x) / x_dist).astype(int)
    pixel_y = -((ul_y - epsg_coord[:, 1]) / y_dist).astype(int)
    return np.stack((pixel_x, pixel_y), axis=1)


def get_polygon_area(polygon_px):
    polygon = Polygon(polygon_px)
    polygon_area = polygon.area
    return polygon_area


def convert_csv_to_geojson(input_csv_file, output_geojson_file):
    features = []
    with open(input_csv_file, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            try:
                polygon_str = row['geometry'][9:-2]
                coordinates = [[list(map(float, point.split())) for point in polygon_str.split(',')]]
                geometry = {"type": "Polygon", "coordinates": coordinates}
                properties = {
                    "latitude": float(row['latitude']),
                    "longitude": float(row['longitude']),
                    "area_in_meters": float(row['area_in_meters']),
                    "confidence": float(row['confidence']),
                    "full_plus_code": row['full_plus_code'],
                }
                feature = {"type": "Feature", "geometry": geometry, "properties": properties}
                features.append(feature)
            except Exception as e:
                print(f"Skipping row due to error: {e}")

    geojson_data = {"type": "FeatureCollection", "features": features}

    with open(output_geojson_file, 'w') as geojson_file:
        json.dump(geojson_data, geojson_file, indent=2)

    print(f"Conversion completed. GeoJSON file saved at: {output_geojson_file}")


def convert_geojson_to_geojsonl(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    files = os.listdir(input_folder)
    for file_name in files:
        if file_name.endswith('.geojson'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace('.geojson', '.geojsonl'))
            with open(input_path, 'r') as geojson_file:
                data = json.load(geojson_file)
            with open(output_path, 'w') as geojsonl_file:
                for feature in data.get('features', []):
                    json.dump(feature, geojsonl_file)
                    geojsonl_file.write('\n')


def convert_geojsonl_to_geojson(input_file, output_geojson_file):
    features = []
    with open(input_file, 'r') as geojsonl_file:
        for line in geojsonl_file:
            feature = json.loads(line.strip())
            features.append(feature)

    geojson_data = {"type": "FeatureCollection", "features": features}

    with open(output_geojson_file, 'w') as geojson_file:
        json.dump(geojson_data, geojson_file, indent=2)

    print(f"Conversion completed. GeoJSON file saved at: {output_geojson_file}")


def concatenate_geojson(folder_path, output_file):
    feature_collection = {"type": "FeatureCollection", "features": []}

    for filename in os.listdir(folder_path):
        if filename.endswith(".geojson"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                geojson_data = json.load(file)
            feature_collection["features"].extend(geojson_data["features"])

    with open(output_file, 'w') as output:
        json.dump(feature_collection, output, indent=2)

    print(f'Merged GeoJSON files into {output_file}')


def split_geojson_by_severity(input_file):
    base_folder = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    with open(input_file, 'r') as f:
        data = json.load(f)

    severities = {"severity_1": [], "severity_2": [], "severity_3": [], "severity_minus_1": []}

    for feature in data['features']:
        severity = feature['properties'].get('Severity', -1)
        severity_key = f"severity_{severity}"
        if severity_key in severities:
            severities[severity_key].append(feature)
        else:
            print(f"Unknown severity level: {severity}")

    for severity_key, features in severities.items():
        output_file = os.path.join(base_folder, f"{base_name}_{severity_key}.geojson")
        with open(output_file, 'w') as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)
        print(f"Created {output_file} with {len(features)} features.")


def split_grids(input_folder, output_folder, grid_size):
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    for filename in os.listdir(input_folder):
        if filename.endswith(".geojson"):
            input_file = os.path.join(input_folder, filename)
            gdf = gpd.read_file(input_file)
            bbox = gdf.total_bounds
            grid_width = (bbox[2] - bbox[0]) / grid_size
            grid_height = (bbox[3] - bbox[1]) / grid_size

            batch_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
            os.makedirs(batch_folder, exist_ok=True)

            for i in range(grid_size):
                for j in range(grid_size):
                    xmin, ymin = bbox[0] + i * grid_width, bbox[1] + j * grid_height
                    square_polygon = Polygon([(xmin, ymin), (xmin + grid_width, ymin),
                                              (xmin + grid_width, ymin + grid_height), (xmin, ymin + grid_height)])

                    subset = gdf[gdf.geometry.intersects(square_polygon)]
                    if not subset.empty:
                        output_file = os.path.join(batch_folder,
                                                   f"{os.path.splitext(filename)[0]}_square_{i}_{j}.geojson")
                        subset.to_file(output_file, driver='GeoJSON')
            print(f"Processed {filename} into grid squares.")


def find_geojson_bbox(geojson_file_path, target_crs='EPSG:3857'):
    bbox = None
    with open(geojson_file_path, "rt", encoding='utf-8') as file:
        geojson_data = json.load(file)
        for feature in geojson_data['features']:
            coords = np.array(feature['geometry']['coordinates'][0])
            xmin, ymin = np.min(coords, axis=0)
            xmax, ymax = np.max(coords, axis=0)

            if bbox is None:
                bbox = [xmin, ymin, xmax, ymax]
            else:
                bbox = [min(bbox[0], xmin), min(bbox[1], ymin), max(bbox[2], xmax), max(bbox[3], ymax)]

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    xmin, ymin = transformer.transform(bbox[0], bbox[1])
    xmax, ymax = transformer.transform(bbox[2], bbox[3])
    transformed_bbox = [xmin, ymin, xmax, ymax]
    return transformed_bbox


def generate_tiles(bbox, step, overlap):
    min_x, min_y, max_x, max_y = bbox
    tiles = []

    x = min_x
    while x < max_x:
        y = min_y
        while y < max_y:
            tile_bbox = (x, y, x + step, y + step - overlap)
            tiles.append(tile_bbox)
            y += step - overlap
        x += step - overlap

    return tiles


def filter_tiles_by_geojson(tiles, geojson_file_path):
    filtered_tiles = []
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    polygons = []
    with open(geojson_file_path, "rt", encoding='utf-8') as file:
        for line in file:
            feature = json.loads(line)
            polygons.append(shape(feature['geometry']))

    for tile in tiles:
        tile_poly = Polygon([(tile[0], tile[1]), (tile[0], tile[3]), (tile[2], tile[3]), (tile[2], tile[1])])
        for polygon in polygons:
            if tile_poly.intersects(polygon):
                filtered_tiles.append(tile)
                break

    return filtered_tiles


def csv2geojson_handler(args):
    convert_csv_to_geojson(args.input_csv_file, args.output_geojson_file)


def geojson2geojsonl_handler(args):
    convert_geojson_to_geojsonl(args.input_folder, args.output_folder)


def geojsonl2geojson_handler(args):
    convert_geojsonl_to_geojson(args.input_file, args.output_geojson_file)


def concatenate_geojson_handler(args):
    concatenate_geojson(args.input_folder, args.output_geojson)


def split_geojson_handler(args):
    split_geojson_by_severity(args.input_file)


def split_geojson_grids_handler(args):
    split_grids(args.input_folder, args.output_folder, args.grid_size)


def find_geojson_bbox_handler(args):
    transformed_bbox = find_geojson_bbox(args.geojson_file, args.target_crs)
    print("Transformed bbox (%s): %s" % (args.target_crs, transformed_bbox))


def generate_tiles_handler(args):
    tiles = generate_tiles(args.bbox, args.step, args.overlap)
    if args.geojson is not None:
        tiles = filter_tiles_by_geojson(tiles, args.geojson)
    with open(args.output, 'w') as f:
        for i, tile in enumerate(tiles):
            f.write(f"{i + 1} {tile}\n")
    print(f"Filtered tiles saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_csv2geojson = subparsers.add_parser('csv_to_geojson', help='Convert CSV from googlr to GeoJSON')
    parser_csv2geojson.add_argument("csv_file", help="Input path of the CSV file.")
    parser_csv2geojson.add_argument("geojson_file", help="Output path for the GeoJSON file.")
    parser_csv2geojson.set_defaults(func=csv2geojson_handler)

    parser_geojson2geojsonl = subparsers.add_parser('geojson_to_geojsonl', help='Convert GeoJSON to GeoJSONL')
    parser_geojson2geojsonl.add_argument("--input_folder", help="Input folder containing GeoJSON files.")
    parser_geojson2geojsonl.add_argument("--output_folder", help="Output folder for GeoJSONL files.")
    parser_geojson2geojsonl.set_defaults(func=geojson2geojsonl_handler)

    parser_geojsonl2geojson = subparsers.add_parser('geojsonl_to_geojson', help='Convert GeoJSONL to GeoJSON')
    parser_geojsonl2geojson.add_argument("input_file", help="Input GeoJSONL file path.")
    parser_geojsonl2geojson.add_argument("output_geojson_file", help="Output GeoJSON file path.")
    parser_geojsonl2geojson.set_defaults(func=geojsonl2geojson_handler)

    parser_concatenate_geojson = subparsers.add_parser('concatenate_geojson',
                                                       help='Concatenate all GeoJSON files in a directory into a single '
                                                            'GeoJSON file')
    parser_concatenate_geojson.add_argument("--input_folder", help="Directory containing GeoJSON files.", required=True)
    parser_concatenate_geojson.add_argument("--output_geojson", help="Path for the concatenated GeoJSON output file.",
                                            required=True)
    parser_concatenate_geojson.set_defaults(func=concatenate_geojson_handler)

    parser_split_geojson = subparsers.add_parser('split_geojson',
                                                 help='Split a GeoJSON file into separate files based on Severity property')
    parser_split_geojson.add_argument("--input_file", help="Path to the input GeoJSON file.", required=True)
    parser_split_geojson.set_defaults(func=split_geojson_handler)

    parser_split_grids = subparsers.add_parser('split_geojson_grids', help='Split GeoJSON files into smaller grids')
    parser_split_grids.add_argument("--input_folder", help="Path to the input folder containing GeoJSON files.",
                                    required=True)
    parser_split_grids.add_argument("--output_folder",
                                    help="Path to the output folder for split GeoJSON files.", required=True)
    parser_split_grids.add_argument("--grid_size", type=int,
                                    help="Size of the grid to split each GeoJSON file into.", required=True)
    parser_split_grids.set_defaults(func=split_geojson_grids_handler)

    parser_find_bbox = subparsers.add_parser('find_bbox', help='Find bbox coordinates of GeoJSON')
    parser_find_bbox.add_argument("--geojson_file", help="Path to the input GeoJSON file.", required=True)
    parser_find_bbox.add_argument("--target_crs", default='EPSG:3857')
    parser_find_bbox.set_defaults(func=find_geojson_bbox_handler)

    parser_generate_tiles = subparsers.add_parser('generate_tiles', help="Generate and filter tiles")
    parser_generate_tiles.add_argument('--bbox', type=float, nargs=4, required=True,
                                       help="Bounding box for tile generation (min_x min_y max_x max_y)")
    parser_generate_tiles.add_argument('--step', type=float, default=10000, help="Step size for tile generation")
    parser_generate_tiles.add_argument('--overlap', type=float, default=100, help="Overlap size between tiles")
    parser_generate_tiles.add_argument('--geojson', type=str, help="Path to the GeoJSON file for filtering")
    parser_generate_tiles.add_argument('--output', type=str, required=True, help="Path to save the filtered tiles")
    parser_generate_tiles.set_defaults(func=generate_tiles_handler)
    args = parser.parse_args()
    args.func(args)

