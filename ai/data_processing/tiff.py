from osgeo import gdal
from pyproj import CRS, Transformer
import tifffile as tiff


def read_tiff(input_tiff):
    dataset = gdal.Open(input_tiff)
    if dataset is None:
        raise Exception("Could not open the input TIFF file.")

    width = dataset.RasterXSize
    height = dataset.RasterYSize

    big_image = tiff.imread(input_tiff)
    print(f"{input_tiff} is read")
    return dataset, width, height, big_image


def get_tif_extent(data_source):
    width = data_source.RasterXSize
    height = data_source.RasterYSize
    gt = data_source.GetGeoTransform()
    minx_tiff = gt[0]
    miny_tiff = gt[3] + width * gt[4] + height * gt[5]
    maxx_tiff = gt[0] + width * gt[1] + height * gt[2]
    maxy_tiff = gt[3]
    return minx_tiff, miny_tiff, maxx_tiff, maxy_tiff


def get_tif_extent_gps(dataset):
    crs_epsg = get_tiff_crs_epsg(dataset)
    minx_tiff, miny_tiff, maxx_tiff, maxy_tiff = get_tif_extent(dataset)
    minx_tiff_gps, miny_tiff_gps, maxx_tiff_gps, maxy_tiff_gps = convert_bbox_tiffepsg_to_gps(crs_epsg, (
    minx_tiff, miny_tiff, maxx_tiff, maxy_tiff))
    return minx_tiff_gps, miny_tiff_gps, maxx_tiff_gps, maxy_tiff_gps


def get_tiff_crs_epsg(data_source):
    crs_wkt = data_source.GetProjection()
    crs = CRS.from_wkt(crs_wkt)
    epsg_code = crs.to_epsg()
    return epsg_code


def convert_bbox_tiffepsg_to_gps(crs_epsg, bbox):
    epsg = CRS(crs_epsg)
    gps = CRS('EPSG:4326')

    min_x, min_y, max_x, max_y = bbox
    transformer = Transformer.from_crs(epsg, gps, always_xy=True)
    min_lon, min_lat = transformer.transform(min_x, min_y)
    max_lon, max_lat = transformer.transform(max_x, max_y)
    gps_bbox = (min_lon, min_lat, max_lon, max_lat)
    return gps_bbox

