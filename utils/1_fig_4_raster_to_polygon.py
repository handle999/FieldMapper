from osgeo import gdal, ogr, osr
import os
import numpy as np


def raster_to_polygon(input_raster, output_shapefile):
    # 打开栅格数据集
    src_ds = gdal.Open(input_raster)
    if src_ds is None:
        print("无法打开栅格数据集")
        return

    # 获取栅格数据集的投影信息和地理转换信息
    proj = src_ds.GetProjectionRef()
    geo_transform = src_ds.GetGeoTransform()

    # 读取栅格数据
    band = src_ds.GetRasterBand(1)
    raster_array = band.ReadAsArray()

    # 寻找像素值不为0的像素
    nonzero_pixels = np.where(raster_array != 0)

    # 创建输出矢量文件
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_shapefile):
        driver.DeleteDataSource(output_shapefile)
    out_ds = driver.CreateDataSource(output_shapefile)

    # 创建图层并指定空间参考
    out_srs = osr.SpatialReference()
    out_srs.ImportFromWkt(proj)
    out_layer = out_ds.CreateLayer("polygons", out_srs, geom_type=ogr.wkbPolygon)

    # 添加属性字段
    out_layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    out_layer.CreateField(ogr.FieldDefn("class", ogr.OFTString))
    out_layer.CreateField(ogr.FieldDefn("confidence", ogr.OFTReal))
    out_layer.CreateField(ogr.FieldDefn("Area", ogr.OFTReal))
    out_layer.CreateField(ogr.FieldDefn("Perimeter", ogr.OFTReal))

    # 从栅格数据集中提取多边形并写入矢量文件
    gdal.FPolygonize(band, None, out_layer, 0, [], callback=None)

    # 遍历每个多边形，删除像素点数量小于200的多边形
    out_layer.ResetReading()
    for feature in out_layer:
        geom = feature.GetGeometryRef()
        if geom.GetGeometryType() == ogr.wkbPolygon:
            print("ID: \t", feature.GetFID())
            print("Area: \t", geom.GetArea())  # 获取多边形的面积
            print("Len: \t", geom.Boundary().Length())  # 获取多边形的周长
            area = geom.GetArea()
            print(area)
            if area < 5e-9:
                out_layer.DeleteFeature(feature.GetFID())

    # 关闭数据集
    src_ds = None
    out_ds = None

    print("转换完成")


if __name__ == "__main__":
    # 指定输入栅格数据集（2.tif）和输出矢量文件名
    input_raster = "./polygon/1024_128_SAM/1024_128_SAM_m.tif"
    output_shapefile = "polygon/1024_128_SAM/0_init/1024_128_SAM.shp"

    # 调用函数进行转换
    raster_to_polygon(input_raster, output_shapefile)
