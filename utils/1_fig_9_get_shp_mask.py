import os
from osgeo import ogr, gdal


def shp_mask_all(input_vector, output_dir, existing_raster):
    # 获取输入矢量数据源
    source_ds = ogr.Open(input_vector)
    source_layer = source_ds.GetLayer()

    # 打开一个已存在的栅格文件
    existing_ds = gdal.Open(existing_raster)
    # 获取已存在栅格文件的宽度和高度
    width = existing_ds.RasterXSize
    height = existing_ds.RasterYSize
    print(width, height)

    # 获取要素计数
    feature_count = source_layer.GetFeatureCount()

    # 遍历每个要素
    for i in range(feature_count):
        # 获取要素
        feature = source_layer.GetFeature(i)

        # 获取要素ID
        fid = feature.GetFID()
        print(fid)

        # 生成输出栅格文件名
        output_raster = os.path.join(output_dir, f'{fid}.tif')

        # 获取要素几何
        geometry = feature.GetGeometryRef()

        # 创建栅格数据源
        driver = gdal.GetDriverByName('GTiff')
        target_ds = driver.Create(output_raster, width, height, 1, gdal.GDT_Byte, options=['COMPRESS=LZW'])

        # 获取仿射变换参数
        geotransform = existing_ds.GetGeoTransform()

        # 设置栅格数据源的投影和仿射变换
        target_ds.SetProjection(existing_ds.GetProjection())
        target_ds.SetGeoTransform(geotransform)

        # 创建筛选器，只选择当前要素
        source_layer.SetAttributeFilter(f"FID={fid}")

        # 使用GDALRasterizeLayer函数进行矢量转栅格
        gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])

        # 关闭数据源
        target_ds = None

    # 关闭输入矢量数据源
    source_ds = None
    existing_ds = None


if __name__ == "__main__":
    # 将一个shapefile文件的所有polygon，获取对应部分的mask，背景用0表示
    input_vector = './polygon/1024_128/2_traj_label/1024_128_.shp'
    output_dir = './polygon/1024_128/9_this/'
    # 打开一个已存在的栅格文件
    existing_raster = './data/z20-google.tif'
    shp_mask_all(input_vector, output_dir, existing_raster)
