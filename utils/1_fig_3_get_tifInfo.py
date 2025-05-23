from osgeo import gdal


def set_georeference(src_tif, target_tif):
    # 打开源 TIFF 文件
    src_ds = gdal.Open(src_tif, gdal.GA_ReadOnly)
    if src_ds is None:
        print("无法打开源 TIFF 文件")
        return False

    # 获取源 TIFF 的地理参考信息
    geotransform = src_ds.GetGeoTransform()
    spatial_ref = src_ds.GetProjection()

    # 打开目标 TIFF 文件
    target_ds = gdal.Open(target_tif, gdal.GA_Update)
    if target_ds is None:
        print("无法打开目标 TIFF 文件")
        return False

    # 设置目标 TIFF 的地理参考信息
    target_ds.SetGeoTransform(geotransform)
    target_ds.SetProjection(spatial_ref)

    # 关闭数据集
    src_ds = None
    target_ds = None

    print("地理参考信息已成功设置")
    return True


def set_png_georeference(src_tif, target_img, output_tif):
    # 打开源 TIFF 文件
    src_ds = gdal.Open(src_tif, gdal.GA_ReadOnly)
    if src_ds is None:
        print("无法打开源 TIFF 文件")
        return False

    # 获取源 TIFF 的地理参考信息
    geotransform = src_ds.GetGeoTransform()
    spatial_ref = src_ds.GetProjection()

    # 打开目标 PNG 文件
    target_ds = gdal.Open(target_img, gdal.GA_ReadOnly)
    if target_ds is None:
        print("无法打开目标 PNG 文件")
        return False

    # 获取目标 PNG 的尺寸
    cols = target_ds.RasterXSize
    rows = target_ds.RasterYSize

    # 创建输出 TIFF 文件
    driver = gdal.GetDriverByName("GTiff")
    output_ds = driver.Create(output_tif, cols, rows, 3, gdal.GDT_Byte)

    # 设置地理参考信息
    output_ds.SetGeoTransform(geotransform)
    output_ds.SetProjection(spatial_ref)

    # 将目标 PNG 的像素数据写入输出 TIFF
    for i in range(1, 4):  # 3 bands (assuming RGB)
        band_data = target_ds.GetRasterBand(i).ReadAsArray()
        output_ds.GetRasterBand(i).WriteArray(band_data)

    # 关闭数据集
    src_ds = None
    target_ds = None
    output_ds = None

    print("地理参考信息已成功设置")
    return True


if __name__ == "__main__":
    # 设置图2的地理参考信息与图1的相同
    if set_georeference("./polygon/pegbis/z18-google.tif", "./polygon/pegbis/z18-0.5-500-50.tif"):
        print("图2已成功设置地理参考信息")
    else:
        print("无法设置图2的地理参考信息")
