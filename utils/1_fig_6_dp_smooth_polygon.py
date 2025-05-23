from osgeo import ogr


# Douglas-Peucker算法近似多边形函数
def douglas_peucker(geom, epsilon):
    # 判断是否是多边形几何
    if geom.GetGeometryType() != ogr.wkbPolygon:
        return geom
    # 执行Douglas-Peucker算法
    simplified_geom = geom.SimplifyPreserveTopology(epsilon)
    return simplified_geom


def smooth_polygon(input_shapefile, output_shapefile, eps):
    # 打开输入.shp文件
    input_ds = ogr.Open(input_shapefile)
    input_layer = input_ds.GetLayer()

    # 创建输出.shp文件
    driver = ogr.GetDriverByName("ESRI Shapefile")
    output_ds = driver.CreateDataSource(output_shapefile)
    output_layer = output_ds.CreateLayer("polygon", geom_type=ogr.wkbPolygon)

    # 复制输入.shp文件的字段
    for i in range(input_layer.GetLayerDefn().GetFieldCount()):
        field_defn = input_layer.GetLayerDefn().GetFieldDefn(i)
        output_layer.CreateField(field_defn)

    # 遍历要素并近似多边形
    for feature in input_layer:
        geometry = feature.GetGeometryRef()
        simplified_geom = douglas_peucker(geometry, epsilon=eps)  # 调整 epsilon 值以控制近似程度

        # 创建新要素并设置几何和属性
        new_feature = ogr.Feature(output_layer.GetLayerDefn())
        new_feature.SetGeometry(simplified_geom)
        for i in range(feature.GetFieldCount()):
            new_feature.SetField(i, feature.GetField(i))

        # 将新要素写入输出图层
        output_layer.CreateFeature(new_feature)
        new_feature = None

    # 关闭数据源
    input_ds = None
    output_ds = None

    print("Approximation completed.")


if __name__ == "__main__":
    # 输入.shp文件路径，平滑.shp并输出
    input_shapefile = "polygon/pegbis/1_cut/1.shp"
    output_shapefile = "polygon/pegbis/2_cut2/1.shp"
    eps = 5e-5
    smooth_polygon(input_shapefile, output_shapefile, eps)
