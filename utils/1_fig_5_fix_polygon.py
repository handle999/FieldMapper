import os
from osgeo import ogr, osr
from shapely.geometry import Polygon, LineString, mapping


def smooth_polygon(input_shp, output_shp, threshold_distance, threshold_ratio):
    # 打开输入 Shapefile 文件
    input_ds = ogr.Open(input_shp)
    if input_ds is None:
        print("无法打开输入文件:", input_shp)
        return

    # 获取输入图层和投影信息
    in_layer = input_ds.GetLayer()
    in_spatial_ref = in_layer.GetSpatialRef()

    # 创建输出 Shapefile 文件和数据源
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_shp):
        driver.DeleteDataSource(output_shp)
    output_ds = driver.CreateDataSource(output_shp)
    if output_ds is None:
        print("无法创建输出文件:", output_shp)
        return

    # 创建图层并指定空间参考
    out_srs = osr.SpatialReference()
    out_srs.ImportFromWkt(in_spatial_ref.ExportToWkt())  # 使用输入图层的投影信息
    out_layer = output_ds.CreateLayer("smoothed_polygons", out_srs, geom_type=ogr.wkbPolygon)

    # 定义图层的属性表结构，复制输入图层的字段定义
    in_layer_defn = in_layer.GetLayerDefn()
    for i in range(in_layer_defn.GetFieldCount()):
        field_defn = in_layer_defn.GetFieldDefn(i)
        out_layer.CreateField(field_defn)

    for in_feature in in_layer:
        id_field = in_feature.GetField("id")
        print(id_field)
        geom = in_feature.GetGeometryRef()

        if geom is None:
            continue

        geom_ = geom.GetGeometryRef(0)
        polygon = Polygon(geom_.GetPoints())

        smoothed_geom = ogr.Geometry(ogr.wkbPolygon)
        num_points = polygon.exterior.coords.__len__()

        i = 0
        while i < num_points:
            x1, y1 = polygon.exterior.coords[i]

            smooth_point = [x1, y1]

            j = i + 1
            while j < num_points:
                x2, y2 = polygon.exterior.coords[j]

                distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

                if distance < threshold_distance:
                    # line = LineString([smooth_point, (x2, y2)])
                    if j - i < 4:
                        j += 1
                        continue
                    if num_points - j + i < 4:
                        j += 1
                        continue

                    new_polygon_forward = Polygon(polygon.exterior.coords[i:j + 1])
                    new_polygon_reverse = Polygon(polygon.exterior.coords[j + 1:] + polygon.exterior.coords[:i])

                    sub_area_forward = new_polygon_forward.area
                    sub_area_reverse = new_polygon_reverse.area

                    if sub_area_forward < sub_area_reverse:
                        smaller = 1
                        sub_area = sub_area_forward
                    else:
                        smaller = 2
                        sub_area = sub_area_reverse
                    if sub_area == 0:
                        j += 1
                        continue
                    ratio = sub_area / polygon.area

                    if 0.01 * threshold_ratio < ratio < threshold_ratio:
                        print("Delete! ", id_field, i, j)
                        print(ratio)
                        # 修改polygon以反映删除的顶点
                        new_points = list(polygon.exterior.coords)
                        if smaller == 1:
                            del new_points[i+1:j]
                        elif smaller == 2:
                            del new_points[j+1:]
                            del new_points[:i]
                        polygon = Polygon(new_points)
                        num_points = len(new_points)
                        break
                j += 1
            i += 1

        # 复制多边形并添加到输出图层。注意不能直接AddGeometry使用Polygon，因为shapely和ogr不互通，需要转wkb
        out_feature = ogr.Feature(out_layer.GetLayerDefn())
        out_feature.SetGeometry(ogr.CreateGeometryFromWkb(polygon.wkb))
        out_layer.CreateFeature(out_feature)

        out_feature = None
    input_ds = None
    output_ds = None


if __name__ == "__main__":
    # 多边形后处理部分，处理掉毛刺部分、孔洞部分
    input_shp = "polygon/pegbis/0_init/1.shp"
    output_shp = "polygon/pegbis/1_cut/1.shp"
    threshold_distance = 5e-5
    threshold_ratio = 0.1
    smooth_polygon(input_shp, output_shp, threshold_distance, threshold_ratio)
