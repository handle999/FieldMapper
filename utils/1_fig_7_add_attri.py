from osgeo import ogr


def add_attributes(input_shapefile, attributes):
    # 打开.shp文件
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(input_shapefile, 1)  # 1表示以读写模式打开

    if dataSource is None:
        print("无法打开输入文件。")
        return

    layer = dataSource.GetLayer()

    # 遍历属性列表，添加新的属性字段
    for attr in attributes:
        field_name = attr["name"]
        field_type = attr["type"]
        default_value = attr["default"]

        layer.CreateField(ogr.FieldDefn(field_name, field_type))

        # 获取新的属性字段的索引
        new_field_index = layer.GetLayerDefn().GetFieldIndex(field_name)

        # 遍历每个要素，为新属性赋予默认值
        for feature in layer:
            feature.SetField(new_field_index, default_value)
            layer.SetFeature(feature)

    # 保存更改并关闭文件
    dataSource.SyncToDisk()
    dataSource = None

    print("新属性添加完成。", attributes)


def delete_attributes(input_shapefile, fields_to_delete):
    # 打开.shp文件
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(input_shapefile, 1)  # 1表示以读写模式打开

    if dataSource is None:
        print("无法打开输入文件。")
        return

    layer = dataSource.GetLayer()

    # 遍历要删除的字段名称列表
    for field_name in fields_to_delete:
        # 获取要素层的定义
        layer_defn = layer.GetLayerDefn()
        # 查找要删除的字段索引
        field_index = layer_defn.GetFieldIndex(field_name)
        if field_index >= 0:
            # 删除字段
            layer.DeleteField(field_index)
        else:
            print(f"字段 '{field_name}' 不存在。")

    # 保存更改并关闭文件
    dataSource.SyncToDisk()
    dataSource = None

    print("属性删除完成。", fields_to_delete)


def rename_attribute(input_shapefile, old_field_name, new_field_name):
    # 打开.shp文件
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(input_shapefile, 1) # 1表示以读写模式打开

    if dataSource is None:
        print("无法打开输入文件。")
        return

    layer = dataSource.GetLayer()

    # 查找要重命名的字段索引
    old_field_index = layer.GetLayerDefn().GetFieldIndex(old_field_name)
    if old_field_index < 0:
        print(f"字段 '{old_field_name}' 不存在。")
        return

    # 创建新的属性字段
    new_field_defn = ogr.FieldDefn(new_field_name, layer.GetLayerDefn().GetFieldDefn(old_field_index).GetType())
    layer.CreateField(new_field_defn)

    # 获取新的属性字段的索引
    new_field_index = layer.GetLayerDefn().GetFieldIndex(new_field_name)

    # 遍历每个要素，将旧属性值复制到新属性
    for feature in layer:
        feature.SetField(new_field_index, feature.GetField(old_field_index))
        layer.SetFeature(feature)

    # 删除旧的属性字段
    layer.DeleteField(old_field_index)

    # 保存更改并关闭文件
    dataSource.SyncToDisk()
    dataSource = None

    print("属性重命名完成。", old_field_name, new_field_name)


if __name__ == "__main__":
    # 添加新属性，主要是几何属性、颜色属性等
    input_shapefile = "./polygon/1024_128/3_hand_label/1024_128_.shp"
    attributes_to_add = [
        {"name": "BoyceClark", "type": ogr.OFTReal, "default": 0.0},
        {"name": "Circular", "type": ogr.OFTReal, "default": 0.0},
        {"name": "NumPoint", "type": ogr.OFTInteger, "default": None},
    ]
    add_attributes(input_shapefile, attributes_to_add)
