import csv
from osgeo import ogr


def save_as_csv(shp_file, output_csv):
    shp_ds = ogr.Open(shp_file)

    if shp_ds is None:
        print("无法打开shapefile文件 '%s'." % shp_file)
        exit(1)

    shp_layer = shp_ds.GetLayer(0)

    # 打开 CSV 文件准备写入
    with open(output_csv, 'w', newline='') as csvfile:
        # 获取要素的字段名
        field_names = [field.name for field in shp_layer.schema]

        # 创建 CSV writer并写入表头
        csv_writer = csv.DictWriter(csvfile, fieldnames=['FID'] + field_names)
        csv_writer.writeheader()

        # 遍历图层中的所有要素
        for feature in shp_layer:
            # 获取要素的 FID
            fid = feature.GetFID()
            # 获取要素的属性
            attributes = feature.items()
            # 将属性添加到字典中
            row_data = {'FID': fid}
            row_data.update(attributes)
            # 写入 CSV 文件
            csv_writer.writerow(row_data)

    print("CSV 文件已保存为 '%s'." % output_csv)


if __name__ == "__main__":
    # 将shapefile文件的所有信息存入csv中（方便后续处理）
    shp_path = "./polygon/1024_128/3_hand_label/1024_128_.shp"
    csv_path = "1024_128_hand.csv"
    save_as_csv(shp_path, csv_path)
