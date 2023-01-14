import os
import json
import pandas


def read_csv(csv_dir: str, csv_name: str):
    # 读取csv文件
    data = pandas.read_csv(os.path.join(csv_dir, csv_name))
    # 获取类标签
    label_set = set(data['label'].drop_duplicates().values)
    # 输出文件信息
    print('{} have {} images and {} classes.'.format(csv_name, data.shape[0], len(label_set)))
    return data, label_set


def calculate(path: str, label_dict: dict, rate: float = 0.2):
    # 读取图片
    image_dir = os.path.join(path, 'images')
    image_list = [i for i in os.listdir(image_dir) if i.endswith('.jpg')]
    # 输出图片数据信息
    print('Find {} images in Mini-ImageNet Dataset.'.format(len(image_list)))
    # 获取原有标签文件
    train_data, train_label = read_csv(path, 'train.csv')
    val_data, val_label = read_csv(path, 'val.csv')
    test_data, test_label = read_csv(path, 'test.csv')
    # 合并train、val、test标签
    labels = (train_label | val_label | test_label)
    labels = list(labels)
    labels.sort()
    print('All classes : {}'.format(len(labels)))
    # 新建索引标签文件 classes_name.json
    classes_label = dict([(label, [index, label_dict[label]]) for index, label in enumerate(labels)])
    json_str = json.dumps(classes_label)
    with open('./data/Mini-ImageNet/classes_name.json', 'w') as json_file:
        json_file.write(json_str)
    # 合并train、val、test数据
    data = pandas.concat([train_data, val_data, test_data], axis=0)
    print("All data : {}".format(len(data)))
    # 划分新的训练集和测试集
    split_train_data = []
    split_val_data = []
    for label in labels:
        # 获取标签对应图片信息
        class_data = data[data['label'] == label]
        # 按照比例划分数据集
        num_train_sample = int(class_data.shape[0] * (1 - rate))
        split_train_data.append(class_data[:num_train_sample])
        split_val_data.append(class_data[num_train_sample:])
    # 合并新划分的train、val数据
    new_train_data = pandas.concat(split_train_data, axis=0)
    new_val_data = pandas.concat(split_val_data, axis=0)
    # 新建索引标签文件 new_train.csv new_val.csv
    new_train_data.to_csv(os.path.join(path, "new_train.csv"))
    new_val_data.to_csv(os.path.join(path, "new_val.csv"))


def main():
    # 数据集根目录
    data_dir = './data/Mini-ImageNet/'
    # 索引标签文件
    json_path = './data/Mini-ImageNet/imagenet_class_index.json'
    # 获取索引标签
    label_dict = json.load(open(json_path, 'r', encoding='utf-8'))
    label_dict = dict([(v[0], v[1]) for k, v in label_dict.items()])
    # 重组数据集
    calculate(data_dir, label_dict)
