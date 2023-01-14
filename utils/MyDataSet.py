import os
import json
import pandas
from PIL import Image
from torch.utils.data import Dataset


class MyDataSet(Dataset):

    def __init__(self,
                 root_dir: str,
                 csv_name: str,
                 json_path: str,
                 transform=None
                 ):
        # 读取图片路径
        images_dir = os.path.join(root_dir, 'images')
        # 获取csv文件路径
        csv_path = os.path.join(root_dir, csv_name)
        # 检查路径是否正确
        assert os.path.exists(images_dir), "dir:'{}' not found.".format(images_dir)
        assert os.path.exists(json_path), "file:'{}' not found.".format(json_path)
        assert os.path.exists(csv_path), "file:'{}' not found.".format(csv_path)
        # 获取类标签索引
        self.label_dict = json.load(open(json_path, "r", encoding='utf-8'))
        # 获取csv文件数据
        csv_data = pandas.read_csv(csv_path)
        self.total_num = csv_data.shape[0]
        self.img_paths = [os.path.join(images_dir, i) for i in csv_data['filename'].values]
        self.img_labels = [self.label_dict[i][0] for i in csv_data['label'].values]
        # Transforms参数
        self.transform = transform

    def __len__(self):
        return self.total_num

    def __getitem__(self, item):
        # 获取图片
        image = Image.open(self.img_paths[item])
        # 获取标签
        label = self.img_labels[item]
        # 判断是否需要Transform
        if self.transform is not None:
            image = self.transform(image)
        return image, label
