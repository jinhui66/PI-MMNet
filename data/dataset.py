import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import SimpleITK as sitk

class infarction_Dataset(Dataset):
    def __init__(self, excel_file, img_prefix):
        """
        初始化 Dataset
        :param excel_file: str, Excel 文件路径
        :param img_prefix: str, 图片路径前缀
        :param transform: torchvision.transforms, 数据增强和预处理
        """
        self.data = pd.read_excel(excel_file, header=1)  # 使用read_excel代替read_csv
        self.img_prefix = img_prefix
        # print(self.data)

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.data)


    def __getitem__(self, idx):
        """
        获取指定索引的数据
        :param idx: int, 数据索引
        :return: tuple (image, label_index)
        """
        idx = int(idx)
        if isinstance(idx, int):
            row = self.data.iloc[idx]
        else:
            raise TypeError("Index must be an integer.")

        # 构造图片路径
        img_name = int(row.iloc[0])
        # print(img_name)  
        img_path = f"{self.img_prefix}/{img_name}.1.nii.gz"
        # print(img_path)
        # 打开图片
        image = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(image).astype('float32')
        image = torch.from_numpy(image).unsqueeze(0) 

        label = row.iloc[1]
        label = torch.tensor(label).to(torch.int64)  # 转换为整数类型

        table = row.iloc[16:18]
        table = table.fillna(5)  # 用 5 填充 NaN 值
        table = torch.tensor(table).to(torch.float32)  # 转换为浮点数类型
        # text = f"入院分数:{table[0]},出院分数{table[1]}"
        # text = f"In {table[0]}, Out {table[1]}"
        # print(text)
        # table = table.fillna(5)  # 用 5 填充 NaN 值
        # text = tokenizer(text)
        # text = tokenizer(text, truncation=True, max_length=5, return_tensors="pt")

        # text = text['input_ids'].squeeze(0)
        # print(text.shape)

        # print(img_path, image.shape, label)

        # print(text.shape)
        return image, label, table
