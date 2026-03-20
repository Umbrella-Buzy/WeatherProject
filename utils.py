import yaml
import torch
import numpy as np
import pandas as pd
import math
from torch.utils.data import Dataset
import os
from datetime import datetime

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.params = yaml.safe_load(file)
        self.model = self.params['Global']['model']

    def __getitem__(self, key):
        if key == 'device' and self.params['Global'][key] == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        if key in self.params[self.model]:
            return self.params[self.model][key]
        else:
            return self.params['Global'][key]

    def __setitem__(self, key, value):
        self.params[self.model][key] = value
        self.__setattr__(key, value)

    def __contains__(self, key):
        return key in self.params[self.model]

    def get_all_keys(self):
        return (list(self.params['Global'].keys()) + list(self.params[self.model].keys()))

    def get_all_values(self):
        return [self[key] for key in self.get_all_keys()]


class WeatherDataset(Dataset):
    def __init__(self, config, mode="train"):
        self.config = config
        self.mode = mode
        self.data, self.labels = self.load_and_preprocess_data()

    def load_single_year_data(self, year):
        year_data = []
        hour_data = []
        file_path = os.path.join(self.config["data_path"], f"grid_data_{year}.csv")
        if not os.path.exists(file_path):
            print(f"警告：{file_path} 不存在，跳过该月")
            return
        df = pd.read_csv(file_path)
        variance = 0
        for index, row in df.iterrows():
            row = row.to_numpy()[5:]
            row  = row.reshape(self.config["lat_grid"], self.config["lon_grid"]).astype(np.float32)
            hour_data.append(row)
            if variance % 13 == 12:
                year_data.append(hour_data)
                hour_data = []
            variance += 1
        return year_data

    def load_and_preprocess_data(self):
        """加载2014-2025年数据，按文档划分训练/验证/测试集"""
        all_data = []
        print(f"loading {self.config["start_year"]}-{self.config["end_year"]} {self.mode} data...")
        for year in range(self.config["start_year"], self.config["end_year"] + 1):
            year_data = self.load_single_year_data(year)
            all_data.append(year_data)
        all_data = np.concatenate(all_data, axis=0)  # （总时间步, 特征, 纬度, 经度）

        # 数据归一化（文档提到的待完善步骤）
        self.mean = np.mean(all_data, axis=(0, 2, 3), keepdims=True)
        self.std = np.std(all_data, axis=(0, 2, 3), keepdims=True)
        all_data = (all_data - self.mean) / (self.std + 1e-8)

        # 构建时序样本：输入12步，预测12步
        samples = []
        labels = []
        total_steps = len(all_data) - self.config["time_steps"] - self.config["pred_steps"] + 1
        for i in range(total_steps):
            input_seq = all_data[i:i + self.config["time_steps"], :, :, :]
            label_seq = all_data[i + self.config["time_steps"]:i + self.config["time_steps"] + self.config["pred_steps"], :, :,
                        :]
            samples.append(input_seq)
            labels.append(label_seq)

        samples = np.array(samples)  # （样本数, 输入步, 特征, 纬度, 经度）
        labels = np.array(labels)  # （样本数, 预测步, 特征, 纬度, 经度）

        # 按文档划分数据集：2014-2021=训练，2022-2023=验证，2024-2025=测试
        train_split = int(len(samples) * self.config["train_ratio"])
        val_split = train_split + int(len(samples) * self.config["val_ratio"])
        if self.mode == "train":
            return samples[:train_split], labels[:train_split]
        elif self.mode == "val":
            return samples[train_split:val_split], labels[train_split:val_split]
        else:
            return samples[val_split:], labels[val_split:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 转换为tensor，维度调整为（输入步, 纬度, 经度, 特征）适配模型输入
        data = torch.tensor(self.data[idx], dtype=torch.float32).permute(0, 2, 3, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).permute(0, 2, 3, 1)
        return data, label


