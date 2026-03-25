import torch
import numpy as np
import math
import calendar
from torch.utils.data import Dataset
import os
import torch.nn as nn

class WeatherDataset(Dataset):
    def __init__(self, config, mode="train"):
        self.config = config
        self.mode = mode
        self.data, self.dec_input, self.labels= self.load_data()

    def load_data(self):
        cities = ["Beijing", "Chengdu", "Guangzhou", "Shanghai", "Tianjin", "Chongqing"]
        samples = []
        labels = []
        dec_inputs = []
        for city in cities:
            weather_data = np.load(os.path.join(self.config["data_path"] + city + ".npy"))
            total_steps = weather_data.shape[1] - self.config["time_steps"] - self.config["max_steps"] + 1
            for i in range(total_steps):
                input = weather_data[:, i:i + self.config["time_steps"], :]
                dec = weather_data[:, i + self.config["time_steps"] - 1:i + self.config["time_steps"] + self.config["max_steps"] - 1, 6:]
                label = weather_data[:, i + self.config["time_steps"]:i + self.config["time_steps"] + self.config["max_steps"], 6:]
                for j in range(len(input)):
                    samples.append(input[j])
                    dec_inputs.append(dec[j])
                    labels.append(label[j])

        '''
        train_split = int(len(samples) * self.config["train_ratio"])
        if self.mode == "train":
            return samples[:train_split], dec_inputs[:train_split], labels[:train_split]
        else:
            return samples[train_split:], dec_inputs[train_split:], labels[train_split:]
        '''
        train_split = int(len(samples) * 0.05)
        if self.mode == "train":
            return samples[:train_split], dec_inputs[:train_split], labels[:train_split]
        else:
            return samples[-train_split:], dec_inputs[-train_split:], labels[-train_split:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        dec_inputs = torch.tensor(self.dec_input[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return data, dec_inputs, label

class Preprocessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lon = None
        self.lat = None
        self.time = None

    def forward(self, x):
        return x

    def clip_and_process_time_space_info(self, x):
        self.lon = x[:, :, 0]
        self.lat = x[:, :, 1]
        y = x[2, :].numpy()
        m = x[3, :].numpy()
        d = x[4, :].numpy()
        h = x[5, :]
        days = []
        for month, year,  in zip(m, y):
            days_in_month = calendar.monthrange(year, int(month))[1]
            days.append(days_in_month)

def get_correct(pred_data, real_data):
    abs_diff = torch.abs(pred_data - real_data)
    is_correct = (abs_diff < 5).all(dim=2)
    correct_counts = is_correct.sum(dim=0)
    return correct_counts.tolist()