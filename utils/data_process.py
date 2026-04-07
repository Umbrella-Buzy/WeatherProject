import torch
import numpy as np
import math
import calendar
import pandas as pd
from torch.utils.data import Dataset
from datetime import datetime
import os
import torch.nn as nn

def normalize(weather_data):
    aqi_data = weather_data[:, -6:]

    log_aqi = np.log(aqi_data)
    log_min = np.log(1)
    log_max = np.log(500)
    aqi_normalized = (log_aqi - log_min) / (log_max - log_min)

    lon_arr = weather_data[:, 0]
    lat_arr = weather_data[:, 1]
    y_arr = weather_data[:, 2].astype(int)
    m_arr = weather_data[:, 3].astype(int)
    d_arr = weather_data[:, 4].astype(int)
    h_arr = weather_data[:, 5].astype(int)
    features = []
    for lon, lat, y, m, d, h in zip(lon_arr, lat_arr, y_arr, m_arr, d_arr, h_arr):
        start = datetime(y, 1, 1, 0, 0, 0)
        current = datetime(y, m, d, 0, 0, 0)
        days = np.array((current - start).total_seconds() / (3600 * 24))
        years = np.array((y - 2000) / 100)
        lon_norm = np.array((lon - 73) / (135 - 73))
        lat_norm = np.array((lat - 4) / (53 - 4))
        h = np.array(h / 24)
        features.append(np.concatenate([[lon_norm], [lat_norm], [years], [days], [h]]))
    result = np.hstack([np.array(features), aqi_normalized])
    return result

class WeatherDataset(Dataset):
    def __init__(self, config, mode="train", do_print=True):
        self.config = config
        self.mode = mode
        self._print = do_print
        self.data, self.dec_input, self.labels= self.load_data()

    def load_data(self):
        cities = ["Beijing", "Chengdu", "Guangzhou", "Shanghai", "Tianjin", "Chongqing"]
        station_nums = [20, 11, 20, 17, 19, 23]
        required_traits = (["经度", "维度", "年份", "月份", "日期", "小时",
                            "PM2.5_AQI", "PM10_AQI", "SO2_AQI", "NO2_AQI", "O3_AQI", "CO_AQI"])
        samples = []
        labels = []
        dec_inputs = []
        for city, station_num in zip(cities, station_nums):
            if self._print:
                print(f'processing {city}...')
            df = pd.read_csv(os.path.join("./data/raw_data", city + ".csv"))
            df = df[required_traits]
            station_data = [normalize(df.iloc[i::station_num].to_numpy()) for i in range(station_num)]
            weather_data = np.stack(station_data,axis=0)
            total_steps = weather_data.shape[1] - self.config["time_steps"] - self.config["max_steps"] + 1
            for i in range(total_steps):
                input = weather_data[:, i:i + self.config["time_steps"], :]
                dec = weather_data[:, i + self.config["time_steps"] - 1:i + self.config["time_steps"] + self.config["max_steps"] - 1, -6:]
                label = weather_data[:, i + self.config["time_steps"]:i + self.config["time_steps"] + self.config["max_steps"], -6:]
                for j in range(len(input)):
                    samples.append(input[j])
                    dec_inputs.append(dec[j])
                    labels.append(label[j])

        if self.config["mode"] != "debug":
            train_split = int(len(samples) * self.config["train_ratio"])
            if self.mode == "train":
                return samples[:train_split], dec_inputs[:train_split], labels[:train_split]
            else:
                return samples[train_split:], dec_inputs[train_split:], labels[train_split:]
        else:
            train_split = int(len(samples) * 0.01)
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

def get_correct(pred_data, real_data):
    log_min = math.log(1)
    log_max = math.log(500)
    pred_denormalized = torch.round(torch.exp(pred_data * (math.log(500) - math.log(1)) + math.log(1)))
    real_denormalized = torch.round(torch.exp(real_data * (math.log(500) - math.log(1)) + math.log(1)))
    abs_diff = torch.abs(pred_denormalized - real_denormalized)
    is_correct10 = (abs_diff <= 10).all(dim=2)
    is_correct5  = (abs_diff <= 5 ).all(dim=2)
    is_correct1  = (abs_diff <= 1 ).all(dim=2)

    correct_counts10 = is_correct10.sum(dim=0).cpu().numpy()
    correct_counts5  = is_correct5 .sum(dim=0).cpu().numpy()
    correct_counts1  = is_correct1 .sum(dim=0).cpu().numpy()
    return np.stack([correct_counts10, correct_counts5, correct_counts1])