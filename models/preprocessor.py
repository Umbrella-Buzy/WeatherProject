import torch
import torch.nn as nn
from datetime import datetime

class preprocessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        return x


class time_space_embedding(nn.Module):
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
        y_arr = x[2, :].numpy().astype(int)  # 年
        m_arr = x[3, :].numpy().astype(int)  # 月
        d_arr = x[4, :].numpy().astype(int)  # 日
        h_arr = x[5, :].numpy().astype(int)  # 时

        # 方法2.1：循环计算（简单但慢）
        hours_list = []
        for y, m, d, h in zip(y_arr, m_arr, d_arr, h_arr):
            start = datetime(y, 1, 1, 0, 0, 0)
            current = datetime(y, m, d, h, 0, 0)
            hours = int((current - start).total_seconds() / 3600)
            hours_list.append(hours)