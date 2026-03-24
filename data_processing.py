import pandas as pd
import os
import numpy as np

cities = ["Beijing", "Chengdu", "Guangzhou", "Shanghai", "Tianjin", "Chongqing"]
station_nums = [20, 11, 20, 17, 19, 23]
#station_nums = [20]
#cities = ["Beijing"]
required_traits = (["经度", "维度", "年份", "月份", "日期", "小时",
                    "PM2.5_AQI", "PM10_AQI", "SO2_AQI", "NO2_AQI", "O3_AQI", "CO_AQI"])

for city, station_num in zip(cities, station_nums):
    print(f'processing {city}...')
    df = pd.read_csv(os.path.join("./data/raw_data", city + ".csv"))
    df = df[required_traits]
    station_data = [df.iloc[i::station_num].to_numpy() for i in range(station_num)]
    stacked_station_data = np.stack(station_data,axis=0)
    save_path = os.path.join("./data/", city + ".npy")
    np.save(save_path, stacked_station_data)
    print(city + " done!")
