import pandas as pd
import numpy as np

# In order to minimize the size of the project, the raw data were removed.
# This program is for reference only
all_data = pd.DataFrame()
for i in range(2014, 2026):
    print(f'start processing data of year {i}')
    meteorology_data = pd.read_csv(f"./data/grid_data_{i}_1.csv", dtype=str)
    pollution_data = pd.read_csv(f"./data/grid_wide_{i}.csv", dtype=str)
    start_month = "1" if i != 2014 else "6"
    meteorology_data = meteorology_data[meteorology_data["M"] >= start_month]
    pollution_data = pollution_data[pollution_data["month"] >= start_month]
    total_meteorology_len = len(meteorology_data)
    total_pollution_len =len(pollution_data)
    columns = ['year', 'month', 'day', 'hour', 'variable']
    grid_heads = [f'grid_{j}' for j in range(1, 196)]
    columns += grid_heads
    meteorology_data.columns = columns
    pollution_data.columns = columns
    start_meteorology = 0
    start_pollution = 8
    while (start_meteorology < total_meteorology_len) and (start_pollution < total_pollution_len):
        meteorology_chunk = meteorology_data.iloc[start_meteorology:start_meteorology + 5]
        pollution_chunk = pollution_data.iloc[start_pollution:start_pollution + 8]
        if meteorology_chunk['hour'].tolist()[0] == '24':
            meteorology_chunk.loc[:,'hour'] = '0'
            day = pollution_chunk['day'].tolist()[0]
            month = pollution_chunk['month'].tolist()[0]
            meteorology_chunk.loc[:,'day'] = day
            meteorology_chunk.loc[:,'month'] = month
        all_data = pd.concat([all_data, meteorology_chunk, pollution_chunk], axis=0, ignore_index=True)
        start_meteorology += 5
        start_pollution += 8
        if start_meteorology % 100 == 0:
            print(f'processing {start_meteorology} / {total_meteorology_len}')
    all_data.to_csv(f"./data/grid_data_{i}.csv", index=False)
    all_data = pd.DataFrame()


