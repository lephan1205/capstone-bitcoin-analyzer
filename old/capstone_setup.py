import numpy as np
import pandas as pd
from datetime import datetime
from fastparquet import write

from finml import compute_vwap
from finml import ohlc


## --- Read csv files and reformat ---
data = pd.read_csv('data/20190722.csv')
data = data[data.symbol == 'XBTUSD']
paths = ['data/20190723.csv', 'data/20190724.csv','data/20190725.csv', 'data/20190726.csv']
for path in paths:
    df = pd.read_csv(path)
    df = df[df.symbol == 'XBTUSD']
    data = data.append(df)
data['timestamp'] = data.timestamp.map(lambda t: datetime.strptime(t[:-3], "%Y-%m-%dD%H:%M:%S.%f")) # timestamp parsing
data.set_index('timestamp', inplace=True)
data.sort_index(inplace=True)

# keep only price and volume from original data
cols =['price', 'size']
data = data[cols].rename(columns={'size': 'volume'})


## --- Implement dollar-sampling method ---

# add cumulative dollar column
data_cm_dollar = data.assign(cmDollar=data['volume'].cumsum())

# compute total_dollars
total_dollars = data_cm_dollar.cmDollar.values[-1]
# user input: specify dollars per bar parameter
dollars_per_bar = 2e6
print('Total dollars:', total_dollars)
print('Dollars per bar:', dollars_per_bar)

# group trade by cmDollar//dollars_per_bar as groupId
data_dollar_grp = data_cm_dollar.assign(grpId=lambda row: row.cmDollar // dollars_per_bar)

# for each groupId, compute vwap and OHLC
data_dollar_ohlc =  data_dollar_grp.groupby('grpId').apply(lambda x: ohlc(compute_vwap(x)))

# drop index level
data_dollar_ohlc.index = data_dollar_ohlc.index.droplevel()

# drop rows with duplicated index but keep the first occurence
data_dollar_ohlc = data_dollar_ohlc[~data_dollar_ohlc.index.duplicated(keep='first')]
data_dollar_ohlc.head()

# save to file
write('data_dollarbars.pq', data_dollar_ohlc)

