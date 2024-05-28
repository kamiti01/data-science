import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#generate random time-series data
np.random.seed(42)
dates = pd.date_range(start ='2022-01-01', periods = 100, freq = 'D')
values = np.random.randn(100).cumsum()
# create a dataframe from the generated data
data = pd.DataFrame({'date':dates, 'values':values})
# set the 'date' column as the index
data.set_index('date', inplace = True)
# plot the time series data
plt.plot(data.index,data['values'])


plt.xlabel('Time')
plt.ylabel('values')
plt.xticks(rotation = 45)
plt.title('TimeSeries data')
plt.show()

#testing for stationarity
from statsmodels.tsa.stattools import adfuller
#assuming 'data' is the time series data
result = adfuller(data)
print('ADF statistic:',result[0])
print('p - value:', result[1])






                     
