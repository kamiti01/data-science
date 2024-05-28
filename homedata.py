import pandas as pd
home_data= pd.read_csv('housing.csv')
print(home_data.head())
#visualize it
home_data = pd.read_csv('housing.csv', usecols = ['longitude', 'latitude', 'median_house_value'])
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(data = home_data, x = 'longitude', y = 'latitude', hue = 'median_house_value')
plt.show()
