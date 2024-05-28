import pandas as pd
home_data= pd.read_csv('housing.csv')
print(home_data.head())
#visualize it
home_data = pd.read_csv('housing.csv', usecols = ['longitude', 'latitude', 'median_house_value'])
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(data = home_data, x = 'longitude', y = 'latitude', hue = 'median_house_value')
plt.show()
#normalize the data
#save as ex5.py
data = pd.read_csv('housing.csv')
from sklearn.model_selection import train_test_split
#2.1
x_train, x_test, y_train, y_test = train_test_split(home_data[['latitude', 'longitude']], home_data[['median_house_value']], test_size =0.33, random_state = 0)
#2.2 normalize then
from sklearn import preprocessing
x_train_norm = preprocessing.normalize(x_train)
x_test_norm = preprocessing.normalize(x_test)
#fitting/training and evaluation
from sklearn.cluster import KMeans
exercise = KMeans(n_clusters = 3, random_state = 0, n_init = 'auto')
exercise.fit(x_train_norm)
#then visualize the results
sns.scatterplot(data = x_train, x = 'longitude', y = 'latitude', hue = exercise.labels_)
                
#plt.show()

sns.boxplot(x=exercise.labels_ , y = y_train['median_house_value'])
plt.show()
# 3.2 evaluate performance
from sklearn.metrics import silhouette_score
perf = silhouette_score(x_train_norm, exercise.labels_ , metric = 'euclidean')
#print(perf)
#3.3 how many clusters to use
K = range(2,8)
fit = []
score= []
for K in K :
#train the model for current value of K on training data
 model = KMeans(n_clusters = K, random_state = 0, n_init = 'auto').fit(x_train_norm)
#append the model to fits
 fit.append(model)
#append the silhouette_score
score.append(silhouette_score(x_train_norm, model.labels_ , metric = 'euclidean'))
#print(fit)
#print(score)

#then visualize afew, start with K = 2
sns.scatterplot(data = x_train, x = 'longitude', y = 'latitude', hue = fit[0].labels_)
#plt.show()

#halves, not good looking

# what about K = 

sns.scatterplot(data = x_train, x = 'longitude', y = 'latitude', hue = fit[2].labels_)
#plt.show()

# is it better or worse ?
#what about 7?
sns.scatterplot(data= x_train, x = 'longitude', y = 'latitude', hue = fit[5].labels_)
#plt.show()

#7 is too many
#use the elbow plot to compare
sns.lineplot(x = K, y = score)
#plt.show()

#choose the point where the performance start to flatten or get worse. Here K = 5
sns.scatterplot(data = x_train, x = 'longitude', y = 'latitude', hue = fit[3].labels_)
#plt.show()
sns.boxplot(x= fit[3].labels_ , y = y_train['median_house_value'])
#plt.show()

                
