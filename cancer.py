import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly as px
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

#loading the dataset from my computer
iris=pd.read_csv('breast.csv')
print('iris')

print('columns')
print(iris.columns)

#display the 1st 8 grouping
print(iris.head(8))

#missing values
print('missing values')
print(iris.isnull().sum())

print(iris.size)

# showing the data types
print(iris.dtypes)

#description of the data
print(iris.describe())
print(iris.shape)

#finding out the correlation between the features
corr = iris.corr()
corr.size

#plotting the heatmap of correlation between features
plt.figure(figsize=(20,20))
sns.heatmap(corr,cbar=True, square= True, fmt='.if', annot= True, annot_kws={'size':15}, cmap='Greens')
plt.show()

# Since the dataset has ID column, we can set it as the index
iris.set_index('id', inplace=True)

# Let's create a facet grid using Seaborn
# We can choose a subset of features for visualization
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']

# Create the facet grid
grid = sns.pairplot(iris, hue='diagnosis', vars=features)

# Adding title
plt.subplots_adjust(top=0.95)
grid.fig.suptitle('Pairplot of Breast Cancer Features')

# Show the plot
plt.show()

#diagnosis=iris['diagnosis'].value_counts().reset_index()

print(iris['diagnosis'].value_counts())

plt.figure(figsize=(6, 4))
sns.barplot(x='diagnosis', y='radius_mean', data= iris,palette='icefire')
plt.xlabel('diagnosis')
plt.ylabel('radius_mean')
plt.title('count')
plt.show()

diagnosis=iris['diagnosis'].value_counts().reset_index()
plt.figure(figsize=(6, 4))
sns.scatterplot(x='diagnosis', y=iris['diagnosis'].value_counts(), data= iris,palette='hsv')
plt.xlabel('diagnosis')
plt.ylabel('count')
plt.title('count')
plt.show()

#splitting target variable and independent variables
x=iris.drop(['diagnosis'], axis = 1)
y = iris['diagnosis']

#splitting the data into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, random_state=0)
print("Size of training set:", x_train.shape)
print("size of test set:", x_test.shape)

#Logistic regression
#import library for logisticregression
from sklearn.linear_model import LogisticRegression

#create a logistic regression classifier
logreg = LogisticRegression()

# train the model using the training sets
logreg.fit(x_train, y_train)

#prediction on test data
y_pred = logreg.predict(x_test)

#calculating the accuracy
acc_logreg = round(metrics.accuracy_score(y_test, y_pred)* 100, 2)
print ('Accuracy of Logistic Regression model : ', acc_logreg)

#import library of Gaussian Bayes model
from sklearn.naive_bayes import GaussianNB

# create a Gaussian Classifier
model = GaussianNB

# train the model using the training sets
model.fit(x_train, y_train)

#import decision tree classifier
from sklearn.tree import DecisionTreeClassifier

#create a Decision Tree Classifier
clf = DecisionTreeClassifier()







