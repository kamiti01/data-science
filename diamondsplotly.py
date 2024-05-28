x = [1,2,3,4]
y = [1,2,3,4]
#x = [1,2,3,4]
#y = [1,2,3,4]
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
diamonds = sns.load_dataset('diamonds')
#fig = px.line(diamonds.sample(20), x = 'price', y = 'carat')
#fig.show()
fig = px.histogram(diamonds.sample(50), x = 'price')
fig.show()
fig = px.histogram(diamonds, y = 'carat')
fig.show()
fig = px.histogram(diamonds.sample(20), x = 'cut')
fig.show()
fig = px.histogram(diamonds.sample(20), x = 'clarity')
fig.show()
#fig = px.histogram(diamonds.sample(20), x = 'cut')
#fig = px.histogram(diamonds.sample(20), x = 'clarity')
#fig = px.histogram(diamonds.sample(20), x = 'clarity')
fig = px.violin(diamonds, x = 'cut', y = 'price')
fig.show()
#fig = px.violin(diamonds, x = 'cut', y = 'price')
fig = px.scatter(diamonds, x = 'cut', y = 'price')
fig.show()

