import pandas as pd

#df is your dataframe
#example function is applicable for all INT dataframe 
spend_score = pd.read_csv("Mall_customers.csv")
print(spend_score)
spend_score = pd.read_csv('Mall_customers.csv', usecols = [ 'Age', 'Annual_Income', 'Spending_score'])
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(data = spend_score, x = 'Age', y = 'Annual_Income', hue = 'Spending_score')
plt.show()
