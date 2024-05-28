import pandas as pdk
import seaborn as sns
diamonds = sns.load_dataset('diamonds')
print(diamonds.to_string())
print(diamonds.info())
print (diamonds.size)
print(diamonds.shape)
print(diamonds.columns)
print(diamonds.info())
print('This is the what we have:')
print(diamonds.info())
print(diamonds.describe())
print(diamonds.isnull())
print(diamonds.isnull().sum())
import matplotlib as plt

