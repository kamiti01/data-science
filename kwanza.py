import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
diamonds = sns.load_dataset('diamonds')
plt.figure(figsize=(6,10))
sns.histplot(diamonds['carat'], kde = True)
plt.show()

           
