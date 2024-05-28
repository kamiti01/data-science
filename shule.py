import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('shule.csv')
print(df.head(15))
print(df.isnull().sum())
sns.catplot(x = ['LANG', 'READ', 'KISW', 'KUSOMA', 'MATHS', 'ENV ACT', 'CRE ACT', 'C/A ACT'], kind = 'count')
plt.show()
