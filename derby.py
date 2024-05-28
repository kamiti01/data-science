import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
#fetch dataset
iris = fetch_ucirepo(id=53)
#data (as pandas dataframes)
x = iris.data.features
y = iris.data.targets
  
# metadata 
print(iris.metadata) 
import pandas as pd  
# variable information 
print(iris.variables) 

df = pd.fetch_ucirepo('iris.metadata')
