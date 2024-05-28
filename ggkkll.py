import numpy as np
from sklearn.linear_model import LinearRegression

x= np.array([5,15,25,35,45,55]).reshape((-1,1))
y= np.array([5,20,14,32,22,38])
print('This are valuesin x:', x)
print('This values in y : ' , y)

Ex4_1= LinearRegression()
print(Ex4_1)
Ex4_1.fit(x,y)
r_sq = Ex4_1.score(x,y)
print(r_sq)
#interpret the r_sq
Ex4_1.intercept_
Ex4_1.coef_
intercept = Ex4_1.intercept_
gradient = Ex4_1.coef_
print('The y- intercept is:', intercept)
print('The gradient is:', gradient)
y_pred = Ex4_1.predict(x)
print(y_pred)
x_new = np.arange(5).reshape((-1,1))
print(x_new)
y_new = Ex4_1.predict(x_new)
print(y_new)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
#get data
x = np.arange(10).reshape(-1,1)
y = np.array([0,0,0,0,1,1,1,1,1,1])
#view the data
print('for x, we have:' , x)
print('....and for y :' , y)
Ex4_2 = LogisticRegression(solver = 'liblinear', random_state = 0)
#to fit, or train it
Ex4_2.fit(x,y)
#evaluate the model
print('answer:', Ex4_2.predict_proba(x))
print(Ex4_2.predict_proba(x))
# the actual predictions
print('This are the predictions:')
print(Ex4_2.predict(x))
# accuracy
print(Ex4_2.score(x,y))
#confusion matrix, it provides the actual and predicted outputs
print(confusion_matrix(y, Ex4_2.predict(x)))
#visualize
cm = confusion_matrix(y, Ex4_2.predict(x))
fig,ax = plt.subplots(figsize = (8,8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks = (0,1), ticklabels = ('predicted 0s','predicted 1s'))
ax.yaxis.set(ticks = (0,1), ticklabels = ('Actual 0s','Actual 1s'))
ax.set_ylim(1.5,-0.5)
for i in range (2):
    for j in range (2):
        ax.text(j,i,cm[i,j], ha = 'center', va = 'center', color = 'white')
        plt.show()
#generate report
print(classification_report(y, Ex4_2.predict(x)))



        
