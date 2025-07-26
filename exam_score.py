import pandas as pd 
import matplotlib.pyplot as mt 
import sklearn.linear_model as lm 
mydata = pd.read_csv("exam.csv")
x = mydata[["study_hour"]]
y = mydata[["score"]]
mt.scatter(x,y)
mt.show()
model = lm.LinearRegression()
model.fit(x,y)
print(model.predict([[7]]))