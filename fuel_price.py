import pandas as pd
import matplotlib.pyplot as mt 
import sklearn.linear_model as lm 
mydata = pd.read_csv("fuel_data.csv")
x = mydata[["month"]]
y = mydata[["price"]]
mt.scatter(x,y) 
mt.show()
model = lm.LinearRegression()
model.fit(x,y)
print(model.predict([[10]]))           