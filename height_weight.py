import pandas as pd
import matplotlib.pyplot as mt
import sklearn.linear_model as lm
mydata = pd.read_csv("new_data.csv")
x = mydata[["height"]]
y = mydata[["weight"]]
mt.scatter(x,y)
mt.show()
model = lm.LinearRegression()
model.fit(x,y)
print(model.predict([[160]]))