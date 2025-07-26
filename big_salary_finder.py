import pandas as pd
import matplotlib.pyplot as pt
from sklearn.preprocessing import LabelEncoder
import sklearn.linear_model as lr 

mydata = pd.read_csv("big_salary_data.csv")
x = mydata[["education_qualification"]]
y = mydata[["salary"]]
print(mydata)
le = LabelEncoder()
mydata["education_qualification_updated"] = le.fit_transform(mydata[["education_qualification"]])
x_new = mydata[["education_qualification_updated"]]
print(mydata)
pt.scatter(x_new,y)
pt.show()
model = lr.LinearRegression()
model.fit(x_new,y)
print(model.predict([[0]]))