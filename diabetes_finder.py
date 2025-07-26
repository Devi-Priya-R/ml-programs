import pandas as pd 
import sklearn.neighbors as ng 
mydata=pd.read_csv("diabetes.csv")
print(mydata)
x=mydata[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
y=mydata[["Outcome"]]
model=ng.KNeighborsClassifier(n_neighbors=3)
model.fit(x,y)
result=(model.predict([[4,110,92,0,0,37.6,0.191,30]])) 
if result[0]==1:
    print("Diabetic patient")
else:
    print("Non-diabetic person")