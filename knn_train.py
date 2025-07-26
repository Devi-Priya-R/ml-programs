import pandas as pd 
import sklearn.neighbors as ng 
import matplotlib.pyplot as pt 
import joblib 

mydata=pd.read_csv("data.csv")

x=mydata[["height"]]
y=mydata[["weight"]]

knn_model = ng.KNeighborsRegressor(n_neighbors=3)
knn_model.fit(x,y) #training the model
print("Training completed")
joblib.dump(knn_model,"knn_model.pkl")
