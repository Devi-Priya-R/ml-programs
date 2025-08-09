import pandas as pd
import sklearn.neighbors as ng 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
mydata=pd.read_csv("car_data.csv")
BrandEncoder = LabelEncoder()
FuelTypeEncoder = LabelEncoder()
mydata["Brand_enc"]= BrandEncoder.fit_transform(mydata["Brand"])
mydata["FuelType_enc"]=FuelTypeEncoder.fit_transform(mydata["FuelType"])
x=mydata[["Year","Brand","FuelType","Mileage"]]
y=mydata[["Price"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=Sequential()
model.add(Dense(10,activation="relu",input_shape=(4,) ))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")
model.fit(x_train,y_train,epochs=10)
print("Training completed")
joblib.dump(model,"car_model.pkl")
test_result=model.predict(x_test)