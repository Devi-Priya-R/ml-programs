import pandas as pd
import sklearn.neighbors as ng
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score,mean_squared_error,confusion_matrix,root_mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import math
mydata = pd.read_csv("dataset_weight_finder.csv")
GenderEncoder = LabelEncoder()
BodyTypeEncoder = LabelEncoder()
mydata["Gender_enc"]= GenderEncoder.fit_transform(mydata["Gender"])
mydata["BodyType_enc"]=BodyTypeEncoder.fit_transform(mydata["BodyType"])
x=mydata[["Age","Gender_enc","BodyType_enc","Height"]]
y=mydata[["Weight"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
knn_weight_model=ng.KNeighborsClassifier(n_neighbors=3)
knn_weight_model.fit(x_train,y_train)
print("Training completed")
joblib.dump(knn_weight_model,"weight_model.pkl")

test_result=knn_weight_model.predict(x_test)
print("Mean Squared Error:",mean_squared_error(y_test,test_result))
print("RMSE:",root_mean_squared_error(y_test,test_result))
print("R2 Score:",r2_score(y_test,test_result))
print("Confusion Matrix:",confusion_matrix(y_test,test_result))