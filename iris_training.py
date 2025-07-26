import pandas as pd
import sklearn.neighbors as ng 
import joblib
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split 
import math
mydata=pd.read_csv("Iris.csv")
x=mydata[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y=mydata[["Species"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
knn_model=ng.KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train,y_train)
print("Training completed")
joblib.dump(knn_model,"iris_model.pkl")
test_result=knn_model.predict(x_test)
accuracy=(accuracy_score(y_test,test_result)*100)
x=round(accuracy,2)
print(x)#print("accuracy",round(accuracy_score(y_test, test_result)*100,2))
print("confusion matrix",confusion_matrix(y_test, test_result))