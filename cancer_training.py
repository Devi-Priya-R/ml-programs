import pandas as pd
import sklearn.neighbors as ng 
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import math
mydata=pd.read_csv("cancer.csv")
x=mydata[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]]
y=mydata[["diagnosis"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
can_model=ng.KNeighborsClassifier(n_neighbors=3)
can_model.fit(x_train,y_train)
print("Training completed")
joblib.dump(can_model,"can_model.pkl")
test_result=can_model.predict(x_test)
print("accuracy",accuracy_score(y_test, test_result)*100)
