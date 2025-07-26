import joblib
knn_model=joblib.load("iris_model.pkl")
print(knn_model.predict([[4.4,2.9,1.4,0.2]]))