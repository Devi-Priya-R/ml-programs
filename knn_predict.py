import joblib
knn_model = joblib.load("knn_model.pkl")
input=int(input("Enter height for prediction: "))
print(knn_model.predict([[input]]))