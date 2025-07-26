import joblib
knn_model = joblib.load("heart_model.pkl")
result=knn_model.predict([[53,1,0,140,203,1,0,155,1,3.1,0,0,3]])
if result[0]==1:
    print("patient has heart disease")
else:
    print("patient does not have heart disease")