import joblib 
can_model=joblib.load("can_model.pkl")
result=can_model.predict([[15.71,13.93,102,761.7,0.09462,0.09462,0.07135,0.05933,0.1816,0.05723,0.3117,0.8155,1.972,27.94,0.005217,0.01515,0.01678,0.01268,0.01669,0.00233,17.5,19.25,114.3,922.8,0.1223,0.1949,0.1709,0.1374,0.2723,0.07071]])
if result[0]=="M":
    print("cancerous")
else:
    print("non-cancerous")