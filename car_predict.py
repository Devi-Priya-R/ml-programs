import joblib
import numpy as np
model=joblib.load("car_model.pkl")
print(model.predict(np.array([[25,1,3,170]])))