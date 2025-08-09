import joblib
import numpy as np
model=joblib.load("model.pkl")
print(model.predict(np.array([[25,1,3,170]])))