import pandas as pd
import joblib
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.joblib")


model = joblib.load(MODEL_PATH)

def predict(input_data):
    
    
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    log_pred = model.predict(input_data)[0]
    real_pred = np.expm1(log_pred)
    return real_pred











