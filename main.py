from fastapi import FastAPI
import joblib
model = joblib.load("My_Iris")

app = FastAPI()

@app.get("/")
def predict_iris(sl:float,sw:float,pl:float,pw:float):
    result = model.predict([[sl,sw,pl,pw]])
    return {"Predicted Species is:":int(result[0])}
