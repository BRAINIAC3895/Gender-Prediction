from fastapi import FastAPI
from pydantic import BaseModel
import gender_guesser.detector as gender

app = FastAPI(title="Gender Prediction API")

detector = gender.Detector()

class NameInput(BaseModel):
    name: str

@app.post("/predict")
def predict_gender(input: NameInput):
    gender_prediction = detector.get_gender(input.name.split()[0])
    
    mapping = {
        "male": ("Male", 0.95),
        "mostly_male": ("Male", 0.75),
        "female": ("Female", 0.95),
        "mostly_female": ("Female", 0.75),
        "andy": ("Unisex", 0.50),
        "unknown": ("Unknown", 0.0),
    }
    
    label, confidence = mapping.get(gender_prediction, ("Unknown", 0.0))
    return {
        "name": input.name,
        "gender": label,
        "confidence": confidence
    }
