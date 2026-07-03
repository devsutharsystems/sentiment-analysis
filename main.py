from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from text_utils import preprocess_text

app = FastAPI(title="Sentiment Analysis API")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "Sentiment Analysis API is running"}

@app.post("/predict")
def predict(input: TextInput):
    cleaned = preprocess_text(input.text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])

    return {
        "text": input.text,
        "sentiment": "positive" if prediction == 1 else "negative",
        "confidence": round(float(confidence), 3)
    }