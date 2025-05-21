from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main import Predictor, logger
from pydantic import BaseModel
from typing import List, Literal

app = FastAPI()
predictor = Predictor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PersonaRequest(BaseModel):
    bio: str
    posts: List[str]
    mode: Literal["trained_model", "zero_shot"] = "trained_model"

class PersonaResponse(BaseModel):
    persona: str
    confidence: float
    description: str

@app.post("/predict", response_model=PersonaResponse)
def predict_persona(request: PersonaRequest):
    logger.info("Received prediction request")
    label, confidence, description = predictor.predict(request.bio, request.posts, mode=request.mode)
    logger.info(f"Returning result: {label} ({confidence:.2f}) - {description}")
    return PersonaResponse(persona=label, confidence=confidence, description=description)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)