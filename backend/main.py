import os
import logging
import pandas as pd
import joblib
from typing import List, Literal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "model.joblib"
ENCODER_PATH = "label_encoder.joblib"
CSV_PATH = "personas_examples.csv"

class Predictor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        logger.info("Initializing predictor...")
        self.embedder = SentenceTransformer(model_name)

        self.persona_labels = [
            "Tech Enthusiast",
            "Foodie Explorer",
            "Fitness Guru",
            "Travel Adventurer"
        ]

        self.persona_descriptions = {
            "Tech Enthusiast": "You're a tech wizard, always coding, tinkering with gadgets, or diving into the latest AI breakthroughs!",
            "Foodie Explorer": "Your taste buds lead the way, exploring new recipes and savoring every culinary adventure!",
            "Fitness Guru": "You live for the gym, crushing workouts and inspiring others with your health journey!",
            "Travel Adventurer": "The world is your playground, chasing sunsets and epic adventures in every corner of the globe!"
        }

        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
            logger.info("Model or encoder not found. Training a new one.")
            self._train_model(CSV_PATH)
        else:
            logger.info("Loading existing model and encoder.")
            self.model = joblib.load(MODEL_PATH)
            self.encoder = joblib.load(ENCODER_PATH)

        logger.info("Initializing zero-shot classifier...")
        self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def _train_model(self, csv_path: str):
        logger.info(f"Loading training data from {csv_path}...")
        df = pd.read_csv(csv_path, sep=';', encoding='latin1', names=['bio', 'posts', 'persona'], header=0)
        df["text"] = df["bio"] + " " + df["posts"].fillna("").str.replace(";", ". ")

        X = self.embedder.encode(df["text"].tolist(), convert_to_tensor=False)
        le = LabelEncoder()
        y = le.fit_transform(df["persona"])

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)

        joblib.dump(clf, MODEL_PATH)
        joblib.dump(le, ENCODER_PATH)
        self.model = clf
        self.encoder = le
        logger.info("Model trained and saved.")

    def predict(self, bio: str, posts: List[str], mode: Literal["trained_model", "zero_shot"] = "trained_model") -> tuple[str, float, str]:
        text = bio + " " + ". ".join(posts)

        if mode == "trained_model":
            logger.info("Using trained model for prediction.")
            embedding = self.embedder.encode([text], convert_to_tensor=False)
            probs = self.model.predict_proba(embedding)[0]
            pred_index = probs.argmax()
            label = self.encoder.inverse_transform([pred_index])[0]
            confidence = float(f"{probs[pred_index] * 100:.2f}")
            description = self.persona_descriptions.get(label, "No description available.")

        elif mode == "zero_shot":
            logger.info("Using zero-shot classification.")
            result = self.zero_shot_classifier(text, candidate_labels=self.persona_labels)
            label = result["labels"][0]
            confidence = float(f"{result['scores'][0] * 100:.2f}")
            description = self.persona_descriptions.get(label, "No description available.")
        else:
            raise ValueError(f"Unsupported prediction mode: {mode}")

        return label, confidence, description