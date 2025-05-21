# 🧠 Digital Persona Classifier

This is a simple, fun AI-powered tool that predicts a user's **digital persona** based on their short bio and a few social media-style posts. Think of it as a personality quiz — but with ML or transformers under the hood!

## 🚀 Features

- 🔍 **Two prediction modes**:
  - `trained_model`: Uses a trained Logistic Regression model on sentence embeddings
  - `zero_shot`: Uses a transformer (BART) to classify without training
- 🧠 Predicts one of 4 personas:
  - **Tech Enthusiast**
  - **Foodie Explorer**
  - **Travel Adventurer**
  - **Fitness Guru**
- 🎯 Provides:
  - Persona label
  - Confidence score
  - Fun description of the persona
