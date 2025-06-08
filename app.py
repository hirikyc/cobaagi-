from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class RestaurantRecommender:
    def __init__(self, vectorizer, place_vectors, places_data):
        self.vectorizer = vectorizer
        self.place_vectors = place_vectors
        self.places_data = places_data
        
    def get_recommendations(self, food_name: str, top_n: int = 5) -> List[Dict[str, Any]]:
        food_vector = self.vectorizer.transform([food_name])
        similarity_scores = cosine_similarity(food_vector, self.place_vectors).flatten()
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        recommendations = []
        for idx in top_indices:
            place = self.places_data.iloc[idx]
            recommendations.append({
                'name': place['name'],
                'cuisine': place['cuisine'],
                'rating': float(place['rating']),
                'similarity_score': float(similarity_scores[idx])
            })
        return recommendations

app = FastAPI(title="FoodLens API", description="API untuk deteksi makanan dan rekomendasi restoran", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and data
model = tf.keras.models.load_model("app/models/food_detection.keras")
with open("app/data/food_labels.pkl", "rb") as f:
    class_names = pickle.load(f)
with open("app/data/food_origins.pkl", "rb") as f:
    food_origins = pickle.load(f)
with open("app/models/recommendation_system.pkl", "rb") as f:
    recommender_data = pickle.load(f)
recommender = RestaurantRecommender(
    vectorizer=recommender_data['vectorizer'],
    place_vectors=recommender_data['place_vectors'],
    places_data=recommender_data['places_data']
)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return img

@app.get("/")
def root():
    return {
        "message": "Selamat datang di FoodLens API",
        "endpoints": {
            "/detect": "POST - Unggah gambar untuk deteksi makanan",
            "/recommend/{food_name}": "GET - Dapatkan rekomendasi restoran untuk makanan tertentu"
        }
    }

@app.post("/detect")
async def detect_food(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File harus berupa gambar")
        contents = await file.read()
        img = preprocess_image(contents)
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = float(predictions[predicted_class])
        top_3_idx = np.argsort(predictions)[-3:][::-1]
        top_predictions = [
            {
                'food_name': class_names[idx],
                'confidence': float(predictions[idx]),
                'origin': food_origins.get(class_names[idx], "Tidak diketahui")
            }
            for idx in top_3_idx
        ]
        detected_food = class_names[predicted_class]
        recommendations = recommender.get_recommendations(detected_food)
        return {
            'detection': {
                'food_name': detected_food,
                'origin': food_origins.get(detected_food, "Tidak diketahui"),
                'confidence': confidence,
                'top_predictions': top_predictions
            },
            'recommendations': recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/recommend/{food_name}")
async def get_recommendations(food_name: str):
    try:
        recommendations = recommender.get_recommendations(food_name)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))