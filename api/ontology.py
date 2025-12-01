# api/ontology.py
import os
import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict

app = FastAPI()

# Загружаем модель один раз при старте (холодный старт ~4–6 сек, потом мгновенно)
model = SentenceTransformer('intfloat/multilingual-e5-large')

# Загружаем твой словарь идентичностей
with open("vocab_id.json", encoding="utf-8") as f:
    VOCAB = json.load(f)

# Предвычисляем прототипы один раз при запуске
prototypes = {}
prototype_vectors = {}

for item in VOCAB:
    name = item["name_ru"]
    phrases = item["phrases_ru"] + item["phrases_en"]
    if phrases:
        vectors = model.encode(phrases, normalize_embeddings=True, show_progress_bar=False)
        prototype_vectors[name] = torch.mean(vectors, dim=0)
        prototypes[name] = item

class Request(BaseModel):
    username: str

def cosine_sim(a, b):
    return float(torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8))

@app.post("/api/ontology")
async def get_ontology(req: Request):
    username = req.username.lstrip("@")
    
    # 1. Получаем данные из Apify (твой токен храним в Vercel Environment Variables)
    APIFY_TOKEN = os.environ.get("APIFY_TOKEN")
    if not APIFY_TOKEN:
        raise HTTPException(500, "APIFY_TOKEN не задан")

    resp = requests.post(
        "https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items",
        params={"token": APIFY_TOKEN},
        json={"usernames": [username], "resultsLimit": 60},
        timeout=90
    )
    
    if resp.status_code != 200 or not resp.json():
        raise HTTPException(404, "Профиль не найден или приватный")

    data = resp.json()[0]
    texts = []
    if data.get("biography"):
        texts.append(data["biography"])
    for post in data.get("latestPosts", [])[:50]:
        if caption := post.get("caption"):
            texts.append(caption)

    if len(texts) < 3:
        raise HTTPException(400, "Слишком мало текстов")

    # 2. Векторизуем все посты пользователя
    post_vectors = model.encode(texts, normalize_embeddings=True, convert_to_tensor=True)

    # 3. Считаем близость к каждому прототипу
    scores = {name: 0.0 for name in prototypes}
    for vec in post_vectors:
        for name, proto_vec in prototype_vectors.items():
            scores[name] += cosine_sim(vec, proto_vec)
    
    total_posts = len(texts)
    results = []
    for name, score_sum in scores.items():
        percent = round(score_sum / total_posts * 100, 1)
        if percent > 1:  # отсекаем шум
            proto = prototypes[name]
            results.append({
                "name": name,
                "percent": percent,
                "valence": proto.get("valence", ""),
                "core_fear": proto.get("core_fear", ""),
                "core_desire": proto.get("core_desire", ""),
                "description": proto["description"]
            })

    # Сортируем по убыванию
    results.sort(key=lambda x: x["percent"], reverse=True)

    return {
        "username": username,
        "total_posts_analyzed": total_posts,
        "identities": results[:10]  # топ-10 самых сильных
    }
