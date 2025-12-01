from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем словарь
with open("vocab_id.json", "r", encoding="utf-8") as f:
    VOCAB = json.load(f)

model = SentenceTransformer('intfloat/multilingual-e5-large')

@app.post("/ontology")
async def ontology(request: dict):
    username = request.get("username", "").lstrip("@")
    if not username:
        return {"error": "нет username"}

    # Apify
    resp = requests.post(
        "https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items",
        params={"token": "apify_api_XFk4W4rmvDDfnSxhsYzkUbHnHPdJ0R1I2wyz"},
        json={"usernames": [username], "resultsLimit": 50},
        timeout=60
    )
    profile_data = resp.json()

    if not profile_data or not profile_data[0]:
        return {"error": "приватный профиль"}

    profile = profile_data[0]
    texts = []
    if profile.get("biography"):
        texts.append(profile["biography"])
    for post in profile.get("latestPosts", [])[:40]:
        if post.get("caption"):
            texts.append(post["caption"])

    if len(texts) < 3:
        return {"error": "мало текстов"}

    post_vectors = model.encode(texts)

    prototypes = []
    for item in VOCAB:
        phrases = item.get("phrases_ru", []) + item.get("phrases_en", [])
        if phrases:
            proto_vec = model.encode(phrases).mean(axis=0)
            prototypes.append({"name": item["name_ru"], "vector": proto_vec})

    results = []
    for vec in post_vectors:
        scores = []
        for proto in prototypes:
            sim = cosine_similarity([vec], [proto["vector"]])[0][0]
            if sim > 0.55:
                scores.append({"name": proto["name"], "score": float(sim)})
        if scores:
            best = max(scores, key=lambda x: x["score"])
            results.append(best["name"])

    if not results:
        return {"clusters": [{"name": "неопределённая идентичность", "weight": 100.0}]}

    count = Counter(results)
    total = len(results)
    clusters = []
    for name, cnt in count.most_common():
        clusters.append({"name": name, "weight": round(cnt / total * 100, 1)})

    return {"username": username, "clusters": clusters, "total_posts": len(texts)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
