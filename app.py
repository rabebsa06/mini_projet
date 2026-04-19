from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI(title="API Recommandation Étudiants")

with open("model.pkl", "rb") as f:
    algo = pickle.load(f)

with open("scores_matrix.pkl", "rb") as f:
    scores = pickle.load(f)

n = scores.shape[0]

@app.get("/")
def root():
    return {"message": "API de recommandation d'étudiants - en ligne ✅"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommend/custom/{student_id}")
def recommend_custom(student_id: int, k: int = 5):
    if student_id < 1 or student_id > n:
        return {"error": f"student_id doit être entre 1 et {n}"}
    idx = student_id - 1
    similarities = scores[idx]
    top_k = np.argsort(similarities)[::-1][:k]
    results = [{"etudiant": int(i + 1), "score": round(float(similarities[i]), 4)} for i in top_k]
    return {"student_id": student_id, "method": "KNN_Custom", "recommendations": results}

@app.get("/recommend/surprise/{student_id}")
def recommend_surprise(student_id: int, k: int = 5):
    if student_id < 1 or student_id > n:
        return {"error": f"student_id doit être entre 1 et {n}"}
    student = f"Etudiant_{student_id}"
    predictions = []
    for j in range(1, n + 1):
        target = f"Etudiant_{j}"
        if target != student:
            pred = algo.predict(student, target)
            predictions.append({"etudiant": j, "score": round(pred.est, 4)})
    predictions.sort(key=lambda x: x["score"], reverse=True)
    return {"student_id": student_id, "method": "KNN_Surprise", "recommendations": predictions[:k]}
