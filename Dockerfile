FROM python:3.10-slim

WORKDIR /app

# Installer gcc nécessaire pour compiler scikit-surprise
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model.pkl .
COPY scores_matrix.pkl .
COPY dataset_etudiants.csv .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
