import requests
import sys

base = "http://localhost:8000"

def test_health():
    r = requests.get(f"{base}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("✅ Health OK")

def test_custom():
    r = requests.get(f"{base}/recommend/custom/1?k=5")
    assert r.status_code == 200
    data = r.json()
    assert len(data["recommendations"]) == 5
    print("✅ KNN Custom OK")

def test_surprise():
    r = requests.get(f"{base}/recommend/surprise/1?k=5")
    assert r.status_code == 200
    data = r.json()
    assert len(data["recommendations"]) == 5
    print("✅ KNN Surprise OK")

if __name__ == "__main__":
    test_health()
    test_custom()
    test_surprise()
    print("\n🎉 Tous les tests passent !")
