from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["response"] == "welcome"

def test_post_lessthan_50k():
    params = {
                "age": 34,
                "workclass": "Private",
                "fnlgt": 245487,
                "education": "7th-8th",
                "education-num": 4,
                "marital-status": "Married-civ-spouse",
                "occupation": "Transport-moving",
                "relationship": "Husband",
                "race": "Amer-Indian-Eskimo",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 45,
                "native-country": "Mexico"
            }
    response = client.post("/predict", json=params)
    assert response.status_code == 200
    assert response.json()['prediction'] == "<=50k"

def test_post_great_50k():
    params = {
                "age": 48,
                "workclass": "Private",
                "fnlgt": 295487,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 45,
                "native-country": "United-States"
            }
    response = client.post("/predict", json=params)
    assert response.status_code == 200
    assert response.json()['prediction'] == ">50k"