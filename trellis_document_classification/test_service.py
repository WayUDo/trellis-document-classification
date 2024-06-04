from fastapi.testclient import TestClient
from service import app  # Import the FastAPI app object from service.py

client = TestClient(app)

def test_predict_medical_success():
    response = client.post("/predict/", json={"text": "This is a medical article"})
    print(response.json())
    assert response.status_code == 200
    assert "class_label" in response.json()
    assert response.json()["class_label"] == "medical"

def test_predict_other_success():
    response = client.post("/predict/", json={"text": "This article is related to nothing"})
    print(response.json())
    assert response.status_code == 200
    assert "class_label" in response.json()
    assert response.json()["class_label"] == "other" 

def test_input_issue_error_none():
    response = client.post("/predict/", json={"text": None}) 
    print(response.json())
    assert response.status_code == 422
    assert "detail" in response.json()

def test_input_issue_error_int():
    response = client.post("/predict/", json={"text": 4}) 
    print(response.json())
    assert response.status_code == 422
    assert "detail" in response.json()

def test_input_issue_error_dict():
    response = client.post("/predict/", json={"text": {'this': 'is some text'}}) 
    print(response.json())
    assert response.status_code == 422
    assert "detail" in response.json()

def test_long_input():
    text = 'a'*70000
    response = client.post("/predict/", json={"text": text}) 
    print(response.json())
    assert response.status_code == 422
    assert "detail" in response.json()