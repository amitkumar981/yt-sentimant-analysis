import os
import pytest
import requests

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5000")

def test_health_check():
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    assert response.status_code == 200, f"Health check failed: {response.text}"

@pytest.mark.parametrize("comments, expected_status", [
    (["This is a great product!"], 200),
    (["Not worth the money.", "It's okay."], 200),
    (None, 400),  # Invalid case
    ([], 400),  # Empty list case
])
def test_predict_endpoint(comments, expected_status):
    data = {"comments": comments}
    response = requests.post(f"{BASE_URL}/predict", json=data, timeout=10)
    assert response.status_code == expected_status, f"Response: {response.text}"
    if response.status_code == 200:
        assert "sentiment" in response.json()[0], "Missing 'sentiment' in response"

def test_generate_wordcloud_endpoint():
    data = {"comments": ["Amazing!", "Terrible!", "Good", "Bad"]}
    response = requests.post(f"{BASE_URL}/generate_wordcloud", json=data, timeout=10)
    assert response.status_code == 200, f"Response: {response.text}"
    assert response.headers["Content-Type"] == "image/png", "Incorrect content type"

def test_generate_wordcloud_with_empty_comments():
    data = {"comments": []}
    response = requests.post(f"{BASE_URL}/generate_wordcloud", json=data, timeout=10)
    assert response.status_code == 400, f"Expected 400 status for empty comments, got {response.status_code}"
    assert "error" in response.json(), "Missing error message in response"


