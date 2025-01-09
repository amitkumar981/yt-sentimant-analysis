import pytest
import requests
import json

BASE_URL = "http://localhost:5000"  # Replace with your deployed URL if needed

def test_predict_endpoint():
    """Test the /predict endpoint with sample comments."""
    data = {
        "comments": ["This is a great product!", "Not worth the money.", "It's okay."]
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)

    assert response.status_code == 200, "Expected status code 200"
    assert isinstance(response.json(), list), "Response should be a list"
    
    for item in response.json():
        assert "comment" in item, "Each item should contain 'comment'"
        assert "sentiment" in item, "Each item should contain 'sentiment'"

def test_predict_with_timestamps_endpoint():
    """Test the /predict_with_timestamps endpoint with comments and timestamps."""
    data = {
        "comments": [
            {"text": "This is fantastic!", "timestamp": "2024-10-25 10:00:00"},
            {"text": "Could be better.", "timestamp": "2024-10-26 14:00:00"}
        ]
    }
    response = requests.post(f"{BASE_URL}/predict_with_timestamps", json=data)

    assert response.status_code == 200, "Expected status code 200"
    assert isinstance(response.json(), list), "Response should be a list"

    for item in response.json():
        assert "comment" in item, "Each item should contain 'comment'"
        assert "sentiment" in item, "Each item should contain 'sentiment'"
        assert "timestamp" in item, "Each item should contain 'timestamp'"

def test_generate_chart_endpoint():
    """Test the /generate_chart endpoint with sentiment counts."""
    data = {
        "sentiment_counts": {"1": 5, "0": 3, "-1": 2}
    }
    response = requests.post(f"{BASE_URL}/generate_chart", json=data)

    assert response.status_code == 200, "Expected status code 200"
    assert response.headers["Content-Type"] == "image/png", "Response should be a PNG image"

    # Verify that the response content is not empty
    assert len(response.content) > 0, "Response content should not be empty"

def test_generate_wordcloud_endpoint():
    """Test the /generate_wordcloud endpoint with sample comments."""
    data = {
        "comments": ["Amazing!", "Could use some improvement.", "Loved it!"]
    }
    response = requests.post(f"{BASE_URL}/generate_wordcloud", json=data)

    assert response.status_code == 200, "Expected status code 200"
    assert response.headers["Content-Type"] == "image/png", "Response should be a PNG image"
    
    # Verify that the response content is not empty
    assert len(response.content) > 0, "Response content should not be empty"

def test_generate_trend_graph_endpoint():
    """Test the /generate_trend_graph endpoint with sentiment data."""
    data = {
        "sentiment_data": [
            {"timestamp": "2024-11-01T10:00:00", "sentiment": 1},
            {"timestamp": "2024-12-01T10:00:00", "sentiment": 0},
            {"timestamp": "2025-01-01T10:00:00", "sentiment": -1}
        ]
    }
    response = requests.post(f"{BASE_URL}/generate_trend_graph", json=data)

    assert response.status_code == 200, "Expected status code 200"
    assert response.headers["Content-Type"] == "image/png", "Response should be a PNG image"
    
    # Verify that the response content is not empty
    assert len(response.content) > 0, "Response content should not be empty"

