import os
import pytest
import requests

# Set the base URL for the API
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5000")

def test_health_check():
    """
    Test the health check endpoint.
    Ensures the endpoint is accessible and returns a 200 status code.
    """
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    assert response.status_code == 200, f"Health check failed: {response.text}"
    assert response.json().get("status") == "healthy", "Health status is not 'healthy'"

@pytest.mark.parametrize("comments, expected_status", [
    (["This is a great product!"], 200),  # Valid input, expect 200
    (["Not worth the money.", "It's okay."], 200),  # Multiple comments, expect 200
    (None, 400),  # Invalid case, no comments provided, expect 400
    ([], 400),  # Empty list of comments, expect 400
])
def test_predict_endpoint(comments, expected_status):
    """
    Test the predict endpoint with various inputs.
    Checks the response status and the presence of expected fields.
    """
    data = {"comments": comments}
    response = requests.post(f"{BASE_URL}/predict", json=data, timeout=10)
    assert response.status_code == expected_status, f"Response: {response.text}"

    if response.status_code == 200:
        # Check that the response contains sentiment data
        json_response = response.json()
        assert isinstance(json_response, list), "Response should be a list of predictions"
        assert all("sentiment" in item for item in json_response), "Missing 'sentiment' in response items"

def test_generate_wordcloud_endpoint():
    """
    Test the word cloud generation endpoint.
    Ensures the endpoint returns an image for valid input.
    """
    data = {"comments": ["Amazing!", "Terrible!", "Good", "Bad"]}
    response = requests.post(f"{BASE_URL}/generate_wordcloud", json=data, timeout=10)
    assert response.status_code == 200, f"Response: {response.text}"
    assert response.headers.get("Content-Type") == "image/png", "Incorrect content type for word cloud"
    # Optional: Save the image to verify manually if needed
    # with open("wordcloud.png", "wb") as f:
    #     f.write(response.content)

def test_generate_wordcloud_with_empty_comments():
    """
    Test the word cloud generation endpoint with empty comments.
    Ensures the endpoint handles invalid input correctly and returns a 400 status.
    """
    data = {"comments": []}
    response = requests.post(f"{BASE_URL}/generate_wordcloud", json=data, timeout=10)
    assert response.status_code == 400, f"Expected 400 status for empty comments, got {response.status_code}"
    error_message = response.json().get("error")
    assert error_message, "Missing error message in response"
    assert "Comments cannot be empty" in error_message, f"Unexpected error message: {error_message}"
