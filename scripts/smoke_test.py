import argparse
import sys
import time
import requests
from io import BytesIO
from PIL import Image

# ANSI colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def print_result(test_name, passed, message=""):
    """Prints test result with color."""
    if passed:
        print(f"{test_name}: {GREEN}PASS{RESET} {message}")
    else:
        print(f"{test_name}: {RED}FAIL{RESET} {message}")


def generate_test_image():
    """Generates a small RGB test image in memory."""
    img = Image.new('RGB', (128, 128), color='blue')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr


def test_health(url):
    """Tests the /health endpoint."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                print_result("Test 1: Health Check", True)
                return True
            else:
                print_result("Test 1: Health Check", False, f"Expected status 'healthy', got '{data.get('status')}'")
                return False
        else:
            print_result("Test 1: Health Check", False, f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_result("Test 1: Health Check", False, f"Exception: {str(e)}")
        return False


def test_predict(url):
    """Tests the /predict endpoint."""
    try:
        img_bytes = generate_test_image()
        files = {'file': ('test_image.jpg', img_bytes, 'image/jpeg')}
        response = requests.post(f"{url}/predict", files=files, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            prediction = data.get("prediction")
            confidence = data.get("confidence")
            
            if prediction in ["cat", "dog"] and 0.0 <= confidence <= 1.0:
                print_result("Test 2: Predict", True, f"Prediction: {prediction}, Confidence: {confidence:.2f}")
                return True
            else:
                print_result("Test 2: Predict", False, f"Invalid prediction '{prediction}' or confidence '{confidence}'")
                return False
        else:
            print_result("Test 2: Predict", False, f"HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print_result("Test 2: Predict", False, f"Exception: {str(e)}")
        return False


def test_metrics(url):
    """Tests the /metrics endpoint."""
    try:
        response = requests.get(f"{url}/metrics", timeout=5)
        if response.status_code == 200:
            if "python_info" in response.text or "http_requests" in response.text:
                print_result("Test 3: Metrics", True)
                return True
            else:
                print_result("Test 3: Metrics", False, "Expected metrics not found in response")
                return False
        else:
            print_result("Test 3: Metrics", False, f"HTTP {response.status_code}")
            return False
    except Exception as e:
        print_result("Test 3: Metrics", False, f"Exception: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Smoke test for Cats vs Dogs API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    args = parser.parse_args()

    print(f"Running smoke tests against {args.url}...\n")
    
    passed_all = True
    passed_all &= test_health(args.url)
    passed_all &= test_predict(args.url)
    passed_all &= test_metrics(args.url)
    
    print("\nSmoke Test Summary:")
    if passed_all:
        print(f"{GREEN}ALL TESTS PASSED{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}SOME TESTS FAILED{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
