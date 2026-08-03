import argparse
import random
import time
import requests
import io
import os
import csv
import json
from PIL import Image

def generate_random_image():
    """Generates a random RGB image for load testing."""
    img = Image.new('RGB', (128, 128), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr


def load_labeled_samples(test_dir, max_samples):
    """Loads labeled test images from data/processed/test if available."""
    samples = []
    for label in ["cat", "dog"]:
        label_dir = os.path.join(test_dir, f"{label}s")
        if not os.path.isdir(label_dir):
            continue
        for name in os.listdir(label_dir):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(label_dir, name), label))

    random.shuffle(samples)
    return samples[:max_samples]

def simulate_requests(url, num_requests, test_dir, output_dir):
    """Sends requests and stores post-deployment performance artifacts."""
    print(f"Simulating {num_requests} requests to {url}/predict...")

    os.makedirs(output_dir, exist_ok=True)
    
    correct_predictions = 0
    total_latency = 0

    labels = ["cat", "dog"]
    request_logs = []
    labeled_samples = load_labeled_samples(test_dir, num_requests)
    use_labeled = len(labeled_samples) > 0

    if use_labeled:
        print(f"Using labeled samples from {test_dir} for post-deployment evaluation.")
    else:
        print("No labeled samples found. Falling back to random synthetic requests.")
    
    for i in range(num_requests):
        if use_labeled and i < len(labeled_samples):
            img_path, true_label = labeled_samples[i]
            with open(img_path, "rb") as fh:
                img_bytes = io.BytesIO(fh.read())
            filename = os.path.basename(img_path)
        else:
            img_bytes = generate_random_image()
            true_label = random.choice(labels)
            filename = f"test_image_{i}.jpg"

        files = {'file': (filename, img_bytes, 'image/jpeg')}
        
        start_time = time.time()
        try:
            response = requests.post(f"{url}/predict", files=files)
            latency = time.time() - start_time
            total_latency += latency
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get("prediction")

                if prediction == true_label:
                    correct_predictions += 1

                request_logs.append({
                    "index": i + 1,
                    "filename": filename,
                    "true_label": true_label,
                    "predicted_label": prediction,
                    "confidence": data.get("confidence"),
                    "latency_sec": round(latency, 4),
                    "status_code": response.status_code
                })

                print(f"Request {i+1}/{num_requests} | True: {true_label} | Predicted: {prediction} | Latency: {latency:.3f}s")
            else:
                request_logs.append({
                    "index": i + 1,
                    "filename": filename,
                    "true_label": true_label,
                    "predicted_label": None,
                    "confidence": None,
                    "latency_sec": round(latency, 4),
                    "status_code": response.status_code
                })
                print(f"Request {i+1}/{num_requests} | Failed with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            request_logs.append({
                "index": i + 1,
                "filename": filename,
                "true_label": true_label,
                "predicted_label": None,
                "confidence": None,
                "latency_sec": None,
                "status_code": "request_error",
                "error": str(e)
            })
            print(f"Request {i+1}/{num_requests} | Error: {e}")
            
        # Small delay between requests
        time.sleep(random.uniform(0.1, 0.5))

    print("\n--- Simulation Summary ---")
    print(f"Total Requests: {num_requests}")
    print(f"Average Latency: {total_latency / num_requests:.3f}s")
    accuracy = (correct_predictions / num_requests * 100)
    print(f"Observed Accuracy: {accuracy:.2f}%")

    csv_path = os.path.join(output_dir, "post_deploy_requests.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["index", "filename", "true_label", "predicted_label", "confidence", "latency_sec", "status_code", "error"]
        )
        writer.writeheader()
        for row in request_logs:
            writer.writerow(row)

    summary_path = os.path.join(output_dir, "post_deploy_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "total_requests": num_requests,
                "correct_predictions": correct_predictions,
                "observed_accuracy_percent": round(accuracy, 2),
                "average_latency_sec": round(total_latency / num_requests, 4),
                "used_labeled_samples": use_labeled,
                "test_data_dir": test_dir
            },
            fh,
            indent=2
        )

    print(f"Saved request log: {csv_path}")
    print(f"Saved summary: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate traffic for the Cats vs Dogs API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--num-requests", type=int, default=10, help="Number of requests to simulate")
    parser.add_argument("--test-dir", default="data/processed/test", help="Directory containing labeled test images")
    parser.add_argument("--output-dir", default="artifacts", help="Directory to store simulation outputs")
    
    args = parser.parse_args()
    simulate_requests(args.url, args.num_requests, args.test_dir, args.output_dir)
