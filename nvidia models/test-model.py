import requests
import base64
import json

# Define the URL and output path
invoke_url = "http://localhost:8000/v1/infer"
output_image_path = "result.jpg"

# Prepare the payload
payload = {
    "prompt": "A simple coffee shop interior",
    "mode": "base",
    "seed": 0,
    "steps": 50
}

# Send the POST request
response = requests.post(invoke_url, json=payload)

# Check if request was successful
if response.status_code == 200:
    try:
        # Parse JSON response
        data = response.json()
        base64_image = data['artifacts'][0]['base64']

        # Decode base64 and write to file
        with open(output_image_path, "wb") as f:
            f.write(base64.b64decode(base64_image))
        print(f"Image saved to {output_image_path}")
    except (KeyError, json.JSONDecodeError) as e:
        print("Failed to process response:", e)
else:
    print(f"Request failed with status code {response.status_code}")
    print("Response body:", response.text)
