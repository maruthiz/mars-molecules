import requests
import json

# Base URL for the API
base_url = "http://localhost:5000"

# Test the root endpoint
print("Testing root endpoint...")
response = requests.get(f"{base_url}/")
if response.status_code == 200:
    print("Root endpoint is working!")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Error: {response.status_code}")
    print(response.text)

# Test the properties endpoint
print("\nTesting properties endpoint...")
response = requests.get(f"{base_url}/properties")
if response.status_code == 200:
    print("Properties endpoint is working!")
    properties = response.json().get("properties", [])
    print(f"Available properties: {properties}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)

# Test prediction with a simple molecule
print("\nTesting prediction endpoint...")
test_data = {
    "smiles": "CCO",  # Ethanol
    "properties": ["logSolubility"]
}
response = requests.post(f"{base_url}/predict", json=test_data)
if response.status_code == 200:
    print("Prediction endpoint is working!")
    print(f"Prediction for Ethanol (CCO):")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Error: {response.status_code}")
    print(response.text)

# Test batch prediction
print("\nTesting batch prediction endpoint...")
test_data = {
    "smiles_list": ["CCO", "CCCC", "c1ccccc1"],  # Ethanol, Butane, Benzene
    "properties": ["logSolubility"]
}
response = requests.post(f"{base_url}/predict_batch", json=test_data)
if response.status_code == 200:
    print("Batch prediction endpoint is working!")
    results = response.json()
    for i, (smiles, result) in enumerate(zip(test_data["smiles_list"], results)):
        print(f"Molecule {i+1} ({smiles}):")
        print(json.dumps(result, indent=2))
else:
    print(f"Error: {response.status_code}")
    print(response.text)