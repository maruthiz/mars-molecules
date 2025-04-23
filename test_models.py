import os
import pickle
from molecular_inference_service import MolecularInferenceService

# Initialize the inference service
inference_service = MolecularInferenceService(model_dir="models")

# Test with a simple molecule
test_smiles = "CCO"  # Ethanol
print(f"Testing prediction for {test_smiles}")

# Predict properties
result = inference_service.predict(test_smiles)
print("Prediction results:")
for prop, value in result.items():
    if not prop.endswith("_units") and not prop.endswith("_dataset"):
        print(f"{prop}: {value}")

print("\nAll available properties:")
print(list(inference_service.models.keys()))