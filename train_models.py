import numpy as np
import os
import pickle
import sys

# Check if the molecular_model_training module exists
try:
    from molecular_model_training import MolecularPropertyPredictor
except ImportError:
    print("Error: Could not import MolecularPropertyPredictor.")
    print("Make sure molecular_model_training.py exists in the current directory.")
    sys.exit(1)

# Set paths to processed data
qm9_data_dir = "processed_data"
esol_data_dir = "processed_data_esol"

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Check if processed data exists
print(f"Checking for processed data in {qm9_data_dir} and {esol_data_dir}...")

# List files in the directories
if os.path.exists(qm9_data_dir):
    print(f"Files in {qm9_data_dir}:")
    for file in os.listdir(qm9_data_dir):
        print(f"  - {file}")
else:
    print(f"Directory {qm9_data_dir} does not exist!")

if os.path.exists(esol_data_dir):
    print(f"Files in {esol_data_dir}:")
    for file in os.listdir(esol_data_dir):
        print(f"  - {file}")
else:
    print(f"Directory {esol_data_dir} does not exist!")

# Process QM9 data if available
if os.path.exists(os.path.join(qm9_data_dir, "features.npy")) and \
   os.path.exists(os.path.join(qm9_data_dir, "properties.npy")) and \
   os.path.exists(os.path.join(qm9_data_dir, "property_names.pkl")):
    
    print("\nProcessing QM9 dataset...")
    
    # Load QM9 data
    features = np.load(os.path.join(qm9_data_dir, "features.npy"))
    properties = np.load(os.path.join(qm9_data_dir, "properties.npy"))
    
    # Load property names
    with open(os.path.join(qm9_data_dir, "property_names.pkl"), "rb") as f:
        qm9_property_names = pickle.load(f)
    
    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features, properties, test_size=0.2, random_state=42
    )
    
    print(f"QM9 data loaded. Training set shape: {X_train.shape}, {y_train.shape}")
    print(f"Properties: {qm9_property_names}")
    
    # Train QM9 models
    qm9_trainer = MolecularPropertyPredictor(
        model_dir="models/qm9",
        results_dir="results/qm9",
        verbose=True
    )
    
    # Train models for all QM9 properties
    qm9_trainer.train_all_properties(
        X_train=X_train,
        y_train=y_train,
        property_names=qm9_property_names,
        model_type="lightgbm",
        scaler_type="robust"
    )
    
    # Evaluate models
    qm9_results = qm9_trainer.evaluate_model(X_test, y_test)
    
    # Plot results for each property
    for prop in qm9_property_names:
        qm9_trainer.plot_prediction_vs_actual(prop)
        qm9_trainer.plot_feature_importance(prop)
    
    # Save models
    qm9_trainer.save_model()
else:
    print("Missing required QM9 files. Please run the data preparation script first.")

# Process ESOL data if available
if os.path.exists(os.path.join(esol_data_dir, "features.npy")) and \
   os.path.exists(os.path.join(esol_data_dir, "properties.npy")) and \
   os.path.exists(os.path.join(esol_data_dir, "property_names.pkl")):
    
    print("\nProcessing ESOL dataset...")
    
    # Load ESOL data
    features = np.load(os.path.join(esol_data_dir, "features.npy"))
    properties = np.load(os.path.join(esol_data_dir, "properties.npy"))
    
    # Load property names
    with open(os.path.join(esol_data_dir, "property_names.pkl"), "rb") as f:
        esol_property_names = pickle.load(f)
    
    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features, properties, test_size=0.2, random_state=42
    )
    
    print(f"ESOL data loaded. Training set shape: {X_train.shape}, {y_train.shape}")
    print(f"Properties: {esol_property_names}")
    
    # Train ESOL model
    esol_trainer = MolecularPropertyPredictor(
        model_dir="models/esol",
        results_dir="results/esol",
        verbose=True
    )
    
    # Train model for ESOL
    esol_trainer.train_all_properties(
        X_train=X_train,
        y_train=y_train,
        property_names=esol_property_names,
        model_type="lightgbm",
        scaler_type="robust"
    )
    
    # Evaluate model
    esol_results = esol_trainer.evaluate_model(X_test, y_test)
    
    # Plot results
    for prop in esol_property_names:
        esol_trainer.plot_prediction_vs_actual(prop)
        esol_trainer.plot_feature_importance(prop)
    
    # Save model
    esol_trainer.save_model()
else:
    print("Missing required ESOL files. Please run the data preparation script first.")

print("Script completed.")