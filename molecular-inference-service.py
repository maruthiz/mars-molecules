import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors
from rdkit.Chem import PandasTools
from rdkit.DataStructs import ConvertToNumpyArray
import threading
import queue
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MolecularInferenceService")

class MolecularFeatureExtractor:
    """
    Extract features from molecules for prediction.
    """
    def __init__(self, fp_radius: int = 2, fp_bits: int = 2048):
        """
        Initialize the feature extractor.
        
        Args:
            fp_radius: Radius for Morgan fingerprints
            fp_bits: Number of bits for fingerprints
        """
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        logger.info(f"Initialized MolecularFeatureExtractor (radius={fp_radius}, bits={fp_bits})")
        
    def generate_features(self, mol: Chem.Mol) -> np.ndarray:
        """
        Generate features for a single molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Feature array
        """
        if mol is None:
            raise ValueError("Invalid molecule")
        
        # Generate Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.fp_radius, nBits=self.fp_bits)
        fp_array = np.zeros((self.fp_bits,))
        ConvertToNumpyArray(fp, fp_array)
        
        # Calculate descriptors
        descriptors = []
        descriptors.append(Descriptors.MolWt(mol))  # Molecular weight
        descriptors.append(Descriptors.HeavyAtomCount(mol))  # Number of heavy atoms
        descriptors.append(Descriptors.NumHDonors(mol))  # Number of H-bond donors
        descriptors.append(Descriptors.NumHAcceptors(mol))  # Number of H-bond acceptors
        descriptors.append(rdMolDescriptors.CalcNumRotatableBonds(mol))  # Number of rotatable bonds
        descriptors.append(rdMolDescriptors.CalcTPSA(mol))  # Topological polar surface area
        descriptors.append(rdMolDescriptors.CalcNumAromaticRings(mol))  # Number of aromatic rings
        
        # Combine fingerprint and descriptors
        features = np.hstack([fp_array, descriptors])
        
        return features
    
    def process_smiles(self, smiles: str) -> Tuple[Chem.Mol, np.ndarray]:
        """
        Process a SMILES string to generate features.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Tuple of (RDKit molecule, feature array)
        """
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate features
        features = self.generate_features(mol)
        
        return mol, features
    
    def process_smiles_batch(self, smiles_list: List[str]) -> Tuple[List[Chem.Mol], np.ndarray]:
        """
        Process a batch of SMILES strings to generate features.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Tuple of (list of RDKit molecules, feature matrix)
        """
        mols = []
        features_list = []
        
        for smiles in smiles_list:
            try:
                mol, features = self.process_smiles(smiles)
                mols.append(mol)
                features_list.append(features)
            except Exception as e:
                logger.error(f"Error processing SMILES '{smiles}': {e}")
                mols.append(None)
                features_list.append(None)
        
        # Filter out None values
        valid_indices = [i for i, x in enumerate(features_list) if x is not None]
        valid_mols = [mols[i] for i in valid_indices]
        valid_features = np.array([features_list[i] for i in valid_indices])
        
        return valid_mols, valid_features, valid_indices


class MolecularInferenceService:
    """
    Service for molecular property prediction inference.
    """
    def __init__(self, 
                model_dir: str = "models",
                cache_size: int = 1000,
                batch_size: int = 32,
                n_workers: int = 4):
        """
        Initialize the inference service.
        
        Args:
            model_dir: Directory containing trained models
            cache_size: Maximum number of molecules to cache
            batch_size: Batch size for processing
            n_workers: Number of worker threads
        """
        self.model_dir = model_dir
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.n_workers = n_workers
        
        # Initialize feature extractor
        self.feature_extractor = MolecularFeatureExtractor()
        
        # Load models
        self.models = {}
        self.scalers = {}
        self.property_info = {}
        self._load_models()
        
        # Initialize cache
        self.cache = {}
        
        # Initialize thread pool
        self.executor = ThreadPoolExecutor(max_workers=n_workers)
        
        logger.info(f"Initialized MolecularInferenceService with {len(self.models)} models")
        
    def _load_models(self):
        """Load all available models from the model directory."""
        # Check QM9 models
        qm9_model_dir = os.path.join(self.model_dir, "qm9")
        if os.path.exists(qm9_model_dir):
            for filename in os.listdir(qm9_model_dir):
                if filename.endswith("_model.pkl"):
                    property_name = filename.replace("_model.pkl", "")
                    model_path = os.path.join(qm9_model_dir, filename)
                    
                    try:
                        with open(model_path, "rb") as f:
                            model = pickle.load(f)
                        
                        self.models[property_name] = model
                        self.property_info[property_name] = {
                            "dataset": "QM9",
                            "units": self._get_property_units(property_name)
                        }
                        logger.info(f"Loaded QM9 model for {property_name}")
                    except Exception as e:
                        logger.error(f"Error loading model {model_path}: {e}")
        
        # Check ESOL models
        esol_model_dir = os.path.join(self.model_dir, "esol")
        if os.path.exists(esol_model_dir):
            for filename in os.listdir(esol_model_dir):
                if filename.endswith("_model.pkl"):
                    property_name = filename.replace("_model.pkl", "")
                    model_path = os.path.join(esol_model_dir, filename)
                    
                    try:
                        with open(model_path, "rb") as f:
                            model = pickle.load(f)
                        
                        self.models[property_name] = model
                        self.property_info[property_name] = {
                            "dataset": "ESOL",
                            "units": "log(mol/L)"
                        }
                        logger.info(f"Loaded ESOL model for {property_name}")
                    except Exception as e:
                        logger.error(f"Error loading model {model_path}: {e}")
    
    def _get_property_units(self, property_name):
        """Get units for QM9 properties."""
        units = {
            "mu": "Debye",
            "alpha": "Bohr^3",
            "homo": "eV",
            "lumo": "eV",
            "gap": "eV",
            "r2": "Bohr^2",
            "zpve": "eV",
            "u0": "eV",
            "u298": "eV",
            "h298": "eV",
            "g298": "eV"
        }
        return units.get(property_name, "")
    
    def predict(self, smiles: str, properties: List[str] = None) -> Dict:
        """
        Predict properties for a single molecule.
        
        Args:
            smiles: SMILES string
            properties: List of properties to predict (None for all)
            
        Returns:
            Dictionary of predictions
        """
        # Check cache
        if smiles in self.cache:
            predictions = self.cache[smiles]
            # Filter by requested properties
            if properties:
                predictions = {k: v for k, v in predictions.items() if k in properties}
            return predictions
        
        # Process SMILES
        try:
            mol, features = self.feature_extractor.process_smiles(smiles)
        except Exception as e:
            logger.error(f"Error processing SMILES '{smiles}': {e}")
            return {"error": str(e)}
        
        # Determine which properties to predict
        if properties is None:
            properties = list(self.models.keys())
        else:
            # Filter to only include available models
            properties = [p for p in properties if p in self.models]
        
        # Make predictions
        predictions = {}
        for prop in properties:
            try:
                model = self.models[prop]
                
                # Reshape features for single sample
                X = features.reshape(1, -1)
                
                # Make prediction
                y_pred = model.predict(X)[0]
                
                # Add to predictions
                predictions[prop] = float(y_pred)
                
                # Add units and dataset info
                if prop in self.property_info:
                    predictions[f"{prop}_units"] = self.property_info[prop]["units"]
                    predictions[f"{prop}_dataset"] = self.property_info[prop]["dataset"]
                
            except Exception as e:
                logger.error(f"Error predicting {prop} for '{smiles}': {e}")
                predictions[f"{prop}_error"] = str(e)
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            self.cache.pop(next(iter(self.cache)))
        self.cache[smiles] = predictions
        
        return predictions
    
    def predict_batch(self, smiles_list: List[str], properties: List[str] = None) -> List[Dict]:
        """
        Predict properties for a batch of molecules.
        
        Args:
            smiles_list: List of SMILES strings
            properties: List of properties to predict (None for all)
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        # Process SMILES batch
        try:
            mols, features, valid_indices = self.feature_extractor.process_smiles_batch(smiles_list)
        except Exception as e:
            logger.error(f"Error processing SMILES batch: {e}")
            return [{"error": str(e)} for _ in smiles_list]
        
        # Determine which properties to predict
        if properties is None:
            properties = list(self.models.keys())
        else:
            # Filter to only include available models
            properties = [p for p in properties if p in self.models]
        
        # Initialize results with errors for invalid SMILES
        results = [{"error": "Invalid SMILES"} for _ in smiles_list]
        
        # Make predictions for valid molecules
        for i, (mol, feature, orig_idx) in enumerate(zip(mols, features, valid_indices)):
            predictions = {}
            
            for prop in properties:
                try:
                    model = self.models[prop]
                    
                    # Reshape features for single sample
                    X = feature.reshape(1, -1)
                    
                    # Make prediction
                    y_pred = model.predict(X)[0]
                    
                    # Add to predictions
                    predictions[prop] = float(y_pred)
                    
                    # Add units and dataset info
                    if prop in self.property_info:
                        predictions[f"{prop}_units"] = self.property_info[prop]["units"]
                        predictions[f"{prop}_dataset"] = self.property_info[prop]["dataset"]
                    
                except Exception as e:
                    logger.error(f"Error predicting {prop} for molecule {i}: {e}")
                    predictions[f"{prop}_error"] = str(e)
            
            # Update results
            results[orig_idx] = predictions
            
            # Update cache
            smiles = smiles_list[orig_idx]
            if len(self.cache) >= self.cache_size:
                # Remove oldest item
                self.cache.pop(next(iter(self.cache)))
            self.cache[smiles] = predictions
        
        return results


# Create Flask app
app = Flask(__name__)

# Initialize inference service
inference_service = MolecularInferenceService()

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for predicting properties of a single molecule."""
    data = request.json
    
    if not data or 'smiles' not in data:
        return jsonify({"error": "SMILES string is required"}), 400
    
    smiles = data['smiles']
    properties = data.get('properties')
    
    try:
        result = inference_service.predict(smiles, properties)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Endpoint for predicting properties of multiple molecules."""
    data = request.json
    
    if not data or 'smiles_list' not in data:
        return jsonify({"error": "List of SMILES strings is required"}), 400
    
    smiles_list = data['smiles_list']
    properties = data.get('properties')
    
    try:
        results = inference_service.predict_batch(smiles_list, properties)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/properties', methods=['GET'])
def get_properties():
    """Endpoint for getting available properties."""
    return jsonify({
        "properties": list(inference_service.models.keys()),
        "property_info": inference_service.property_info
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "models_loaded": len(inference_service.models)})

# Add this route after the other route definitions but before the if __name__ == '__main__' block

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API documentation."""
    return jsonify({
        "name": "Molecular Property Prediction API",
        "version": "1.0.0",
        "description": "API for predicting molecular properties using machine learning models",
        "endpoints": {
            "/predict": "POST - Predict properties for a single molecule",
            "/predict_batch": "POST - Predict properties for multiple molecules",
            "/properties": "GET - List available properties",
            "/health": "GET - Health check"
        },
        "example": {
            "curl": "curl -X POST -H \"Content-Type: application/json\" -d '{\"smiles\":\"CCO\", \"properties\":[\"logSolubility\"]}' http://localhost:5000/predict"
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)


