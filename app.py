from flask import Flask, request, jsonify, render_template

from urllib.parse import quote as url_quote  # Use Python's built-in quote function instead
import os
import joblib
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.DataStructs import ConvertToNumpyArray

app = Flask(__name__)

# Load models from models directory
model_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(model_dir, exist_ok=True)

# Dictionary to store loaded models, scalers, and params
models = {}
scalers = {}
params = {}

# Property information with units and descriptions
property_info = {
    'logSolubility': {'units': 'log mol/L', 'description': 'Water Solubility'},
    'alpha': {'units': 'Bohr³', 'description': 'Polarizability'},
    'gap': {'units': 'eV', 'description': 'HOMO-LUMO Gap'},
    'homo': {'units': 'eV', 'description': 'HOMO Energy'},
    'lumo': {'units': 'eV', 'description': 'LUMO Energy'},
    'mu': {'units': 'Debye', 'description': 'Dipole Moment'},
    'r2': {'units': 'Bohr²', 'description': 'Electronic Spatial Extent'},
    'zpve': {'units': 'eV', 'description': 'Zero Point Vibrational Energy'},
    'u0': {'units': 'eV', 'description': 'Internal Energy at 0K'},
    'u298': {'units': 'eV', 'description': 'Internal Energy at 298K'},
    'h298': {'units': 'eV', 'description': 'Enthalpy at 298K'},
    'g298': {'units': 'eV', 'description': 'Free Energy at 298K'},
    # RDKit descriptors
    'logP': {'units': 'log units', 'description': 'Octanol-Water Partition Coefficient'},
    'TPSA': {'units': 'Å²', 'description': 'Topological Polar Surface Area'},
    'MolWt': {'units': 'g/mol', 'description': 'Molecular Weight'},
    'NumHAcceptors': {'units': 'count', 'description': 'Number of H-Bond Acceptors'},
    'NumHDonors': {'units': 'count', 'description': 'Number of H-Bond Donors'}
}

def load_models():
    print(f"Looking for models in directory: {model_dir}")
    if not os.path.exists(model_dir):
        print(f"Warning: Model directory {model_dir} does not exist!")
        return
        
    # Load ESOL models
    esol_dir = os.path.join(model_dir, 'esol')
    if os.path.exists(esol_dir):
        print(f"Found ESOL directory: {esol_dir}")
        load_model_set(esol_dir, 'logSolubility')
    else:
        print(f"Warning: ESOL directory {esol_dir} does not exist!")
    
    # Load QM9 models
    qm9_dir = os.path.join(model_dir, 'qm9')
    if os.path.exists(qm9_dir):
        print(f"Found QM9 directory: {qm9_dir}")
        qm9_properties = [
            'alpha', 'gap', 'homo', 'lumo', 'mu', 
            'r2', 'zpve', 'u0', 'u298', 'h298', 'g298'
        ]
        for prop in qm9_properties:
            load_model_set(qm9_dir, prop)
    else:
        print(f"Warning: QM9 directory {qm9_dir} does not exist!")
    
    if not models:
        print("Warning: No models were loaded successfully!")
    else:
        print(f"Successfully loaded {len(models)} models: {list(models.keys())}")

def load_model_set(directory, property_name):
    model_path = os.path.join(directory, f"{property_name}_model.pkl")
    scaler_path = os.path.join(directory, f"{property_name}_scaler.pkl")
    params_path = os.path.join(directory, f"{property_name}_params.json")
    
    print(f"Checking for model files for {property_name}:")
    print(f"  - Model: {model_path} (exists: {os.path.exists(model_path)})")
    print(f"  - Scaler: {scaler_path} (exists: {os.path.exists(scaler_path)})")
    print(f"  - Params: {params_path} (exists: {os.path.exists(params_path)})")
    
    if os.path.exists(model_path):
        try:
            models[property_name] = joblib.load(model_path)
            print(f"Successfully loaded model for {property_name}")
            
            if os.path.exists(scaler_path):
                scalers[property_name] = joblib.load(scaler_path)
                print(f"Successfully loaded scaler for {property_name}")
            else:
                print(f"Warning: No scaler found for {property_name}")
            
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params[property_name] = json.load(f)
                print(f"Successfully loaded parameters for {property_name}")
            else:
                print(f"Warning: No parameters found for {property_name}")
                
            return True
        except Exception as e:
            print(f"Error loading {property_name} model set: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"Error: Model file for {property_name} does not exist")
        return False

def calculate_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Calculate Morgan fingerprints with 2055 bits to match the trained models
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2055)
        fp_array = np.zeros((1, 2055))
        ConvertToNumpyArray(fp, fp_array[0])
        
        # Calculate basic RDKit descriptors
        descriptors = {
            'logP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'MolWt': Descriptors.MolWt(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumHDonors': Descriptors.NumHDonors(mol)
        }
        
        return {
            'mol': mol,
            'fingerprints': fp_array,
            'descriptors': descriptors
        }
    except Exception as e:
        print(f"Error calculating features: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/properties')
def get_properties():
    """Return available properties and their information."""
    available_properties = list(models.keys()) + list(property_info.keys())
    available_properties = list(set(available_properties))  # Remove duplicates
    
    return jsonify({
        'properties': available_properties,
        'property_info': property_info
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        smiles = data.get('smiles', '').strip()
        if not smiles:
            return jsonify({'error': 'No SMILES provided'}), 400

        print(f"Processing prediction request for SMILES: {smiles}")
        features = calculate_features(smiles)
        if features is None:
            return jsonify({'error': 'Invalid SMILES or calculation error'}), 400

        # Start with basic descriptors
        results = features['descriptors'].copy()
        results['smiles'] = smiles
        
        # Add units to descriptors
        # Create a list of keys to avoid 'dictionary changed size during iteration' error
        for prop in list(results.keys()):
            if prop in property_info and 'units' in property_info[prop]:
                results[f"{prop}_units"] = property_info[prop]['units']

        # Check if we have any models loaded
        if not models:
            print("Warning: No models are loaded, only returning RDKit descriptors")
            return jsonify({
                **results,
                'warning': 'No prediction models are loaded. Only returning RDKit descriptors.'
            })

        print(f"Making predictions with {len(models)} loaded models")
        # Use list() to create a copy of items to avoid potential dictionary modification issues
        for prop, model in list(models.items()):
            try:
                X = features['fingerprints']
                if prop in scalers:
                    X = scalers[prop].transform(X)
                prediction = model.predict(X)[0]
                results[prop] = float(prediction)
                
                # Add units and dataset info if available
                if prop in property_info:
                    if 'units' in property_info[prop]:
                        results[f"{prop}_units"] = property_info[prop]['units']
                    if 'description' in property_info[prop]:
                        results[f"{prop}_description"] = property_info[prop]['description']
                
                print(f"Predicted {prop}: {results[prop]}")
            except Exception as e:
                print(f"Error predicting {prop}: {str(e)}")
                import traceback
                traceback.print_exc()

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Load models when starting the server
load_models()

if __name__ == '__main__':
    print("Starting server on port 3000...")
    app.run(host='0.0.0.0', port=3000, debug=True)