import pandas as pd
import numpy as np
import os
import pickle
import warnings
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# More aggressive warning suppression
warnings.filterwarnings("ignore")
import sys
if not sys.warnoptions:
    import os
    os.environ["PYTHONWARNINGS"] = "ignore"

# Import RDKit after warning suppression
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, MolSurf
from rdkit import DataStructs

# Disable RDKit logging
RDLogger.DisableLog('rdApp.*')

class MolecularDataPreparation:
    """
    Class for handling molecular data preparation and feature engineering.
    """
    def __init__(self, dataset_path=None, dataset_type="QM9"):
        """
        Initialize the data preparation class.
        
        Args:
            dataset_path (str): Path to the dataset file (CSV).
            dataset_type (str): Type of dataset ('QM9' or 'ESOL').
        """
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.data = None
        self.mols = []
        self.features = []
        self.properties = []
        self.property_names = []
        
    def load_dataset(self):
        """
        Load the molecular dataset based on the specified type.
        """
        if self.dataset_path is not None:
            print(f"Loading {self.dataset_type} dataset from {self.dataset_path}")
            self.data = pd.read_csv(self.dataset_path)
            
            # Set property names based on dataset type
            if self.dataset_type == "QM9":
                self.property_names = ['gap', 'mu', 'alpha', 'homo', 'lumo', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298']
            elif self.dataset_type == "ESOL":
                # Check if 'logSolubility' exists, otherwise look for alternative column names
                if 'logSolubility' in self.data.columns:
                    self.property_names = ['logSolubility']
                elif 'measured log solubility in mols per litre' in self.data.columns:
                    self.property_names = ['measured log solubility in mols per litre']
                    # You might want to rename this column for easier handling
                    self.data.rename(columns={'measured log solubility in mols per litre': 'logSolubility'}, inplace=True)
                    self.property_names = ['logSolubility']
        else:
            print(f"Creating sample {self.dataset_type} dataset")
            # Create sample data for testing
            if self.dataset_type == "QM9":
                # Sample SMILES for QM9-like molecules
                smiles = [
                    'C', 'CC', 'CCC', 'CCCC', 'CCCCC', 
                    'c1ccccc1', 'CCO', 'CCOC', 'CCN', 'CC(=O)O'
                ]
                
                # Create dummy properties
                data = {'smiles': smiles}
                self.property_names = ['gap', 'mu', 'alpha', 'homo', 'lumo', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298']
                for prop in self.property_names:
                    data[prop] = np.random.normal(0, 1, len(smiles))
                    
                self.data = pd.DataFrame(data)
                
            elif self.dataset_type == "ESOL":
                # Sample SMILES for ESOL-like molecules
                smiles = [
                    'CCO', 'CCOC', 'CCN', 'CC(=O)O', 'c1ccccc1',
                    'c1ccccc1O', 'c1ccccc1N', 'CC(=O)N', 'CCCl', 'CCBr'
                ]
                
                # Create dummy solubility values
                self.data = pd.DataFrame({
                    'smiles': smiles,
                    'logSolubility': np.random.normal(-2, 1, len(smiles))
                })
                self.property_names = ['logSolubility']
        
        print(f"Dataset loaded with {len(self.data)} molecules and {len(self.property_names)} properties.")
        return self.data
    
    def preprocess_molecules(self):
        """
        Convert SMILES strings to RDKit molecule objects and filter invalid molecules.
        """
        print("Converting SMILES to RDKit molecules and filtering invalid structures...")
        valid_indices = []
        
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            try:
                smiles = row['smiles']
                mol = Chem.MolFromSmiles(smiles)
                
                if mol is not None:
                    # Add hydrogens if they're not explicitly represented
                    mol = Chem.AddHs(mol)
                    self.mols.append(mol)
                    valid_indices.append(idx)
                else:
                    print(f"Warning: Invalid SMILES at index {idx}: {smiles}")
            except Exception as e:
                print(f"Error processing molecule at index {idx}: {e}")
                
        # Filter the dataframe to keep only valid molecules
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        print(f"Preprocessing complete. {len(self.mols)} valid molecules retained.")
        
    def generate_fingerprints(self, radius=2, nBits=2048):
        """
        Generate Morgan fingerprints (ECFP) for all molecules.
        
        Args:
            radius (int): Radius for the Morgan fingerprint.
            nBits (int): Number of bits in the fingerprint.
        """
        print(f"Generating Morgan fingerprints (radius={radius}, nBits={nBits})...")
        
        for mol in tqdm(self.mols):
            # Generate Morgan fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
            # Convert to numpy array
            array = np.zeros((nBits,))  # Fixed array size
            DataStructs.ConvertToNumpyArray(fp, array)
            self.features.append(array)
            
        # Convert list to numpy array
        self.features = np.array(self.features)
        print(f"Fingerprint generation complete. Feature shape: {self.features.shape}")
        return self.features
    
    def generate_descriptors(self):
        """
        Calculate molecular descriptors for all molecules.
        """
        print("Calculating molecular descriptors...")
        
        descriptor_functions = [
            Descriptors.MolWt,              # Molecular weight
            Descriptors.HeavyAtomCount,     # Number of heavy atoms
            Descriptors.NumHDonors,         # Number of H-bond donors
            Descriptors.NumHAcceptors,      # Number of H-bond acceptors
            Lipinski.NumRotatableBonds,     # Number of rotatable bonds
            MolSurf.TPSA,                   # Topological polar surface area
            Descriptors.NumAromaticRings    # Number of aromatic rings
        ]
        
        descriptor_names = [
            "MolWt", "HeavyAtomCount", "NumHDonors", "NumHAcceptors", 
            "NumRotatableBonds", "TPSA", "NumAromaticRings"
        ]
        
        descriptors = []
        for mol in tqdm(self.mols):
            mol_descriptors = []
            for func in descriptor_functions:
                try:
                    mol_descriptors.append(func(mol))
                except:
                    mol_descriptors.append(0)
            descriptors.append(mol_descriptors)
            
        # Convert to DataFrame for easier handling
        descriptors_df = pd.DataFrame(descriptors, columns=descriptor_names)
        print(f"Descriptor calculation complete. Generated {len(descriptor_names)} descriptors.")
        return descriptors_df
    
    def combine_features(self, fingerprints, descriptors_df):
        """
        Combine fingerprints and descriptors into a single feature matrix.
        
        Args:
            fingerprints (numpy.ndarray): Morgan fingerprints.
            descriptors_df (pd.DataFrame): Molecular descriptors.
        """
        # Convert descriptors to numpy array
        descriptors = descriptors_df.values
        
        # Combine by concatenating along axis 1
        combined_features = np.hstack([fingerprints, descriptors])
        print(f"Combined features shape: {combined_features.shape}")
        return combined_features
    
    def extract_properties(self):
        """
        Extract target properties from the dataset.
        """
        print("Extracting target properties...")
        
        self.properties = self.data[self.property_names].values
        print(f"Properties extracted. Shape: {self.properties.shape}")
        return self.properties
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data to use for testing.
            random_state (int): Random seed for reproducibility.
        """
        X = self.features
        y = self.properties
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets.")
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, output_dir):
        """
        Save processed data to disk.
        
        Args:
            output_dir (str): Directory to save the processed data.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save feature matrix
        np.save(os.path.join(output_dir, "features.npy"), self.features)
        
        # Save property values
        np.save(os.path.join(output_dir, "properties.npy"), self.properties)
        
        # Save property names
        with open(os.path.join(output_dir, "property_names.pkl"), "wb") as f:
            pickle.dump(self.property_names, f)
            
        print(f"Processed data saved to {output_dir}")
        
    def process_pipeline(self, radius=2, nBits=2048):
        """
        Run the complete data processing pipeline.
        
        Args:
            radius (int): Radius for Morgan fingerprints.
            nBits (int): Number of bits for fingerprints.
        """
        # 1. Load dataset
        self.load_dataset()
        
        # 2. Preprocess molecules
        self.preprocess_molecules()
        
        # 3. Generate fingerprints
        fingerprints = []
        print(f"Generating Morgan fingerprints (radius={radius}, nBits={nBits})...")
        
        for mol in tqdm(self.mols):
            # Generate Morgan fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
            # Convert to numpy array
            array = np.zeros((nBits,))
            DataStructs.ConvertToNumpyArray(fp, array)
            fingerprints.append(array)
            
        # Convert list to numpy array
        fingerprints = np.array(fingerprints)
        print(f"Fingerprint generation complete. Feature shape: {fingerprints.shape}")
        
        # 4. Generate descriptors
        descriptors_df = self.generate_descriptors()
        
        # 5. Combine features
        self.features = self.combine_features(fingerprints, descriptors_df)
        
        # 6. Extract properties
        self.extract_properties()
        
        # 7. Return processed data
        return self.features, self.properties, self.property_names

# Example usage:
if __name__ == "__main__":
    # Use the downloaded QM9 dataset
    data_prep = MolecularDataPreparation(
        dataset_path="data/qm9.csv",  # Path to the downloaded dataset
        dataset_type="QM9"
    )
    
    # Run the complete pipeline
    features, properties, property_names = data_prep.process_pipeline()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = data_prep.split_data()
    
    # Save processed data
    data_prep.save_processed_data("processed_data")
    
    print("Feature extraction and data preparation complete!")
    
    # Optionally, you can also process the ESOL dataset
    print("\n\nProcessing ESOL dataset...")
    esol_prep = MolecularDataPreparation(
        dataset_path="data/esol.csv",
        dataset_type="ESOL"
    )
    
    # Run the pipeline for ESOL
    esol_features, esol_properties, esol_property_names = esol_prep.process_pipeline()
    
    # Split ESOL data
    esol_X_train, esol_X_test, esol_y_train, esol_y_test = esol_prep.split_data()
    
    # Save processed ESOL data
    esol_prep.save_processed_data("processed_data_esol")
    
    print("ESOL data processing complete!")


def select_important_features(self, X_train, y_train, n_features=10):
    """
    Select the most important features using a Random Forest model.
    
    Args:
        X_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training target values.
        n_features (int): Number of top features to select.
        
    Returns:
        numpy.ndarray: Indices of the most important features.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel
    
    print(f"Selecting {n_features} most important features...")
    
    # Train a Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train.ravel())
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Sort feature indices by importance
    indices = np.argsort(importances)[::-1]
    
    # Print top features
    print("Top features ranked by importance:")
    for i in range(min(n_features, X_train.shape[1])):
        print(f"{i+1}. Feature {indices[i]}: {importances[indices[i]]:.4f}")
    
    # Select top features
    selector = SelectFromModel(rf, threshold=-np.inf, max_features=n_features)
    selector.fit(X_train, y_train.ravel())
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    
    print(f"Feature selection complete. Selected {len(selected_indices)} features.")
    return selected_indices
    
    # Feature selection for ESOL dataset
    print("\nPerforming feature selection for ESOL dataset...")
    selected_indices = esol_prep.select_important_features(esol_X_train, esol_y_train, n_features=20)
    
    # Use selected features
    esol_X_train_selected = esol_X_train[:, selected_indices]
    esol_X_test_selected = esol_X_test[:, selected_indices]
    
    print(f"Reduced feature dimensions: {esol_X_train_selected.shape}")
    
    # Save selected features
    np.save(os.path.join("processed_data_esol", "selected_features_indices.npy"), selected_indices)
    np.save(os.path.join("processed_data_esol", "X_train_selected.npy"), esol_X_train_selected)
    np.save(os.path.join("processed_data_esol", "X_test_selected.npy"), esol_X_test_selected)
    
    print("Feature selection complete!")
