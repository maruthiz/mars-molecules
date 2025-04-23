import numpy as np
import pandas as pd
import os
import pickle
import time
import json
import warnings
from datetime import datetime
from functools import partial
from typing import Dict, List, Tuple, Union, Optional

# ML Libraries
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from bayes_opt import BayesianOptimization
import joblib

# Parallel processing
from joblib import Parallel, delayed
import multiprocessing

# For hyperparameter visualization
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope

# For feature importance
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# Silence non-essential warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class MolecularPropertyPredictor:
    """
    High-performance molecular property prediction pipeline with advanced model training,
    hyperparameter optimization, and evaluation capabilities.
    """
    
    def __init__(self, 
                model_dir: str = "models",
                results_dir: str = "results",
                n_jobs: int = -1, 
                random_state: int = 42,
                verbose: bool = True):
        """
        Initialize the molecular property predictor.
        
        Args:
            model_dir: Directory to save trained models
            results_dir: Directory to save evaluation results
            n_jobs: Number of parallel jobs (-1 for all processors)
            random_state: Random seed for reproducibility
            verbose: Whether to print detailed information
        """
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self.random_state = random_state
        self.verbose = verbose
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize containers
        self.models = {}
        self.feature_importances = {}
        self.scalers = {}
        self.hyperparams = {}
        self.evaluation_results = {}
        self.best_params = {}
        self.property_names = []
        
        # Available models
        self.model_types = {
            'lightgbm': self._create_lgb_model,
            'xgboost': self._create_xgb_model,
            'random_forest': self._create_rf_model
        }
        
        if self.verbose:
            print(f"Initialized MolecularPropertyPredictor with {self.n_jobs} workers")
    
    def _log(self, message: str):
        """Log message if verbose is enabled"""
        if self.verbose:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
    
    def _create_lgb_model(self, params: Dict = None) -> lgb.LGBMRegressor:
        """Create a LightGBM model with given parameters"""
        default_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 7,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': -1
        }
        
        if params:
            default_params.update(params)
            
        return lgb.LGBMRegressor(**default_params)
    
    def _create_xgb_model(self, params: Dict = None) -> xgb.XGBRegressor:
        """Create an XGBoost model with given parameters"""
        default_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 7,
            'gamma': 0.1,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbosity': 0
        }
        
        if params:
            default_params.update(params)
            
        return xgb.XGBRegressor(**default_params)
    
    def _create_rf_model(self, params: Dict = None) -> RandomForestRegressor:
        """Create a Random Forest model with given parameters"""
        default_params = {
            'n_estimators': 500,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': 0
        }
        
        if params:
            default_params.update(params)
            
        return RandomForestRegressor(**default_params)
    
    def preprocess_features(self, 
                           X_train: np.ndarray, 
                           X_test: np.ndarray = None,
                           scaler_type: str = 'standard') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess features using specified scaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            scaler_type: Type of scaler ('standard', 'robust', or None)
            
        Returns:
            Tuple of scaled training features and test features (if provided)
        """
        if scaler_type is None:
            return X_train, X_test
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Fit scaler on training data
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Store the scaler for later use
        self.scalers['features'] = scaler
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        else:
            return X_train_scaled, None
    
    def train_single_property_model(self,
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  property_name: str,
                                  model_type: str = 'lightgbm',
                                  params: Dict = None,
                                  scaler_type: str = 'robust') -> BaseEstimator:
        """
        Train a model for a single molecular property.
        
        Args:
            X_train: Training features
            y_train: Training target values (for a single property)
            property_name: Name of the property
            model_type: Type of model ('lightgbm', 'xgboost', or 'random_forest')
            params: Model hyperparameters
            scaler_type: Type of scaler for preprocessing
            
        Returns:
            Trained model
        """
        start_time = time.time()
        self._log(f"Training {model_type} model for {property_name}")
        
        # Preprocess features
        X_train_scaled, _ = self.preprocess_features(X_train, scaler_type=scaler_type)
        
        # Create and train model
        if model_type not in self.model_types:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = self.model_types[model_type](params)
        model.fit(X_train_scaled, y_train)
        
        # Store model and hyperparameters
        self.models[property_name] = model
        self.hyperparams[property_name] = model.get_params()
        
        # Calculate feature importance
        if hasattr(model, 'feature_importances_'):
            self.feature_importances[property_name] = model.feature_importances_
        
        elapsed_time = time.time() - start_time
        self._log(f"Finished training {model_type} model for {property_name} in {elapsed_time:.2f} seconds")
        
        return model
    
    def train_all_properties(self,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            property_names: List[str],
                            model_type: str = 'lightgbm',
                            params: Dict = None,
                            scaler_type: str = 'robust') -> Dict[str, BaseEstimator]:
        """
        Train models for multiple molecular properties in parallel.
        
        Args:
            X_train: Training features
            y_train: Training target values (for multiple properties)
            property_names: Names of the properties
            model_type: Type of model
            params: Model hyperparameters
            scaler_type: Type of scaler for preprocessing
            
        Returns:
            Dictionary of trained models
        """
        self.property_names = property_names
        
        self._log(f"Training models for {len(property_names)} properties using {model_type}")
        
        # Preprocess features once
        X_train_scaled, _ = self.preprocess_features(X_train, scaler_type=scaler_type)
        
        # Train models in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._parallel_train_model)(
                X_train_scaled, y_train[:, i], property_name, model_type, params
            )
            for i, property_name in enumerate(property_names)
        )
        
        # Store results
        for property_name, model, hyperparams, importance in results:
            self.models[property_name] = model
            self.hyperparams[property_name] = hyperparams
            if importance is not None:
                self.feature_importances[property_name] = importance
        
        self._log(f"Finished training models for all properties")
        
        return self.models
    
    def _parallel_train_model(self,
                             X_train_scaled: np.ndarray,
                             y_train_prop: np.ndarray,
                             property_name: str,
                             model_type: str,
                             params: Dict) -> Tuple[str, BaseEstimator, Dict, np.ndarray]:
        """Helper function for parallel model training"""
        start_time = time.time()
        
        # Create and train model
        model = self.model_types[model_type](params)
        model.fit(X_train_scaled, y_train_prop)
        
        # Get hyperparameters and feature importance
        hyperparams = model.get_params()
        importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        
        elapsed_time = time.time() - start_time
        print(f"Trained {model_type} model for {property_name} in {elapsed_time:.2f} seconds")
        
        return property_name, model, hyperparams, importance
    
    def optimize_hyperparameters(self,
                               X_train: np.ndarray,
                               y_train: np.ndarray,
                               property_name: str,
                               model_type: str = 'lightgbm',
                               scaler_type: str = 'robust',
                               n_iter: int = 50,
                               optimizer: str = 'bayesian',
                               cv_folds: int = 5) -> Dict:
        """
        Optimize hyperparameters for a specific property model.
        
        Args:
            X_train: Training features
            y_train: Training target values
            property_name: Name of the property
            model_type: Type of model
            scaler_type: Type of scaler for preprocessing
            n_iter: Number of optimization iterations
            optimizer: Optimization strategy ('bayesian' or 'hyperopt')
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of optimized hyperparameters
        """
        self._log(f"Optimizing hyperparameters for {property_name} using {optimizer} optimization")
        
        # Preprocess features
        X_train_scaled, _ = self.preprocess_features(X_train, scaler_type=scaler_type)
        
        # Define parameter space based on model type
        if model_type == 'lightgbm':
            if optimizer == 'bayesian':
                param_space = {
                    'learning_rate': (0.01, 0.3),
                    'num_leaves': (20, 150),
                    'max_depth': (3, 12),
                    'min_child_samples': (5, 50),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0),
                    'reg_alpha': (0.0, 10.0),
                    'reg_lambda': (0.0, 10.0)
                }
                
                # Define objective function for Bayesian optimization
                def lgb_objective(learning_rate, num_leaves, max_depth, min_child_samples, 
                                subsample, colsample_bytree, reg_alpha, reg_lambda):
                    params = {
                        'learning_rate': learning_rate,
                        'num_leaves': int(num_leaves),
                        'max_depth': int(max_depth),
                        'min_child_samples': int(min_child_samples),
                        'subsample': subsample,
                        'colsample_bytree': colsample_bytree,
                        'reg_alpha': reg_alpha,
                        'reg_lambda': reg_lambda,
                        'n_estimators': 500,
                        'random_state': self.random_state,
                        'n_jobs': 1  # Use 1 job within CV to avoid nested parallelism
                    }
                    
                    # Cross-validation
                    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                    cv_scores = []
                    
                    for train_idx, val_idx in kf.split(X_train_scaled):
                        X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                        
                        model = lgb.LGBMRegressor(**params)
                        model.fit(X_cv_train, y_cv_train, 
                                 eval_set=[(X_cv_val, y_cv_val)],
                                 early_stopping_rounds=50,
                                 verbose=False)
                        
                        y_pred = model.predict(X_cv_val)
                        mse = mean_squared_error(y_cv_val, y_pred)
                        cv_scores.append(mse)
                    
                    # Return negative mean for minimization
                    return -np.mean(cv_scores)
                
                # Run Bayesian optimization
                optimizer = BayesianOptimization(
                    f=lgb_objective,
                    pbounds=param_space,
                    random_state=self.random_state,
                    verbose=2
                )
                
                optimizer.maximize(init_points=5, n_iter=n_iter)
                
                # Get best parameters
                best_params = optimizer.max['params']
                best_params['num_leaves'] = int(best_params['num_leaves'])
                best_params['max_depth'] = int(best_params['max_depth'])
                best_params['min_child_samples'] = int(best_params['min_child_samples'])
                best_params['n_estimators'] = 1000  # Use more estimators for final model
                best_params['random_state'] = self.random_state
                best_params['n_jobs'] = self.n_jobs
                
            elif optimizer == 'hyperopt':
                param_space = {
                    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
                    'num_leaves': scope.int(hp.quniform('num_leaves', 20, 150, 1)),
                    'max_depth': scope.int(hp.quniform('max_depth', 3, 12, 1)),
                    'min_child_samples': scope.int(hp.quniform('min_child_samples', 5, 50, 1)),
                    'subsample': hp.uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(10.0)),
                    'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(10.0))
                }
                
                # Define objective function for hyperopt
                def lgb_hyperopt_objective(params):
                    params = {
                        'learning_rate': params['learning_rate'],
                        'num_leaves': int(params['num_leaves']),
                        'max_depth': int(params['max_depth']),
                        'min_child_samples': int(params['min_child_samples']),
                        'subsample': params['subsample'],
                        'colsample_bytree': params['colsample_bytree'],
                        'reg_alpha': params['reg_alpha'],
                        'reg_lambda': params['reg_lambda'],
                        'n_estimators': 500,
                        'random_state': self.random_state,
                        'n_jobs': 1  # Use 1 job within CV to avoid nested parallelism
                    }
                    
                    # Cross-validation
                    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                    cv_scores = []
                    
                    for train_idx, val_idx in kf.split(X_train_scaled):
                        X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                        
                        model = lgb.LGBMRegressor(**params)
                        model.fit(X_cv_train, y_cv_train, 
                                 eval_set=[(X_cv_val, y_cv_val)],
                                 early_stopping_rounds=50,
                                 verbose=False)
                        
                        y_pred = model.predict(X_cv_val)
                        mse = mean_squared_error(y_cv_val, y_pred)
                        cv_scores.append(mse)
                    
                    return {'loss': np.mean(cv_scores), 'status': STATUS_OK}
                
                # Run hyperopt optimization
                trials = Trials()
                best = fmin(
                    fn=lgb_hyperopt_objective,
                    space=param_space,
                    algo=tpe.suggest,
                    max_evals=n_iter,
                    trials=trials,
                    rstate=np.random.RandomState(self.random_state)
                )
                
                # Get best parameters
                best_params = {
                    'learning_rate': best['learning_rate'],
                    'num_leaves': int(best['num_leaves']),
                    'max_depth': int(best['max_depth']),
                    'min_child_samples': int(best['min_child_samples']),
                    'subsample': best['subsample'],
                    'colsample_bytree': best['colsample_bytree'],
                    'reg_alpha': best['reg_alpha'],
                    'reg_lambda': best['reg_lambda'],
                    'n_estimators': 1000,  # Use more estimators for final model
                    'random_state': self.random_state,
                    'n_jobs': self.n_jobs
                }
                
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
                
        elif model_type == 'xgboost':
            # Define XGBoost parameter space (similar approach)
            if optimizer == 'bayesian':
                param_space = {
                    'learning_rate': (0.01, 0.3),
                    'max_depth': (3, 12),
                    'min_child_weight': (1, 10),
                    'gamma': (0, 1),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0),
                    'reg_alpha': (0.0, 10.0),
                    'reg_lambda': (0.0, 10.0)
                }
                
                # Define objective function for Bayesian optimization
                def xgb_objective(learning_rate, max_depth, min_child_weight, gamma, 
                                subsample, colsample_bytree, reg_alpha, reg_lambda):
                    params = {
                        'learning_rate': learning_rate,
                        'max_depth': int(max_depth),
                        'min_child_weight': min_child_weight,
                        'gamma': gamma,
                        'subsample': subsample,
                        'colsample_bytree': colsample_bytree,
                        'reg_alpha': reg_alpha,
                        'reg_lambda': reg_lambda,
                        'n_estimators': 500,
                        'random_state': self.random_state,
                        'n_jobs': 1  # Use 1 job within CV to avoid nested parallelism
                    }
                    
                    # Cross-validation (similar to LightGBM)
                    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                    cv_scores = []
                    
                    for train_idx, val_idx in kf.split(X_train_scaled):
                        X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                        
                        model = xgb.XGBRegressor(**params)
                        model.fit(X_cv_train, y_cv_train, 
                                 eval_set=[(X_cv_val, y_cv_val)],
                                 early_stopping_rounds=50,
                                 verbose=False)
                        
                        y_pred = model.predict(X_cv_val)
                        mse = mean_squared_error(y_cv_val, y_pred)
                        cv_scores.append(mse)
                    
                    # Return negative mean for minimization
                    return -np.mean(cv_scores)
                
                # Run Bayesian optimization
                optimizer = BayesianOptimization(
                    f=xgb_objective,
                    pbounds=param_space,
                    random_state=self.random_state,
                    verbose=2
                )
                
                optimizer.maximize(init_points=5, n_iter=n_iter)
                
                # Get best parameters
                best_params = optimizer.max['params']
                best_params['max_depth'] = int(best_params['max_depth'])
                best_params['n_estimators'] = 1000  # Use more estimators for final model
                best_params['random_state'] = self.random_state
                best_params['n_jobs'] = self.n_jobs
                
            else:
                # Hyperopt for XGBoost (similar approach)
                param_space = {
                    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
                    'max_depth': scope.int(hp.quniform('max_depth', 3, 12, 1)),
                    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
                    'gamma': hp.uniform('gamma', 0, 1),
                    'subsample': hp.uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(10.0)),
                    'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(10.0))
                }
                
                def xgb_hyperopt_objective(params):
                    params = {
                        'learning_rate': params['learning_rate'],
                        'max_depth': int(params['max_depth']),
                        'min_child_weight': params['min_child_weight'],
                        'gamma': params['gamma'],
                        'subsample': params['subsample'],
                        'colsample_bytree': params['colsample_bytree'],
                        'reg_alpha': params['reg_alpha'],
                        'reg_lambda': params['reg_lambda'],
                        'n_estimators': 500,
                        'random_state': self.random_state,
                        'n_jobs': 1
                    }
                    
                    # Cross-validation (similar to LightGBM)
                    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                    cv_scores = []
                    
                    for train_idx, val_idx in kf.split(X_train_scaled):
                        X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                        
                        model = xgb.XGBRegressor(**params)
                        model.fit(X_cv_train, y_cv_train, 
                                 eval_set=[(X_cv_val, y_cv_val)],
                                 early_stopping_rounds=50,
                                 verbose=False)
                        
                        y_pred = model.predict(X_cv_val)
                        mse = mean_squared_error(y_cv_val, y_pred)
                        cv_scores.append(mse)
                    
                    return {'loss': np.mean(cv_scores), 'status': STATUS_OK}
                
                # Run hyperopt optimization
                trials = Trials()
                best = fmin(
                    fn=xgb_hyperopt_objective,
                    space=param_space,
                    algo=tpe.suggest,
                    max_evals=n_iter,
                    trials=trials,
                    rstate=np.random.RandomState(self.random_state)
                )
                
                # Get best parameters
                best_params = {
                    'learning_rate': best['learning_rate'],
                    'max_depth': int(best['max_depth']),
                    'min_child_weight': best['min_child_weight'],
                    'gamma': best['gamma'],
                    'subsample': best['subsample'],
                    'colsample_bytree': best['colsample_bytree'],
                    'reg_alpha': best['reg_alpha'],
                    'reg_lambda': best['reg_lambda'],
                    'n_estimators': 1000,
                    'random_state': self.random_state,
                    'n_jobs': self.n_jobs
                }
        
        elif model_type == 'random_forest':
            # Define Random Forest parameter space
            if optimizer == 'bayesian':
                param_space = {
                    'n_estimators': (100, 1000),
                    'max_depth': (5, 30),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10),
                    'max_features': (0.1, 1.0)
                }
                
                def rf_objective(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
                    params = {
                        'n_estimators': int(n_estimators),
                        'max_depth': int(max_depth),
                        'min_samples_split': int(min_samples_split),
                        'min_samples_leaf': int(min_samples_leaf),
                        'max_features': max_features,
                        'random_state': self.random_state,
                        'n_jobs': 1
                    }
                    
                    # Cross-validation
                    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                    cv_scores = []
                    
                    for train_idx, val_idx in kf.split(X_train_scaled):
                        X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                        
                        model = RandomForestRegressor(**params)
                        model.fit(X_cv_train, y_cv_train)
                        
                        y_pred = model.predict(X_cv_val)
                        mse = mean_squared_error(y_cv_val, y_pred)
                        cv_scores.append(mse)
                    
                    return -np.mean(cv_scores)
                
                # Run Bayesian optimization
                optimizer = BayesianOptimization(
                    f=rf_objective,
                    pbounds=param_space,
                    random_state=self.random_state,
                    verbose=2
                )
                
                optimizer.maximize(init_points=5, n_iter=n_iter)
                
                # Get best parameters
                best_params = optimizer.max['params']
                best_params['n_estimators'] = int(best_params['n_estimators'])
                best_params['max_depth'] = int(best_params['max_depth'])
                best_params['min_samples_split'] = int(best_params['min_samples_split'])
                best_params['min_samples_split'] = int(best_params['min_samples_split'])
                best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
                best_params['random_state'] = self.random_state
                best_params['n_jobs'] = self.n_jobs
            
            else:
                # Hyperopt for Random Forest
                param_space = {
                    'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 10)),
                    'max_depth': scope.int(hp.quniform('max_depth', 5, 30, 1)),
                    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
                    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
                    'max_features': hp.uniform('max_features', 0.1, 1.0)
                }
                
                def rf_hyperopt_objective(params):
                    params = {
                        'n_estimators': int(params['n_estimators']),
                        'max_depth': int(params['max_depth']),
                        'min_samples_split': int(params['min_samples_split']),
                        'min_samples_leaf': int(params['min_samples_leaf']),
                        'max_features': params['max_features'],
                        'random_state': self.random_state,
                        'n_jobs': 1  # Use 1 job within CV to avoid nested parallelism
                    }
                    
                    # Cross-validation
                    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                    cv_scores = []
                    
                    for train_idx, val_idx in kf.split(X_train_scaled):
                        X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                        
                        model = RandomForestRegressor(**params)
                        model.fit(X_cv_train, y_cv_train)
                        
                        y_pred = model.predict(X_cv_val)
                        mse = mean_squared_error(y_cv_val, y_pred)
                        cv_scores.append(mse)
                    
                    return {'loss': np.mean(cv_scores), 'status': STATUS_OK}
                
                # Run hyperopt optimization
                trials = Trials()
                best = fmin(
                    fn=rf_hyperopt_objective,
                    space=param_space,
                    algo=tpe.suggest,
                    max_evals=n_iter,
                    trials=trials,
                    rstate=np.random.RandomState(self.random_state)
                )
                
                # Get best parameters
                best_params = {
                    'n_estimators': int(best['n_estimators']),
                    'max_depth': int(best['max_depth']),
                    'min_samples_split': int(best['min_samples_split']),
                    'min_samples_leaf': int(best['min_samples_leaf']),
                    'max_features': best['max_features'],
                    'random_state': self.random_state,
                    'n_jobs': self.n_jobs
                }
        else:
            raise ValueError(f"Hyperparameter optimization not implemented for model type: {model_type}")
        
        # Store best parameters
        self.best_params[property_name] = best_params
        self._log(f"Best parameters for {property_name}: {best_params}")
        
        # Train final model with best parameters
        final_model = self.train_single_property_model(
            X_train=X_train,
            y_train=y_train,
            property_name=property_name,
            model_type=model_type,
            params=best_params,
            scaler_type=scaler_type
        )
        
        return best_params
    
    def evaluate_model(self,
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     property_name: str = None) -> Dict:
        """
        Evaluate model performance on test data for a single property.
        
        Args:
            X_test: Test features
            y_test: Test target values
            property_name: Name of the property (if None, evaluate all properties)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if property_name is None and len(self.property_names) > 0:
            # Evaluate all models
            results = {}
            for i, prop_name in enumerate(self.property_names):
                prop_results = self.evaluate_model(X_test, y_test[:, i], prop_name)
                results[prop_name] = prop_results
            return results
        
        if property_name not in self.models:
            raise ValueError(f"No trained model found for property: {property_name}")
        
        # Preprocess test features
        if 'features' in self.scalers:
            X_test_scaled = self.scalers['features'].transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Make predictions
        model = self.models[property_name]
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'y_true': y_test.tolist() if isinstance(y_test, np.ndarray) else y_test,
            'y_pred': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred
        }
        
        self.evaluation_results[property_name] = results
        
        self._log(f"Evaluation results for {property_name}:")
        self._log(f"  MSE: {mse:.6f}")
        self._log(f"  RMSE: {rmse:.6f}")
        self._log(f"  MAE: {mae:.6f}")
        self._log(f"  R²: {r2:.6f}")
        
        return results
    
    def plot_feature_importance(self, property_name: str, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot feature importance for a specific property model.
        
        Args:
            property_name: Name of the property
            top_n: Number of top features to display
            figsize: Figure size (width, height)
        """
        if property_name not in self.feature_importances:
            raise ValueError(f"No feature importance found for property: {property_name}")
        
        importances = self.feature_importances[property_name]
        
        # Get indices of top features
        indices = np.argsort(importances)[-top_n:]
        
        # Create generic feature names if needed
        feature_names = [f"Feature {i}" for i in range(len(importances))]
        
        plt.figure(figsize=figsize)
        plt.title(f"Top {top_n} Feature Importances for {property_name}")
        plt.barh(range(top_n), importances[indices], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.results_dir, f"{property_name}_feature_importance.png")
        plt.savefig(plot_path)
        plt.close()
        
        self._log(f"Feature importance plot saved to {plot_path}")
    
    def plot_prediction_vs_actual(self, property_name: str, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot predicted vs actual values for a specific property.
        
        Args:
            property_name: Name of the property
            figsize: Figure size (width, height)
        """
        if property_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for property: {property_name}")
        
        results = self.evaluation_results[property_name]
        y_true = np.array(results['y_true'])
        y_pred = np.array(results['y_pred'])
        
        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Plot perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title(f"Predicted vs Actual Values for {property_name}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True, alpha=0.3)
        
        # Add metrics to plot
        metrics_text = (
            f"R² = {results['r2']:.4f}\n"
            f"RMSE = {results['rmse']:.4f}\n"
            f"MAE = {results['mae']:.4f}"
        )
        plt.text(
            0.05, 0.95, metrics_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', alpha=0.5)
        )
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.results_dir, f"{property_name}_pred_vs_actual.png")
        plt.savefig(plot_path)
        plt.close()
        
        self._log(f"Prediction vs actual plot saved to {plot_path}")
    
    def save_model(self, property_name: str = None, format: str = 'pickle'):
        """
        Save trained model(s) to disk.
        
        Args:
            property_name: Name of the property (if None, save all models)
            format: Model format ('pickle' or 'onnx')
        """
        if property_name is None:
            # Save all models
            for prop_name in self.models:
                self.save_model(prop_name, format)
            return
        
        if property_name not in self.models:
            raise ValueError(f"No trained model found for property: {property_name}")
        
        model = self.models[property_name]
        
        if format == 'pickle':
            # Save model with pickle
            model_path = os.path.join(self.model_dir, f"{property_name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
            # Save scaler if available
            if 'features' in self.scalers:
                scaler_path = os.path.join(self.model_dir, f"{property_name}_scaler.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers['features'], f)
                    
            # Save hyperparameters
            if property_name in self.hyperparams:
                params_path = os.path.join(self.model_dir, f"{property_name}_params.json")
                with open(params_path, 'w') as f:
                    json.dump(self.hyperparams[property_name], f, indent=2)
                    
            self._log(f"Model for {property_name} saved to {model_path}")
            
        elif format == 'onnx':
            # Convert and save as ONNX
            try:
                import onnxmltools
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                
                # Define input shape
                initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
                
                # Convert to ONNX
                onnx_model = convert_sklearn(model, initial_types=initial_type)
                
                # Save ONNX model
                model_path = os.path.join(self.model_dir, f"{property_name}_model.onnx")
                onnxmltools.utils.save_model(onnx_model, model_path)
                
                self._log(f"ONNX model for {property_name} saved to {model_path}")
                
            except ImportError:
                self._log("ONNX conversion requires onnxmltools and skl2onnx. Using pickle instead.")
                self.save_model(property_name, format='pickle')
                
        else:
            raise ValueError(f"Unsupported model format: {format}")
    
    def load_model(self, property_name: str, model_path: str = None, scaler_path: str = None):
        """
        Load a trained model from disk.
        
        Args:
            property_name: Name of the property
            model_path: Path to the model file (if None, use default path)
            scaler_path: Path to the scaler file (if None, use default path)
        """
        if model_path is None:
            model_path = os.path.join(self.model_dir, f"{property_name}_model.pkl")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load model
        with open(model_path, 'rb') as f:
            self.models[property_name] = pickle.load(f)
            
        # Load scaler if available
        if scaler_path is None:
            scaler_path = os.path.join(self.model_dir, f"{property_name}_scaler.pkl")
            
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scalers['features'] = pickle.load(f)
                
        # Load hyperparameters if available
        params_path = os.path.join(self.model_dir, f"{property_name}_params.json")
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                self.hyperparams[property_name] = json.load(f)
                
        self._log(f"Model for {property_name} loaded from {model_path}")
    
    def predict(self, X: np.ndarray, property_name: str = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make predictions for new molecules.
        
        Args:
            X: Feature matrix
            property_name: Name of the property (if None, predict all properties)
            
        Returns:
            Predictions for the specified property or dictionary of predictions for all properties
        """
        if property_name is None:
            # Predict all properties
            predictions = {}
            for prop_name in self.models:
                predictions[prop_name] = self.predict(X, prop_name)
            return predictions
        
        if property_name not in self.models:
            raise ValueError(f"No trained model found for property: {property_name}")
        
        # Preprocess features
        if 'features' in self.scalers:
            X_scaled = self.scalers['features'].transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        model = self.models[property_name]
        predictions = model.predict(X_scaled)
        
        return predictions
    
    def get_uncertainty(self, X: np.ndarray, property_name: str, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate prediction uncertainty using bootstrapping or model-specific methods.
        
        Args:
            X: Feature matrix
            property_name: Name of the property
            n_samples: Number of bootstrap samples
            
        Returns:
            Tuple of mean predictions and standard deviations
        """
        if property_name not in self.models:
            raise ValueError(f"No trained model found for property: {property_name}")
        
        model = self.models[property_name]
        
        # Preprocess features
        if 'features' in self.scalers:
            X_scaled = self.scalers['features'].transform(X)
        else:
            X_scaled = X
        
        # Check if model supports prediction with standard deviation
        if hasattr(model, 'predict_with_std'):
            # For models that have built-in uncertainty estimation
            mean_preds, std_preds = model.predict_with_std(X_scaled)
            return mean_preds, std_preds
        
        # For tree-based models, use bootstrapping approach
        if isinstance(model, (lgb.LGBMRegressor, xgb.XGBRegressor, RandomForestRegressor)):
            preds = []
            n_trees = model.n_estimators
            
            # For LightGBM
            if isinstance(model, lgb.LGBMRegressor):
                for i in range(n_samples):
                    # Randomly select trees for each bootstrap sample
                    tree_indices = np.random.choice(n_trees, size=n_trees//2, replace=False)
                    pred = np.zeros(X_scaled.shape[0])
                    
                    for tree_idx in tree_indices:
                        # Get prediction from individual tree
                        pred += model.predict(X_scaled, start_iteration=tree_idx, num_iteration=1)
                        
                    pred /= len(tree_indices)
                    preds.append(pred)
            
            # For XGBoost
            elif isinstance(model, xgb.XGBRegressor):
                for i in range(n_samples):
                    tree_indices = np.random.choice(n_trees, size=n_trees//2, replace=False)
                    booster = model.get_booster()
                    
                    # Create new model with subset of trees
                    pred = np.zeros(X_scaled.shape[0])
                    for tree_idx in tree_indices:
                        # Extract prediction from individual tree
                        tree_pred = booster.predict(xgb.DMatrix(X_scaled), 
                                                  ntree_limit=tree_idx+1) - booster.predict(xgb.DMatrix(X_scaled), 
                                                                                           ntree_limit=tree_idx)
                        pred += tree_pred
                    
                    pred /= len(tree_indices)
                    preds.append(pred)
            
            # For Random Forest, simply use the trees directly
            elif isinstance(model, RandomForestRegressor):
                # Get predictions from all trees
                trees = model.estimators_
                
                for i in range(n_samples):
                    # Sample trees with replacement
                    tree_indices = np.random.choice(len(trees), size=len(trees), replace=True)
                    
                    # Average predictions from sampled trees
                    bootstrap_preds = np.array([trees[idx].predict(X_scaled) for idx in tree_indices])
                    preds.append(np.mean(bootstrap_preds, axis=0))
            
            # Convert predictions to array and calculate statistics
            preds_array = np.array(preds)
            mean_preds = np.mean(preds_array, axis=0)
            std_preds = np.std(preds_array, axis=0)
            
            return mean_preds, std_preds
        
        # For other models, just return predictions with zeros for std
        mean_preds = model.predict(X_scaled)
        std_preds = np.zeros_like(mean_preds)
        
        return mean_preds, std_preds
    
    def generate_prediction_report(self, molecule_smiles: str, rdkit_mol: Chem.Mol, predictions: Dict[str, float], 
                                  uncertainties: Dict[str, float] = None):
        """
        Generate a comprehensive prediction report for a molecule.
        
        Args:
            molecule_smiles: SMILES string of the molecule
            rdkit_mol: RDKit molecule object
            predictions: Dictionary of property predictions
            uncertainties: Dictionary of uncertainty estimates (optional)
            
        Returns:
            Dictionary containing the report data
        """
        # Ensure we have a valid RDKit molecule
        if rdkit_mol is None:
            rdkit_mol = Chem.MolFromSmiles(molecule_smiles)
            
        if rdkit_mol is None:
            raise ValueError(f"Invalid SMILES string: {molecule_smiles}")
        
        # Create report dictionary
        report = {
            'smiles': molecule_smiles,
            'predictions': predictions,
            'molecular_weight': Descriptors.MolWt(rdkit_mol),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add uncertainties if available
        if uncertainties:
            report['uncertainties'] = uncertainties
        
        # Add basic descriptors
        report['descriptors'] = {
            'heavy_atoms': Descriptors.HeavyAtomCount(rdkit_mol),
            'aromatic_rings': Descriptors.NumAromaticRings(rdkit_mol),
            'h_donors': Descriptors.NumHDonors(rdkit_mol),
            'h_acceptors': Descriptors.NumHAcceptors(rdkit_mol),
            'rotatable_bonds': Lipinski.NumRotatableBonds(rdkit_mol),
            'tpsa': MolSurf.TPSA(rdkit_mol)
        }
        
        # Save report
        report_path = os.path.join(self.results_dir, f"prediction_{int(time.time())}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def find_similar_molecules(self, query_mol: Chem.Mol, dataset_mols: List[Chem.Mol], 
                             dataset_props: Dict[str, List[float]], top_n: int = 5) -> List[Dict]:
        """
        Find molecules in the dataset similar to the query molecule.
        
        Args:
            query_mol: Query molecule (RDKit Mol object)
            dataset_mols: List of dataset molecules (RDKit Mol objects)
            dataset_props: Dictionary of property values for dataset molecules
            top_n: Number of similar molecules to return
            
        Returns:
            List of dictionaries containing similar molecules and their properties
        """
        # Generate fingerprint for query molecule
        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
        
        # Calculate similarities
        similarities = []
        for i, mol in enumerate(dataset_mols):
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(query_fp, fp)
            similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top similar molecules
        similar_mols = []
        for i in range(min(top_n, len(similarities))):
            idx, sim = similarities[i]
            mol = dataset_mols[idx]
            
            # Get properties
            props = {}
            for prop_name, values in dataset_props.items():
                if idx < len(values):
                    props[prop_name] = values[idx]
            
            similar_mols.append({
                'smiles': Chem.MolToSmiles(mol),
                'similarity': sim,
                'properties': props
            })
        
        return similar_mols


# Example usage for training models:
if __name__ == "__main__":
    # Load processed data
    features = np.load("processed_data/features.npy")
    properties = np.load("processed_data/properties.npy")
    
    with open("processed_data/property_names.pkl", "rb") as f:
        property_names = pickle.load(f)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, properties, test_size=0.2, random_state=42
    )
    
    # Initialize predictor
    predictor = MolecularPropertyPredictor(
        model_dir="models",
        results_dir="results",
        n_jobs=-1,  # Use all available cores
        random_state=42
    )
    
    # Train models for all properties
    models = predictor.train_all_properties(
        X_train=X_train,
        y_train=y_train,
        property_names=property_names,
        model_type='lightgbm',
        scaler_type='robust'
    )
    
    # Optimize hyperparameters for one property (optional)
    # best_params = predictor.optimize_hyperparameters(
    #     X_train=X_train,
    #     y_train=y_train[:, 0],  # First property
    #     property_name=property_names[0],
    #     model_type='lightgbm',
    #     n_iter=30
    # )
    
    # Evaluate models
    evaluation_results = predictor.evaluate_model(X_test, y_test)
    
    # Plot results for first property
    predictor.plot_feature_importance(property_names[0])
    predictor.plot_prediction_vs_actual(property_names[0])
    
    # Save models
    predictor.save_model()
    
    print("Model training and evaluation complete!")
