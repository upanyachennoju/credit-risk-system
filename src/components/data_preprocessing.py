import os
import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import yaml

from src.components.data_ingestion import DataIngestion

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessing:
    """
    Handles data preprocessing including:
    1. Train-test split
    2. Feature imputation and scaling
    3. Saving preprocessors and transformed data
    """
    
    def __init__(self, config_path: str = "config/schema.yaml", test_size: float = 0.2, random_state: int = 42):
        """
        Initialize DataPreprocessing.
        
        Args:
            config_path: Path to schema configuration file
            test_size: Proportion of test set (default 0.2)
            random_state: Random seed for reproducibility
        """
        self.config_path = self._resolve_path(config_path)
        self.test_size = test_size
        self.random_state = random_state
        
        # Load schema configuration
        self.schema = self._load_schema()
        self.numerical_columns = self.schema['metadata']['numerical_columns']
        self.categorical_columns = self.schema['metadata']['categorical_columns']
        self.target_column = 'target'
        self.preprocessor = None
        
    def _resolve_path(self, path: str) -> Path:
        """Resolve relative paths to absolute paths."""
        path = Path(path)
        
        if path.is_absolute() and path.exists():
            return path
        
        # Find project root by looking for 'config' folder
        current = Path.cwd()
        while current != current.parent:
            if (current / "config").exists():
                resolved = current / path
                if resolved.exists():
                    return resolved
            current = current.parent
        
        if Path(path).exists():
            return Path(path).resolve()
        
        raise FileNotFoundError(f"Config file not found: {path}")
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load schema configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                schema = yaml.safe_load(f)
            logger.info(f"Schema loaded from {self.config_path}")
            return schema
        except Exception as e:
            logger.error(f"Error loading schema: {e}")
            raise
    
    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Create ColumnTransformer with pipelines for numeric and categorical features.
        
        Returns:
            ColumnTransformer: Fitted preprocessor
        """
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, self.numerical_columns),
                ('cat', categorical_pipeline, self.categorical_columns)
            ]
        )
        
        logger.info("✓ Preprocessor (ColumnTransformer) created")
        return preprocessor
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline:
        1. Split X and y
        2. Stratified train-test split
        3. Fit preprocessor on train data
        4. Transform train and test data
        
        Args:
            df: Input DataFrame with all features and target
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test) as numpy arrays
        """
        logger.info("=" * 60)
        logger.info("Starting Data Preprocessing Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Split X and y
        logger.info("\n[1/4] Splitting features and target...")
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Step 2: Stratified train-test split
        logger.info("\n[2/4] Performing stratified train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")
        
        # Step 3: Create and fit preprocessor on train data
        logger.info("\n[3/4] Creating and fitting preprocessor on training data...")
        self.preprocessor = self._create_preprocessor()
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        logger.info(f"Preprocessor fitted on train data")
        logger.info(f"Transformed shape: {X_train_transformed.shape}")
        
        # Step 4: Transform test data (using fitted preprocessor)
        logger.info("\n[4/4] Transforming test data...")
        X_test_transformed = self.preprocessor.transform(X_test)
        logger.info(f"Test data transformed")
        logger.info(f"Transformed shape: {X_test_transformed.shape}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Preprocessing Complete!")
        logger.info("=" * 60)
        logger.info("Note: Class imbalance handled via threshold tuning in model training")
        
        return X_train_transformed, X_test_transformed, y_train, y_test
    
    def save_preprocessor(self, output_dir: str = "artifacts/preprocessors"):
        """
        Save preprocessor pipeline to pickle file.
        
        Args:
            output_dir: Directory to save preprocessor
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        preprocessor_file = output_path / "preprocessing.pkl"
        
        try:
            with open(preprocessor_file, 'wb') as f:
                pickle.dump(self.preprocessor, f)
            logger.info(f"✓ Preprocessor saved to {preprocessor_file}")
        except Exception as e:
            logger.error(f"✗ Error saving preprocessor: {e}")
            raise
    
    def save_transformed_data(self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        output_dir: str = "artifacts/transformed_data"):
        """
        Save transformed data arrays to files.
        
        Args:
            X_train: Transformed training features
            X_test: Transformed test features
            y_train: Training target
            y_test: Test target
            output_dir: Directory to save transformed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            np.save(output_path / "X_train.npy", X_train)
            np.save(output_path / "X_test.npy", X_test)
            np.save(output_path / "y_train.npy", y_train)
            np.save(output_path / "y_test.npy", y_test)
            
            logger.info(f"Transformed data saved to {output_path}")
            logger.info(f" - X_train.npy: {X_train.shape}")
            logger.info(f" - X_test.npy: {X_test.shape}")
            logger.info(f" - y_train.npy: {y_train.shape}")
            logger.info(f" - y_test.npy: {y_test.shape}")
        except Exception as e:
            logger.error(f"✗ Error saving transformed data: {e}")
            raise


def run_preprocessing_pipeline(
    data_path: str = "data/raw/synthetic_credit_risk.csv",
    config_path: str = "config/schema.yaml",
    test_size: float = 0.2,
    preprocessor_output_dir: str = "artifacts/preprocessors",
    data_output_dir: str = "artifacts/transformed_data"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute the complete preprocessing pipeline.
    
    Args:
        data_path: Path to raw data CSV
        config_path: Path to schema YAML
        test_size: Test set proportion
        preprocessor_output_dir: Directory to save preprocessor
        data_output_dir: Directory to save transformed data
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test) as model-ready arrays
    """
    # Load data
    data_ingestion = DataIngestion(data_path=data_path)
    df = data_ingestion.load_data()
    df = data_ingestion.validate_data(df)
    
    # Preprocess data
    preprocessor = DataPreprocessing(config_path=config_path, test_size=test_size)
    X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
    
    # Save artifacts
    preprocessor.save_preprocessor(preprocessor_output_dir)
    preprocessor.save_transformed_data(X_train, X_test, y_train, y_test, data_output_dir)
    
    logger.info("\n🎯 Preprocessing pipeline completed successfully!")
    logger.info(f"   Model-ready arrays ready for training")
    
    return X_train, X_test, y_train, y_test

