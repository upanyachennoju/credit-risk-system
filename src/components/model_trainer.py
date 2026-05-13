import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains multiple credit risk models with configured hyperparameters.
    
    Responsibilities:
    1. Initiate model instances with best hyperparameters
    2. Train models on preprocessed data
    3. Return dictionary of trained models
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize ModelTrainer.
        
        Args:
            config_path: Path to configuration file with model hyperparameters
        """
        self.config_path = self._resolve_path(config_path)
        self.config = self._load_config()
        self.random_state = self.config['training']['random_state']
        self.model_params = {}
        
    def _resolve_path(self, path: str) -> Path:
        """Resolve relative paths to absolute paths."""
        path = Path(path)
        
        if path.is_absolute() and path.exists():
            return path
        
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
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✓ Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"✗ Error loading config: {e}")
            raise
    
    def initiate_models(self) -> Dict[str, Any]:
        """
        Initiate model instances with configured hyperparameters.
        
        Returns:
            Dict: Dictionary of initialized model instances {model_name: model}
        """
        logger.info("\n" + "=" * 60)
        logger.info("Initiating Models")
        logger.info("=" * 60)
        
        try:
            models = {}
            
            # Logistic Regression
            lr_params = self.config['models']['logistic_regression']['best_params']
            models['logistic_regression'] = LogisticRegression(**lr_params)
            self.model_params['logistic_regression'] = lr_params
            logger.info("✓ Logistic Regression initialized")
            
            # Random Forest
            rf_params = self.config['models']['random_forest']['best_params']
            models['random_forest'] = RandomForestClassifier(**rf_params)
            self.model_params['random_forest'] = rf_params
            logger.info("✓ Random Forest initialized")
            
            # XGBoost
            xgb_params = self.config['models']['xgboost']['best_params'].copy()
            models['xgboost'] = XGBClassifier(**xgb_params)
            self.model_params['xgboost'] = xgb_params
            logger.info("✓ XGBoost initialized")
            
            logger.info("✓ All models initiated successfully\n")
            return models
        
        except Exception as e:
            logger.error(f"✗ Error initiating models: {e}")
            raise
    
    def train_models(
        self,
        models: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train all models on training data.
        
        Args:
            models: Dictionary of model instances {model_name: model}
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dict: Dictionary of trained models {model_name: trained_model}
        """
        logger.info("\n" + "=" * 60)
        logger.info("Training Models")
        logger.info("=" * 60)
        
        trained_models = {}
        
        try:
            # Handle scale_pos_weight auto-calculation for XGBoost
            if 'xgboost' in models:
                xgb_params = self.model_params['xgboost']
                if xgb_params.get('scale_pos_weight') == 'auto':
                    neg_count = np.sum(y_train == 0)
                    pos_count = np.sum(y_train == 1)
                    scale_pos_weight = neg_count / pos_count
                    xgb_params['scale_pos_weight'] = scale_pos_weight
                    models['xgboost'] = XGBClassifier(**xgb_params)
                    logger.info(f"  Auto-calculated XGBoost scale_pos_weight: {scale_pos_weight:.4f}")
            
            # Train each model
            for i, (model_name, model) in enumerate(models.items(), 1):
                logger.info(f"\n[{i}/{len(models)}] Training {model_name}...")
                
                try:
                    model.fit(X_train, y_train)
                    trained_models[model_name] = model
                    logger.info(f"✓ {model_name} trained successfully")
                
                except Exception as e:
                    logger.error(f"✗ Error training {model_name}: {e}")
                    raise
            
            logger.info("\n✓ All models trained successfully\n")
            return trained_models
        
        except Exception as e:
            logger.error(f"✗ Error in train_models: {e}")
            raise


if __name__ == "__main__":
    pass
        