import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates trained models on test data with comprehensive metrics.
    
    Responsibilities:
    1. Evaluate each model with threshold tuning
    2. Calculate comprehensive metrics (precision, recall, F1, ROC-AUC, PR-AUC)
    3. Log results to MLflow (if available)
    4. Compare models and select best
    5. Save best model
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize ModelEvaluator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = self._resolve_path(config_path)
        self.config = self._load_config()
        
        # Results storage
        self.evaluation_results = {}
        self.metrics_df = None
        self.best_model = None
        self.best_model_name = None
        self.best_metrics = None
        
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
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single model with comprehensive metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            threshold: Probability threshold (None uses default predict)
            
        Returns:
            Dict: Evaluation results including metrics and predictions
        """
        logger.info(f"\nEvaluating {model_name}...")
        
        try:
            # Get predictions
            y_proba = model.predict_proba(X_test)[:, 1]
            
            if threshold is not None:
                y_pred = (y_proba >= threshold).astype(int)
            else:
                y_pred = model.predict(X_test)
                threshold = 0.5
            
            # Calculate metrics
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)
            
            # Calculate PR-AUC
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recall_curve, precision_curve)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # Balanced accuracy
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            balanced_accuracy = (sensitivity + specificity) / 2
            
            results = {
                'model_name': model_name,
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'specificity': specificity,
                'balanced_accuracy': balanced_accuracy,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
            logger.info(f"  ✓ Precision:         {precision:.4f}")
            logger.info(f"  ✓ Recall:            {recall:.4f}")
            logger.info(f"  ✓ F1-Score:          {f1:.4f}")
            logger.info(f"  ✓ ROC-AUC:           {roc_auc:.4f}")
            logger.info(f"  ✓ PR-AUC:            {pr_auc:.4f}")
            logger.info(f"  ✓ Specificity:       {specificity:.4f}")
            logger.info(f"  ✓ Balanced Accuracy: {balanced_accuracy:.4f}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            raise
    
    def log_to_mlflow(
        self,
        model_name: str,
        metrics: Dict[str, float],
        model_params: Dict[str, Any],
        model: Any
    ) -> None:
        """
        Log model parameters, metrics, and artifact to MLflow.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
            model_params: Dictionary of model parameters
            model: Trained model object
        """
        
        try:
            with mlflow.start_run(run_name=model_name):
                # Log parameters
                mlflow.log_params(model_params)
                
                # Log metrics
                mlflow.log_metrics({
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'roc_auc': metrics['roc_auc'],
                    'pr_auc': metrics['pr_auc'],
                    'specificity': metrics['specificity'],
                    'balanced_accuracy': metrics['balanced_accuracy']
                })
                
                # Log model artifact
                model_path = f"artifacts/mlflow/{model_name}_model.pkl"
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                mlflow.sklearn.log_model(model, "model")
                
                logger.info(f"MLflow logging completed for {model_name}")
        
        except Exception as e:
            logger.warning(f"MLflow logging failed for {model_name}: {e}")
    
    def evaluate_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_params: Dict[str, Dict[str, Any]] = None
    ) -> None:
        """
        Evaluate all models.
        
        Args:
            models: Dictionary of trained models {model_name: model_object}
            X_test: Test features
            y_test: Test target
            model_params: Dictionary of model parameters for logging
        """
        logger.info("\n" + "=" * 60)
        logger.info("Model Evaluation Pipeline")
        logger.info("=" * 60)
        
        for model_name, model in models.items():
            # Get threshold from config
            threshold = self.config['models'][model_name].get('threshold')
            
            # Evaluate model
            results = self.evaluate_model(
                model, X_test, y_test, model_name, threshold
            )
            
            self.evaluation_results[model_name] = results
            
            # Log to MLflow
            if model_params and model_name in model_params:
                self.log_to_mlflow(
                    model_name,
                    results,
                    model_params[model_name],
                    model
                )
    
    def compare_models(self) -> pd.DataFrame:
        """
        Create comparison dataframe of all models.
        
        Returns:
            pd.DataFrame: Metrics comparison for all models
        """
        logger.info("\n" + "-" * 60)
        logger.info("Model Comparison Summary")
        logger.info("-" * 60)
        
        metrics_data = []
        for model_name, results in self.evaluation_results.items():
            metrics_data.append({
                'Model': model_name,
                'Threshold': f"{results['threshold']:.2f}",
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1'],
                'ROC-AUC': results['roc_auc'],
                'PR-AUC': results['pr_auc'],
                'Specificity': results['specificity'],
                'Balanced Accuracy': results['balanced_accuracy']
            })
        
        self.metrics_df = pd.DataFrame(metrics_data)
        
        logger.info("\n" + self.metrics_df.to_string(index=False))
        
        return self.metrics_df
    
    def select_best_model(self, models: Dict[str, Any]) -> Tuple[Any, str]:
        """
        Select best model based on highest F1 score.
        
        Args:
            models: Dictionary of trained models
            
        Returns:
            Tuple: (best_model, best_model_name)
        """
        logger.info("\n" + "-" * 60)
        logger.info("Selecting Best Model")
        logger.info("-" * 60)
        
        best_f1 = -1
        for model_name, results in self.evaluation_results.items():
            f1_score_val = results['f1']
            if f1_score_val > best_f1:
                best_f1 = f1_score_val
                self.best_model_name = model_name
                self.best_model = models[model_name]
                self.best_metrics = results
        
        logger.info(f"\n✓ Best Model: {self.best_model_name}")
        logger.info(f"  F1-Score: {best_f1:.4f}")
        logger.info(f"  Precision: {self.best_metrics['precision']:.4f}")
        logger.info(f"  Recall: {self.best_metrics['recall']:.4f}")
        logger.info(f"  ROC-AUC: {self.best_metrics['roc_auc']:.4f}")
        
        return self.best_model, self.best_model_name
    
    def save_best_model(self, output_dir: str = "artifacts/models") -> Path:
        """
        Save best model to pickle file.
        
        Args:
            output_dir: Directory to save model
            
        Returns:
            Path: Path where model was saved
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_file = output_path / "best_model.pkl"
        
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(self.best_model, f)
            
            logger.info(f"\n✓ Best model saved to {model_file}")
            logger.info(f"  Model: {self.best_model_name}")
            logger.info(f"  F1-Score: {self.best_metrics['f1']:.4f}")
            
            return model_file
        
        except Exception as e:
            logger.error(f"✗ Error saving best model: {e}")
            raise
    


def run_model_evaluation_pipeline(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_params: Dict[str, Dict[str, Any]] = None,
    config_path: str = "config/config.yaml",
    model_output_dir: str = "artifacts/models",
    report_output_dir: str = "artifacts/reports"
) -> Tuple[Any, str, pd.DataFrame]:
    """
    Execute the complete model evaluation pipeline.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test target
        model_params: Optional model parameters for MLflow logging
        config_path: Path to config file
        model_output_dir: Directory to save best model
        report_output_dir: Directory to save reports
        
    Returns:
        Tuple: (best_model, best_model_name, metrics_df)
    """
    # Initialize evaluator
    evaluator = ModelEvaluator(config_path=config_path)
    
    # Evaluate all models
    evaluator.evaluate_models(models, X_test, y_test, model_params)
    
    # Compare models
    metrics_df = evaluator.compare_models()
    
    # Select best model
    best_model, best_model_name = evaluator.select_best_model(models)
    
    # Save artifacts
    evaluator.save_best_model(model_output_dir)
    report_path = Path(report_output_dir) / "model_evaluation_metrics.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(report_path, index=False)
    logger.info(f"✓ Metrics report saved to {report_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Model evaluation pipeline completed successfully!")
    logger.info("=" * 60)
    
    return best_model, best_model_name, metrics_df

