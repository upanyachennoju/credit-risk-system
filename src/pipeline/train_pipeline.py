"""
End-to-end pipeline example for credit risk model training with MLflow.

This script demonstrates the complete workflow:
1. Data ingestion
2. Data preprocessing
3. Model training
4. Model evaluation
5. MLflow logging
"""

import logging
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import run_model_evaluation_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Execute complete credit risk model training pipeline with MLflow tracking."""
    
    logger.info("\n" + "=" * 80)
    logger.info("CREDIT RISK MODEL TRAINING PIPELINE WITH MLFLOW")
    logger.info("=" * 80)
    
    try:
        logger.info("\n[STEP 1/5] Initializing MLflow...")
        mlflow_config = initialize_mlflow(config_path="config/config.yaml")
        
        logger.info("\n[STEP 2/5] Loading and ingesting data...")
        data_ingestion = DataIngestion(data_path="data/raw/synthetic_credit_risk.csv")
        df = data_ingestion.load_data()
        df = data_ingestion.validate_data(df)
        
        logger.info("\n[STEP 3/5] Preprocessing data...")
        preprocessor = DataPreprocessing(config_path="config/schema.yaml")
        X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
        preprocessor.save_preprocessor()
        preprocessor.save_transformed_data(X_train, X_test, y_train, y_test)
        
        logger.info("\n[STEP 4/5] Training models...")
        trainer = ModelTrainer(config_path="config/config.yaml")
        
        models = trainer.initiate_models()
        trained_models = trainer.train_models(models, X_train, y_train)
        
        logger.info("\n[STEP 5/5] Evaluating models and logging to MLflow...")
        best_model, best_model_name, metrics_df = run_model_evaluation_pipeline(
            models=trained_models,
            X_test=X_test,
            y_test=y_test,
            model_params=trainer.model_params,
            config_path="config/config.yaml"
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"\nBest Model: {best_model_name}")
        logger.info(f"\nMetrics Summary:\n{metrics_df.to_string(index=False)}")
        logger.info("\n✓ Model artifacts saved to: artifacts/models/best_model.pkl")
        logger.info("✓ Metrics report saved to: artifacts/reports/model_evaluation_metrics.csv")
        logger.info("✓ All runs logged to MLflow!")
        
        return best_model, best_model_name, metrics_df
    
    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    best_model, best_model_name, metrics_df = main()
