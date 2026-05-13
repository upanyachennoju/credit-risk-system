import os
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Handles data loading, validation, and saving for the credit risk model.
    
    This class:
    1. Loads raw data from CSV files
    2. Validates data integrity (duplicates, missing values, schema)
    3. Saves processed data to output directory
    """
    
    def __init__(self, data_path="data/raw/synthetic_credit_risk.csv"):
        """
        Initialize DataIngestion with a data path.
        
        Args:
            data_path: Path to the raw CSV file (relative or absolute)
        """
        self.data_path = self._resolve_path(data_path)
        self.expected_columns = None  
        
    def _resolve_path(self, path: str) -> Path:
        """
        Resolve relative paths to absolute paths, finding project root.
        This ensures the code works regardless of where it's run from.
        """
        path = Path(path)
        
        # If it's already absolute and exists, use it
        if path.is_absolute() and path.exists():
            return path
        
        # Find project root by looking for 'data' folder
        current = Path.cwd()
        while current != current.parent:
            if (current / "data").exists():
                resolved = current / path
                if resolved.exists():
                    return resolved
            current = current.parent
        
        # If not found, try relative to current directory
        if Path(path).exists():
            return Path(path).resolve()
        
        raise FileNotFoundError(f"Data file not found: {path}")

    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            pd.DataFrame: The loaded dataset
            
        Raises:
            Exception: If file cannot be read
        """
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully from {self.data_path}")
            logger.info(f"-Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
            logger.info(f"-Columns: {', '.join(df.columns.tolist())}")
            
            # Store expected columns for future validation
            self.expected_columns = df.columns.tolist()
            
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise Exception(f"Failed to load data: {e}")
        
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data integrity.
        
        Checks for:
        - Duplicate rows
        - Missing values
        - Schema consistency
        
        Args:
            df: DataFrame to validate
            
        Returns:
            pd.DataFrame: The validated dataframe
            
        Raises:
            Exception: If validation fails
        """
        logger.info("Starting data validation...")
        
        # Check duplicates
        duplicates = df.duplicated().sum()
        logger.info(f"  Duplicate rows found: {duplicates}")
        if duplicates > 0:
            logger.warning(f"  Warning: {duplicates} duplicate rows detected")
        
        # Check missing values
        missing = df.isnull().sum()
        missing_count = missing.sum()
        logger.info(f"Missing values: {missing_count}")
        if missing_count > 0:
            logger.warning(f"Columns with missing values:\n{missing[missing > 0]}")
        
        # Check schema consistency
        if self.expected_columns and list(df.columns) != self.expected_columns:
            raise Exception(f"Schema mismatch! Expected {self.expected_columns}, got {list(df.columns)}")
        
        logger.info("Schema validation passed")
        logger.info(f"All {len(df.columns)} columns are present and in correct order")
        
        return df
    
    def save_data(self, df: pd.DataFrame, output_path="data/processed/validated_data.csv") -> Path:
        """
        Save processed data to CSV file.
        
        Args:
            df: DataFrame to save
            output_path: Path where to save the file (creates directories if needed)
            
        Returns:
            Path: The path where data was saved
        """
        output_path = self._resolve_path(output_path) if Path(output_path).is_absolute() else Path(output_path)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            df.to_csv(output_path, index=False)
            logger.info(f"✓ Validated data saved to {output_path}")
            logger.info(f"  Saved {df.shape[0]} rows x {df.shape[1]} columns")
            return output_path
        except Exception as e:
            logger.error(f"✗ Error saving data: {e}")
            raise Exception(f"Failed to save data: {e}")
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get a summary of the dataset.
        
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_rows': df.shape[0],
            'total_columns': df.shape[1],
            'column_names': df.columns.tolist(),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum()
        }
        return summary
