import os
from pathlib import Path

project_name = "src"

list_of_files = [

    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_preprocessing.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",

    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/train_pipeline.py",
    f"{project_name}/pipeline/predict_pipeline.py",

    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/common.py",

    "data/raw/.gitkeep",
    "data/processed/.gitkeep",

    "artifacts/models/.gitkeep",
    "artifacts/preprocessors/.gitkeep",
    "artifacts/reports/.gitkeep",
    "artifacts/transformed_data/.gitkeep",

    "api/main.py",

    "monitoring/drift_report.py",
    "monitoring/reports/.gitkeep",

    "notebooks/eda.ipynb",
    "notebooks/exp_notebook.ipynb",

    "config/config.yaml",

    ".gitignore",
    ".env",
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
]


for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        print(f"Creating directory: {filedir}")

    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()
        print(f"Creating file: {filepath}")

    else:
        print(f"File already exists: {filepath}")