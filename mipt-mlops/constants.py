from pathlib import Path


package_dir = Path(__file__).resolve().parent
repo_dir = package_dir.parent
models_dir = repo_dir / "models"
data_dir = repo_dir / "data"
predictions_dir = repo_dir / "predictions"

BATCH_SIZE_DEFAULT = 64
