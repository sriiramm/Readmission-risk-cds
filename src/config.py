# Target definition for UCI Diabetes dataset
TARGET_COL = "readmitted"
POSITIVE_LABEL = "<30"

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42

# Target definition: 1 if readmitted within 30 days
TARGET_COL = "readmitted"
POSITIVE_LABEL = "<30"

# Remove identifiers (HIPAA-style)
DROP_COLS = ["encounter_id", "patient_nbr"]

# Sensitive columns (for fairness audit / reporting)
SENSITIVE_COLS = ["race", "gender", "age"]
