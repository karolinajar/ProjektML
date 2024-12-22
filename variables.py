
PATH_TO_FILE = 'datasets/stroke.csv'
RESULTS_PATH = 'results/'

# Parametry przetwarzania danych
THRESHOLD_CORRELATION = 0.9
THRESHOLD_VARIANCE = 0.01

# Parametry modelowania
TEST_SIZE = 0.2
RANDOM_STATE = 42
PCA_COMPONENTS = 0.95
MAX_PCA_COMPONENTS = 5
BALANCE_METHOD = "smote"
MODEL_TYPE = "xgboost"


COLUMNS_TO_DROP = ['id']
TARGET_COLUMN = 'stroke'

# Inżynieria danych
CREATE_NEW_FEATURES = True

# Optymalizacja modelu
USE_GRID_SEARCH = True

# Ustawienia wyników
SAVE_MODEL = True
PLOT_FEATURE_IMPORTANCE = True

# Interpretacja wyników
PLOT_FEATURE_IMPORTANCE = True
INTERPRET_RESULTS = True
