# __init__.py for the src package
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Suppress warnings in production
warnings.filterwarnings('ignore')

# Optionally, you might want to initialize logging here if your application uses it
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# You could also include utility functions here that are used across various modules
def print_info(data):
    """Print basic information about a DataFrame."""
    print(data.info())
    print(data.head())

# Ensure that this file also manages imports so modules can use them directly
__all__ = [
    "pd",
    "plt",
    "train_test_split",
    "StandardScaler",
    "OneHotEncoder",
    "LabelEncoder",
    "ColumnTransformer",
    "Pipeline",
    "SimpleImputer",
    "LogisticRegression",
    "classification_report",
    "confusion_matrix",
    "accuracy_score",
    "SARIMAX",
    "warnings",
    "logging",
    "print_info"
]
