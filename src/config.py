"""
Central configuration: filesystem paths, expected raw-file schema, and
modelling constants. Every other module imports from here so that paths
are defined exactly once.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
SQL_DIR = PROJECT_ROOT / "sql"

for _d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR, FIGURE_DIR):
    if _d.exists() and not _d.is_dir():
        raise RuntimeError(f"Expected directory but found a file: {_d}")
    _d.mkdir(parents=True, exist_ok=True)

# Key artefacts
MASTER_PARQUET = PROCESSED_DIR / "master_dataset.parquet"
MASTER_CSV = PROCESSED_DIR / "master_dataset.csv"
CLEANING_LOG = REPORT_DIR / "cleaning_log.md"
EDA_REPORT = REPORT_DIR / "eda_report.md"
SQLITE_DB = PROCESSED_DIR / "tourism.db"

REG_MODEL_PATH = MODEL_DIR / "rating_regressor.joblib"
CLF_MODEL_PATH = MODEL_DIR / "visitmode_classifier.joblib"
RECO_MODEL_PATH = MODEL_DIR / "recommender.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# --------------------------------------------------------------------------
# Raw table schema
#
# The source workbook ships one sheet/file per table. Filenames vary between
# copies of the dataset (Transaction.xlsx / transaction.csv / Transaction
# data.xlsx ...), so the loader resolves each logical table by matching the
# aliases below case-insensitively against whatever is in data/raw.
# --------------------------------------------------------------------------
RAW_TABLES: dict[str, dict] = {
    "transaction": {
        "aliases": ["transaction", "transactions", "transaction data", "txn"],
        "columns": [
            "TransactionId", "UserId", "VisitYear", "VisitMonth",
            "VisitMode", "AttractionId", "Rating",
        ],
        "key": "TransactionId",
    },
    "user": {
        "aliases": ["user", "users", "user data"],
        "columns": ["UserId", "ContinentId", "RegionId", "CountryId", "CityId"],
        "key": "UserId",
    },
    "city": {
        "aliases": ["city", "cities", "city data"],
        "columns": ["CityId", "CityName", "CountryId"],
        "key": "CityId",
    },
    "attraction_type": {
        "aliases": ["type", "attraction type", "attractiontype", "types"],
        "columns": ["AttractionTypeId", "AttractionType"],
        "key": "AttractionTypeId",
    },
    "visit_mode": {
        "aliases": ["mode", "visit mode", "visitmode", "mode data"],
        "columns": ["VisitModeId", "VisitMode"],
        "key": "VisitModeId",
    },
    "continent": {
        "aliases": ["continent", "continents", "continent data"],
        "columns": ["ContinentId", "Continent"],
        "key": "ContinentId",
    },
    "country": {
        "aliases": ["country", "countries", "country data"],
        "columns": ["CountryId", "Country", "RegionId"],
        "key": "CountryId",
    },
    "region": {
        "aliases": ["region", "regions", "region data"],
        "columns": ["RegionId", "Region", "ContinentId"],
        "key": "RegionId",
    },
    "item": {
        "aliases": ["item", "items", "attraction", "attractions", "item data"],
        "columns": [
            "AttractionId", "AttractionCityId", "AttractionTypeId",
            "Attraction", "AttractionAddress",
        ],
        "key": "AttractionId",
    },
}

# --------------------------------------------------------------------------
# Modelling constants
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

RATING_MIN, RATING_MAX = 1, 5

# Target names
REG_TARGET = "Rating"
CLF_TARGET = "VisitMode"

# Feature groups shared by the regression and classification pipelines.
# `Rating` is deliberately absent from both -- it is the regression target and
# leaks directly into the classifier if included naively.
CATEGORICAL_FEATURES = [
    "Continent", "Region", "Country", "CityName",
    "AttractionType", "AttractionCity", "Attraction",
    "Season",
]

NUMERIC_FEATURES = [
    "VisitYear", "VisitMonth",
    "UserVisitCount", "UserAvgRating", "UserRatingStd",
    "AttractionVisitCount", "AttractionAvgRating", "AttractionRatingStd",
    "CountryAvgRating", "AttractionTypeAvgRating",
    "MonthsSinceFirstVisit", "UserAttractionTypeAffinity",
]

# Extra feature used only by the classifier (rating is known at the time a
# visit mode is inferred retrospectively, e.g. for segmenting past visitors).
CLF_EXTRA_FEATURES = ["Rating"]

# Minimum interactions for a user/attraction to enter the collaborative model.
MIN_USER_INTERACTIONS = 2
MIN_ITEM_INTERACTIONS = 2

TOP_N_RECOMMENDATIONS = 10
