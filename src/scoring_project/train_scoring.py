# -----------------------------------------------------------
# Binary scoring (yd: 0 = non-default, 1 = default) with model comparison
#
# Script goals:
#  - Build a robust credit scoring model to predict default probability.
#  - Compare multiple classifiers under a consistent preprocessing pipeline.
#  - Select the best model via ROC AUC on a hold-out validation set.
#  - Provide a leaderboard with rich metrics useful for imbalanced datasets.
#
# Methodological choices:
#  - Preprocessing = imputation + standardization (numerical) + one-hot (categorical),
#    encapsulated in a Pipeline → prevents data leakage.
#  - GridSearchCV with StratifiedKFold(5), scoring="roc_auc":
#    ROC AUC is threshold-independent and robust to imbalance.
#  - Validation on a hold-out set (dumVE=1) distinct from Estimation (dumVE=0).
#  - Metrics:
#    * ROC AUC: global ranking quality.
#    * PR AUC: emphasizes the minority (default) class.
#    * F1 & Accuracy: decision-level metrics at fixed/optimized thresholds.
#    * Confusion matrix: operational view of errors.
#  - Thresholds: report both fixed 0.5 and optimized-for-F1,
#    since the optimal threshold is business-context dependent.
# -----------------------------------------------------------

import os
import warnings
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# ======== CONFIG =========
# =========================
CSV_PATH = "./defaut2000.csv"  
TARGET = "yd"  # Binary target: 0 = non-default, 1 = default
RANDOM_STATE = 42

# Feature list:
# - Leave [] → use all columns except ["yd","dumVE","id"].
# - Otherwise, define explicitly (example: ["ebita","opita","reta"]).
FEATURES: List[str] = [
    "ebita",
    "opita",
    "reta",
    "gempl",
]


# -----------------------
# 1) Load & split
# -----------------------
def load_data(csv_path: str = CSV_PATH) -> pl.DataFrame:
    """
    Load the CSV using Polars and create dumVE.
    Steps & rationale:
      - separator=";" : matches your dataset format.
      - null_values=["-99.99"] : standardizes this code into NaN.
      - fill_null(np.nan): ensures compatibility with scikit-learn later.
      - sort(['yd','reta']) if present: kept from your snippet (ensures stable ordering).
      - drop_nans(): removes entirely empty rows.
      - dumVE from even/odd row index:
          even   → dumVE=0 (Estimation set)
          odd    → dumVE=1 (Validation set)
        → simple deterministic train/validation split.
    """
    df = pl.read_csv(
        csv_path, use_pyarrow=True, separator=";", null_values=["-99.99"]
    ).fill_null(np.nan)

    sort_cols = [c for c in ["yd", "reta"] if c in df.columns]
    if sort_cols:
        df = df.sort(sort_cols)
        
    df = df.with_row_index("id").with_columns(
        ((pl.col("id") % 2 == 0).cast(pl.Int8)).alias("dumVE")
    )
    return df


# -----------------------
# 2) Feature selection 
# -----------------------
def select_features_polars(df_pl: pl.DataFrame, requested: List[str]) -> List[str]:
    """
    Determine the final list of explanatory variables from the Polars schema.

    Rules:
      - If 'requested' is empty → use all columns except {TARGET, 'dumVE', 'id'}.
      - If 'requested' is non-empty → strict intersection with DataFrame columns.

    Returns: list of feature names to use.
    """
    cols = df_pl.columns
    base_exclude = {TARGET, "dumVE", "id"}

    if not requested:
        feats = [c for c in cols if c not in base_exclude]
    else:
        feats = [c for c in requested if c in cols]

    if len(feats) == 0:
        raise ValueError(
            f"No valid features found. Requested: {requested} | "
            f"Available columns: {cols}"
        )
    return feats


# -----------------------
# 3) Convert to pandas & check target
# -----------------------
def to_pandas_and_check_target(df_pl: pl.DataFrame) -> pd.DataFrame:
    """
    Convert Polars → pandas and validate the binary target.
    - Ensures TARGET exists.
    - Ensures values are in {0,1}.
    - Casts to int (avoid float 0.0/1.0).
    """
    df_pd = df_pl.to_pandas()
    if TARGET not in df_pd.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")
    uniq = set(pd.unique(df_pd[TARGET].dropna()))
    if not uniq.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Target '{TARGET}' must be binary {{0,1}}. Found: {uniq}")
    df_pd[TARGET] = df_pd[TARGET].astype(int)
    return df_pd


# -----------------------
# 4) Build train/validation sets
# -----------------------
def build_train_valid(df_pd: pd.DataFrame, features: List[str]):
    """
    Construct Estimation and Validation sets:
      - Estimation = dumVE==0 ; Validation = dumVE==1
      - Drop rows with missing target
      - Restrict X to the chosen features
    Rationale: hold-out Validation (dumVE=1) is never used in CV → honest generalization check.
    """
    train = df_pd[df_pd["dumVE"] == 0].copy()
    valid = df_pd[df_pd["dumVE"] == 1].copy()

    train = train.dropna(subset=[TARGET])
    valid = valid.dropna(subset=[TARGET])

    X_train, y_train = train[features], train[TARGET].astype(int)
    X_valid, y_valid = valid[features], valid[TARGET].astype(int)
    return features, X_train, y_train, X_valid, y_valid


# -----------------------
# 5) Preprocessing (scikit-learn)
# -----------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Standard preprocessing:
      - Numeric (+ boolean) → median imputation + StandardScaler
        * Median imputation: robust to outliers.
        * StandardScaler: important for linear models (LogReg) and gradient boosting.
      - Categorical → mode imputation + OneHotEncoder(handle_unknown='ignore')
        * 'ignore' makes it robust to unseen categories in production.
    Encapsulated in ColumnTransformer for clean pipelines and no leakage.
    """
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preproc


# -----------------------
# 6) Models & grids (GridSearchCV)
# -----------------------
def get_models_and_grids() -> Dict[str, Tuple[object, Dict[str, List]]]:
    """
    Models considered:
      - LogisticRegression(class_weight='balanced'): linear baseline, interpretable, handles imbalance.
      - RandomForestClassifier(class_weight='balanced'): non-linear, robust, good for tabular data.
      - GradientBoostingClassifier: classic boosting, strong for subtle patterns.
      - HistGradientBoostingClassifier: efficient boosting with histogram splits.

    Grids: small but meaningful hyperparameter grids
    → avoids exploding compute time, but explores key knobs (depth, n_estimators, learning_rate).
    """
    return {
        "logreg": (
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            {
                "model__C": [0.1, 1.0, 3.0],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs", "saga"],
            },
        ),
        "rf": (
            RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
            ),
            {
                "model__n_estimators": [300, 600],
                "model__max_depth": [None, 12],
                "model__min_samples_leaf": [1, 2],
            },
        ),
        "gboost": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
        "hgb": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [None, 8],
                "model__l2_regularization": [0.0, 0.1],
            },
        ),
    }


# -----------------------
# 7) Metrics & thresholds
# -----------------------
def evaluate_probs(y_true: np.ndarray, p1: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute metrics from predicted probabilities (class 1 = default).
    Why these metrics?
      - ROC AUC: global ranking quality, threshold-free.
      - PR AUC: emphasizes performance on minority class.
      - F1 & Accuracy @ threshold: decision metrics for a given cutoff.
      - Confusion matrix: operational view of FP/FN trade-offs.
    """
    y_pred = (p1 >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, p1)),
        "pr_auc": float(average_precision_score(y_true, p1)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def find_best_threshold(y_true: np.ndarray, p1: np.ndarray) -> float:
    """
    Find the probability threshold that maximizes F1 on the Validation set.
    Why F1? Balanced measure when FP/FN costs are not specified.
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (p1 >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


# -----------------------
# 8) Training, comparison & saving
# -----------------------
def compare_models(
    X_train, y_train, X_valid, y_valid, save_best_to: str = "best_classifier.joblib"
):
    """
    Full evaluation pipeline:
      1) Build preprocessing on X_train (avoid leakage).
      2) For each model:
         - GridSearchCV (5-fold StratifiedKFold, scoring='roc_auc') → 'cv_roc_auc'.
         - Predict probabilities on Validation.
         - Compute metrics at threshold 0.5 and at best-F1 threshold.
         - Save a row into 'leaderboard'.
      3) Sort leaderboard by Validation ROC AUC (main criterion).
      4) Save best pipeline (preprocessing + model) for reuse/deployment.
    """
    preproc = build_preprocessor(X_train)
    models = get_models_and_grids()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    leaderboard = []
    best_val_auc = -np.inf
    best_name, best_pipe = None, None

    for name, (estimator, grid) in models.items():
        pipe = Pipeline(
            [
                ("preprocessor", preproc),
                ("model", estimator),
            ]
        )

        gs = GridSearchCV(
            pipe,
            param_grid=grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)

        # Get predicted probabilities for Validation
        p1_valid = (
            gs.best_estimator_.predict_proba(X_valid)[:, 1]
            if hasattr(gs.best_estimator_["model"], "predict_proba")
            else gs.best_estimator_.decision_function(X_valid)
        )

        metrics_05 = evaluate_probs(y_valid, p1_valid, threshold=0.5)
        t_best = find_best_threshold(y_valid, p1_valid)
        metrics_t = evaluate_probs(y_valid, p1_valid, threshold=t_best)

        # === Leaderboard row ===
        leaderboard.append(
            {
                "model": name,  # Model short name
                "cv_roc_auc": float(
                    gs.best_score_
                ),  # Cross-validated ROC AUC on Estimation (selection score)
                "val_roc_auc": metrics_05[
                    "roc_auc"
                ],  # ROC AUC on Validation (hold-out, threshold-free)
                "val_pr_auc": metrics_05[
                    "pr_auc"
                ],  # Precision-Recall AUC on Validation
                "val_f1@0.5": metrics_05["f1"],  # F1 score at fixed threshold 0.5
                "val_acc@0.5": metrics_05["accuracy"],  # Accuracy at threshold 0.5
                "best_threshold_val": t_best,  # Threshold maximizing F1 on Validation
                "val_f1@best_t": metrics_t["f1"],  # F1 at that optimized threshold
                "best_params": gs.best_params_,  # Hyperparameters chosen by GridSearchCV
                "cm@best_t": metrics_t[
                    "confusion_matrix"
                ],  # Confusion matrix @ best_t: [[TN,FP],[FN,TP]]
            }
        )

        if metrics_05["roc_auc"] > best_val_auc:
            best_val_auc = metrics_05["roc_auc"]
            best_name = name
            best_pipe = gs.best_estimator_

    lb_df = (
        pd.DataFrame(leaderboard)
        .sort_values("val_roc_auc", ascending=False)
        .reset_index(drop=True)
    )

    if best_pipe is not None:
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_pipe, os.path.join("models", save_best_to))

    return lb_df, best_name, best_val_auc


# -----------------------
# 9) Main
# -----------------------
if __name__ == "__main__":
    # (a) Load data with Polars
    pl_df = load_data(CSV_PATH)

    # (b) Select features (auto if FEATURES=[])
    features = select_features_polars(pl_df, FEATURES)
    print(f"\nUsing features ({len(features)}): {features}")

    # (c) Convert to pandas & check target
    df_pd = to_pandas_and_check_target(pl_df)

    # (d) Build Estimation/Validation sets
    features, X_tr, y_tr, X_va, y_va = build_train_valid(df_pd, features)

    # (e) Compare models + save best pipeline
    leaderboard, best_name, best_auc = compare_models(X_tr, y_tr, X_va, y_va)

    # (f) Print leaderboard
    pd.set_option("display.max_colwidth", 200)
    print("\n=== Leaderboard (sorted by Validation ROC AUC) ===")
    cols_show = [
        "model",
        "cv_roc_auc",
        "val_roc_auc",
        "val_pr_auc",
        "val_f1@0.5",
        "val_acc@0.5",
        "best_threshold_val",
        "val_f1@best_t",
        "best_params",
        "cm@best_t",
    ]
    print(leaderboard[cols_show])
    print(f"\nBest model (Validation AUC): {best_name} (AUC={best_auc:.4f})")
    print("Best pipeline saved in ./models/best_classifier.joblib")

# %%
