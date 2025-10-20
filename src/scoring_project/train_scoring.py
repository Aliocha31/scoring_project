import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# ======== CONFIG =========
# =========================
CSV_PATH = "./defaut2000.csv"
TARGET = "yd"
RANDOM_STATE = 42

FEATURES: List[str] = [
    "ebita",
    "opita",
    "reta",
    "gempl",
]


# -----------------------------------------------------------
# 0) Wrappers for statsmodels Probit & OLS (sklearn API)
# -----------------------------------------------------------
class ProbitClassifier(BaseEstimator, ClassifierMixin):
    """Statsmodels Probit wrapped for sklearn API."""

    def __init__(self, max_iter: int = 200):
        self.max_iter = max_iter
        self.model_ = None

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        X = np.asarray(X)
        Xc = sm.add_constant(X, has_constant="add")
        self.model_ = sm.Probit(y, Xc).fit(disp=0, maxiter=self.max_iter)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        Xc = sm.add_constant(X, has_constant="add")
        p = np.asarray(self.model_.predict(Xc)).reshape(-1)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class OLSClassifier(BaseEstimator, ClassifierMixin):
    """
    OLS on binary y; clip predictions to [0,1] and treat as probabilities.
    Useful as a simple linear baseline.
    """

    def __init__(self):
        self.model_ = None

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        X = np.asarray(X)
        Xc = sm.add_constant(X, has_constant="add")
        self.model_ = sm.OLS(y, Xc).fit()
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        Xc = sm.add_constant(X, has_constant="add")
        p = np.asarray(self.model_.predict(Xc)).reshape(-1)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# -----------------------------------------------------------
# 1) Load & split
# -----------------------------------------------------------
def load_data(csv_path: str = CSV_PATH) -> pl.DataFrame:
    df = pl.read_csv(
        csv_path, use_pyarrow=True, separator=";", null_values=["-99.99"]
    ).fill_null(np.nan)
    # Keep your historical ordering
    sort_cols = [c for c in ["yd", "reta"] if c in df.columns]
    if sort_cols:
        df = df.sort(sort_cols)
    # Even/odd split flag
    df = df.with_row_index("id").with_columns(
        ((pl.col("id") % 2 == 0).cast(pl.Int8)).alias("dumVE")
    )
    return df


# -----------------------------------------------------------
# 2) Feature selection
# -----------------------------------------------------------
def select_features_polars(df_pl: pl.DataFrame, requested: List[str]) -> List[str]:
    cols = df_pl.columns
    base_exclude = {TARGET, "dumVE", "id"}
    if not requested:
        feats = [c for c in cols if c not in base_exclude]
    else:
        feats = [c for c in requested if c in cols]
    if len(feats) == 0:
        raise ValueError(
            f"No valid features found. Requested: {requested} | Available: {cols}"
        )
    return feats


# -----------------------------------------------------------
# 3) Convert to pandas & check target
# -----------------------------------------------------------
def to_pandas_and_check_target(df_pl: pl.DataFrame) -> pd.DataFrame:
    df_pd = df_pl.to_pandas()
    if TARGET not in df_pd.columns:
        raise ValueError(f"Target '{TARGET}' not found.")
    uniq = set(pd.unique(df_pd[TARGET].dropna()))
    if not uniq.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Target '{TARGET}' must be binary {{0,1}}. Found: {uniq}")
    df_pd[TARGET] = df_pd[TARGET].astype(int)
    return df_pd


# -----------------------------------------------------------
# 4) KNN imputation that ONLY fills missing entries
# -----------------------------------------------------------
def knn_impute_only_missing(
    df_pd: pd.DataFrame,
    features: List[str],
    n_neighbors: int = 3,
    scale: bool = True,
) -> pd.DataFrame:
    """
    Impute only missing entries in 'features' using KNN.
    Observed (non-missing) values are left EXACTLY as they were.
    """
    df_out = df_pd.copy()
    X = df_pd[features].copy()

    mask_missing = X.isna()

    # Scale for distance computation (recommended)
    if scale:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), index=X.index, columns=X.columns
        )
    else:
        X_scaled = X

    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_imputed_scaled = imputer.fit_transform(X_scaled)

    if scale:
        X_imputed = pd.DataFrame(
            scaler.inverse_transform(X_imputed_scaled), index=X.index, columns=X.columns
        )
    else:
        X_imputed = pd.DataFrame(X_imputed_scaled, index=X.index, columns=X.columns)

    # Write back ONLY where values were missing
    X_filled = X.copy()
    X_filled[mask_missing] = X_imputed[mask_missing]

    # Replace in original df
    df_out[features] = X_filled

    # Optional safety check:
    # assert np.allclose(X_filled[~mask_missing], X[~mask_missing], equal_nan=True)

    return df_out


# -----------------------------------------------------------
# 5) Build splits
# -----------------------------------------------------------
def build_train_valid(df_pd: pd.DataFrame, features: List[str]):
    """Deterministic even/odd split via dumVE."""
    train = df_pd[df_pd["dumVE"] == 0].copy()
    valid = df_pd[df_pd["dumVE"] == 1].copy()
    X_train, y_train = train[features], train[TARGET]
    X_valid, y_valid = valid[features], valid[TARGET]
    return features, X_train, y_train, X_valid, y_valid


def build_random_split(
    df_pd: pd.DataFrame, features: List[str], test_size: float = 0.25
):
    """Stratified random split for a fair comparison."""
    X = df_pd[features]
    y = df_pd[TARGET]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    return features, X_train, y_train, X_valid, y_valid


# -----------------------------------------------------------
# 6) Preprocessing and models
# -----------------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Imputation has already been done globally → only scale + onehot
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipe = Pipeline([("scaler", StandardScaler())])

    # Dense output for universal compatibility (HGB, statsmodels, etc.)
    try:
        cat_pipe = Pipeline(
            [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        )
    except TypeError:
        cat_pipe = Pipeline(
            [("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))]
        )

    preproc = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preproc


def get_models_and_grids() -> Dict[str, Tuple[object, Dict[str, List]]]:
    """
    All models expose a usable score for AUC:
      - logreg, probit, ols -> predict_proba (or proba-like)
      - ridge, rf, hgb -> decision_function or predict_proba (handled in _scores_for_auc_flexible)
    """
    return {
        "logreg": (
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            {"model__C": [0.1, 1.0, 3.0], "model__solver": ["lbfgs", "saga"]},
        ),
        "probit": (ProbitClassifier(max_iter=200), {}),
        "ols": (OLSClassifier(), {}),
        "ridge": (
            RidgeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"model__alpha": [0.5, 1.0, 3.0, 10.0]},
        ),
        "rf": (
            RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
            ),
            {
                "model__n_estimators": [300, 600],
                "model__max_depth": [5, 8, 10],
                "model__min_samples_leaf": [2, 4],
            },
        ),
        "hgb": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5, 8],
                "model__l2_regularization": [0.1, 0.3],
            },
        ),
    }


# -----------------------------------------------------------
# 7) Evaluation utilities (robust CV AUC)
# -----------------------------------------------------------
def _scores_for_auc_flexible(estimator, X):
    """Return a continuous score for AUC, whatever the estimator exposes."""
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        if isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        elif isinstance(p, np.ndarray) and p.ndim == 1:
            return p
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    # Fallback (less ideal) : raw predictions
    pred = estimator.predict(X)
    return np.asarray(pred).reshape(-1)


def cross_val_auc_safe(pipeline, X, y, cv, random_state=42):
    """
    Robust CV AUC for all models:
    - manual KFolds
    - fit on train fold
    - get continuous score on validation fold (proba/decision/predict)
    - average AUC across folds
    """
    aucs = []
    # standardize indices
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Ensure a StratifiedKFold object
    if isinstance(cv, StratifiedKFold):
        skf = cv
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        est = clone(pipeline)
        est.fit(X_tr, y_tr)

        s_va = _scores_for_auc_flexible(est, X_va)
        try:
            auc = roc_auc_score(y_va, s_va)
            aucs.append(auc)
        except Exception:
            # skip if scoring failed on this fold
            continue

    if len(aucs) == 0:
        return np.nan
    return float(np.mean(aucs))


def evaluate_auc(estimator, X_train, y_train, X_test, y_test):
    s_tr = _scores_for_auc_flexible(estimator, X_train)
    s_te = _scores_for_auc_flexible(estimator, X_test)
    auc_train = roc_auc_score(y_train, s_tr)
    auc_test = roc_auc_score(y_test, s_te)
    return auc_train, auc_test


def compare_models(X_train, y_train, X_valid, y_valid):
    preproc = build_preprocessor(X_train)
    models = get_models_and_grids()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    leaderboard = []

    for name, (estimator, grid) in models.items():
        pipe = Pipeline([("preprocessor", preproc), ("model", estimator)])

        # 1) GridSearch pour hyperparamètres s'il y en a, sinon fit direct
        has_grid = (grid is not None) and (len(grid) > 0)
        if has_grid:
            gs = GridSearchCV(
                pipe, param_grid=grid, cv=cv, scoring=None, n_jobs=-1, verbose=0
            )
            gs.fit(X_train, y_train)
            best_est = gs.best_estimator_
            best_params = gs.best_params_
        else:
            best_est = pipe.fit(X_train, y_train)
            best_params = {}

        # 2) CV AUC robuste (manuel) pour TOUS les modèles
        cv_auc = cross_val_auc_safe(best_est, X_train, y_train, cv, RANDOM_STATE)

        # 3) AUC train/test
        auc_train, auc_valid = evaluate_auc(
            best_est, X_train, y_train, X_valid, y_valid
        )

        leaderboard.append(
            {
                "model": name,
                "cv_auc": cv_auc,
                "train_auc": float(auc_train),
                "test_auc": float(auc_valid),
                "best_params": best_params,
            }
        )

    return (
        pd.DataFrame(leaderboard)
        .sort_values("test_auc", ascending=False)
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------
    # ROC curves for all models on a given split
    # -----------------------------------------------------------


def fit_models_and_rocs(X_train, y_train, X_valid, y_valid, random_state=RANDOM_STATE):
    """
    Re-fit each model (avec meilleure config si grille) sur X_train/y_train,
    calcule les courbes ROC sur X_valid/y_valid et renvoie les données.
    """
    preproc = build_preprocessor(X_train)
    models = get_models_and_grids()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    curves = []  # list of dicts: {name, fpr, tpr, auc, estimator}
    for name, (estimator, grid) in models.items():
        pipe = Pipeline([("preprocessor", preproc), ("model", estimator)])
        # GridSearch si grille, sinon fit direct
        has_grid = (grid is not None) and (len(grid) > 0)
        if has_grid:
            gs = GridSearchCV(
                pipe, param_grid=grid, cv=cv, scoring=None, n_jobs=-1, verbose=0
            )
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
        else:
            best = pipe.fit(X_train, y_train)
        # Score "continu" flexible pour ROC/AUC
        s_valid = _scores_for_auc_flexible(best, X_valid)
        fpr, tpr, _ = roc_curve(y_valid, s_valid)
        auc_val = auc(fpr, tpr)
        curves.append(
            {
                "name": name,
                "fpr": fpr,
                "tpr": tpr,
                "auc": float(auc_val),
                "estimator": best,
            }
        )
    # trier par AUC décroissante pour une légende lisible
    curves.sort(key=lambda d: d["auc"], reverse=True)

    return curves


def plot_rocs(curves, title="ROC curves"):
    """Affiche la courbe ROC de chaque modèle + la diagonale aléatoire."""
    plt.figure(figsize=(8, 6))
    for c in curves:
        plt.plot(c["fpr"], c["tpr"], label=f"{c['name']} (AUC = {c['auc']:.3f})")
    # Classif aléatoire
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# 8) Main pipeline
# -----------------------------------------------------------
if __name__ == "__main__":
    # Load
    pl_df = load_data(CSV_PATH)
    features = select_features_polars(pl_df, FEATURES)
    df_pd = to_pandas_and_check_target(pl_df)

    # KNN impute ONLY missing entries on selected features
    df_pd = knn_impute_only_missing(df_pd, features, n_neighbors=3, scale=True)

    print(f"\nUsing features: {features}")

    # --- Deterministic Split (even/odd) ---
    print("\n=== Deterministic Split (dumVE) ===")
    _, X_tr_det, y_tr_det, X_va_det, y_va_det = build_train_valid(df_pd, features)
    leaderboard_det = compare_models(X_tr_det, y_tr_det, X_va_det, y_va_det)
    print(leaderboard_det)

    # --- Random Split (stratified) ---
    print("\n=== Random Split ===")
    _, X_tr_rand, y_tr_rand, X_va_rand, y_va_rand = build_random_split(df_pd, features)
    leaderboard_rand = compare_models(X_tr_rand, y_tr_rand, X_va_rand, y_va_rand)
    print(leaderboard_rand)

    print(X_tr_rand.describe())
    print(X_va_rand.describe())

    # --- Equality of means and variance ---
    variables = ["ebita", "opita", "reta", "gempl"]
    results = []
    results_var = []
    for var in variables:
        t_stat, p_val = stats.ttest_ind(
            X_tr_rand[var],
            X_va_rand[var],
            equal_var=False,
        )
        results.append({"Variable": var, "t-statistic": t_stat, "p-value": p_val})

    df_ttest = pd.DataFrame(results)
    print(df_ttest)

    for var in variables:
        stat, p_val = stats.levene(X_tr_rand[var], X_va_rand[var], center="median")
        results_var.append(
            {"Variable": var, "Levene statistic": stat, "p-value": p_val}
        )

    df_var = pd.DataFrame(results_var)
    print(df_var)

    # --- Summary comparison (best rows) ---
    summary = pd.DataFrame(
        {
            "Split": ["Deterministic", "Random"],
            "Best Model": [
                leaderboard_det.iloc[0]["model"],
                leaderboard_rand.iloc[0]["model"],
            ],
            "Train AUC": [
                leaderboard_det.iloc[0]["train_auc"],
                leaderboard_rand.iloc[0]["train_auc"],
            ],
            "Test AUC": [
                leaderboard_det.iloc[0]["test_auc"],
                leaderboard_rand.iloc[0]["test_auc"],
            ],
        }
    ).set_index("Split")

    print("\n=== AUC Comparison: Deterministic vs Random Split ===")
    print(summary)

    curves_det = fit_models_and_rocs(X_tr_det, y_tr_det, X_va_det, y_va_det)
    plot_rocs(curves_det, title="ROC curves — Deterministic split (dumVE)")

    curves_rand = fit_models_and_rocs(X_tr_rand, y_tr_rand, X_va_rand, y_va_rand)
    plot_rocs(curves_rand, title="ROC curves — Random stratified split")
