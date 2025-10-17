# Extensions from the the original homework
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import Logit, Probit
from statsmodels.iolib.summary2 import summary_col

# === compute Nan with nearest neighbor and then to the regressions with random split ===
# ===Part 1: Impute missing data===
df_full = pl.read_csv(
    "./defaut2000.csv", use_pyarrow=True, separator=";", null_values=["-99.99"]
).to_pandas()


target_col = "yd"
assert target_col in df_full.columns, "Target 'yd' not found in dataframe."

X_full = df_full.drop(columns=[target_col])
y_full = df_full[target_col]

mask_missing = X_full.isna()

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_full), columns=X_full.columns, index=X_full.index
)

# Apply KNN imputation on scaled features
imputer = KNNImputer(n_neighbors=3)
X_imputed_scaled = imputer.fit_transform(X_scaled)

# Back to original scale
X_imputed = pd.DataFrame(
    scaler.inverse_transform(X_imputed_scaled),
    columns=X_full.columns,
    index=X_full.index,
)

# Replace only original NaNs in features
X_final = X_full.copy()
X_final[mask_missing] = X_imputed[mask_missing]

# Recompose full DataFrame (y unchanged)
df_final = pd.concat([X_final, y_full], axis=1)

df_pl = pl.from_pandas(df_final).with_row_index("id")

print("Missing values before (features):", X_full.isna().sum().sum())
print("Missing values after  (features):", X_final.isna().sum().sum())

# === PART 2: Train/test split (random) ===
VAR = ["tdta", "gempl", "opita", "invsls"]

feature_cols = [c for c in df_pl.columns if c not in ("id", target_col)]

core_features = [c for c in VAR if c in feature_cols]
if len(core_features) == 0:
    raise ValueError(f"None of the core features {VAR} found in dataframe.")

X_all = df_pl.select(feature_cols)
y_all = df_pl.select(["id", target_col])


X_train_pl, X_test_pl, y_train_pl, y_test_pl = train_test_split(
    X_all, y_all, test_size=0.25, random_state=42
)


y_train = y_train_pl[target_col].to_pandas()
X_train = sm.add_constant(X_train_pl[core_features].to_pandas())


train_id = y_train_pl["id"].to_list()

# === PART 3: Fit models and predict on TRAIN ===
# OLS
model_ols = sm.OLS(y_train, X_train).fit()

# Logit
model_log = Logit(y_train, X_train).fit()

# Probit
model_prob = Probit(y_train, X_train).fit()

# Predictions (aligned to train rows)
p_ols = model_ols.predict(X_train)  # continuous
p_logit = model_log.predict(X_train)  # probabilities
p_probit = model_prob.predict(X_train)  # probabilities

# === PART 4: Build aligned score table ===
preds_pl = pl.DataFrame(
    {
        "id": train_id,
        "score_ols": p_ols,
        "score_logit": p_logit,
        "score_probit": p_probit,
    }
)

df_train_view = df_pl.filter(pl.col("id").is_in(train_id)).select(
    ["id", target_col] + [c for c in core_features if c in df_pl.columns]
)

df_score = df_train_view.join(preds_pl, on="id", how="left").to_pandas()


# === PART 5: Metrics (AUC + R²) ===
def safe_auc(y_true, scores):
    """Compute ROC AUC with a safety net."""
    try:
        return roc_auc_score(np.asarray(y_true), np.asarray(scores))
    except Exception:
        return np.nan


auc_ols = safe_auc(y_train, p_ols)
auc_logit = safe_auc(y_train, p_logit)
auc_probit = safe_auc(y_train, p_probit)

# === Combined results table ===
results_table = summary_col(
    [model_ols, model_log, model_prob],
    stars=True,
    model_names=["OLS", "Logit", "Probit"],
    info_dict={
        "N": lambda m: f"{int(m.nobs)}",
        "R²": lambda m: (
            f"{m.rsquared:.3f}"
            if hasattr(m, "rsquared")
            else f"{m.prsquared:.3f}"
            if hasattr(m, "prsquared")
            else "—"
        ),
        "AUC": lambda m: (
            f"{auc_ols:.3f}"
            if m is model_ols
            else f"{auc_logit:.3f}"
            if m is model_log
            else f"{auc_probit:.3f}"
        ),
    },
)

tbl = results_table.tables[0]
tbl.drop(
    index=[i for i in tbl.index if "R-squared" in i],
    inplace=True,
    errors="ignore",
)

# === Display ===
print(results_table)
print("\n=== OLS ===\n", model_ols.summary())
print("\n=== Logit ===\n", model_log.summary())
print("\n=== Probit ===\n", model_prob.summary())
