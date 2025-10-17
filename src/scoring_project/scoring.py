import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import kurtosis, skew
from sklearn.metrics import auc, roc_auc_score, roc_curve
from statsmodels.discrete.discrete_model import Logit, Probit
from statsmodels.iolib.summary2 import summary_col

# convert to latex : True to createnew file, False to skip this step
convert_to_latex = True

if convert_to_latex == False:
    warnings.warn(
        "Latex conversion skipped. Set convert_to_latex = True to enable it.",
        UserWarning,
    )
else:
    warnings.warn(
        "Latex conversion not skipped. Set convert_to_latex = False to stop it.",
        UserWarning,
    )


df_sco = (
    pl.read_csv(
        "./defaut2000.csv", use_pyarrow=True, separator=";", null_values=["-99.99"]
    ).fill_null(np.nan)
).drop_nans()

df_sort = df_sco.sort(("yd", "reta"))

# dumVE = 0 => odd rows => Estimation
# dumVE = 1 => even rows => Validation
df_dum = df_sort.with_row_index("id").with_columns(
    ((pl.col("id") % 2 == 0).cast(pl.Int8)).alias("dumVE")
)

target = ["yd"]

# == features ==
X_train = df_dum.filter(pl.col("dumVE") == 0).select(
    c for c in df_dum.columns if c not in (target)
)
X_test = df_dum.filter(pl.col("dumVE") == 1).select(
    c for c in df_dum.columns if c not in (target)
)


# == target ==
y_train = df_dum.filter(pl.col("dumVE") == 0).select(["yd", "id"])
y_test = df_dum.filter(pl.col("dumVE") == 1).select(["yd", "id"])


# create X_train for default and non default
X_train_def = df_dum.filter((pl.col("dumVE") == 0) & (pl.col("yd") == 1)).select(
    c for c in df_dum.columns if c not in (target)
)
X_train_non_def = df_dum.filter((pl.col("dumVE") == 0) & (pl.col("yd") == 0)).select(
    c for c in df_dum.columns if c not in (target)
)


# == df for plot and visualisations ==

df_final = df_dum.filter(pl.col("dumVE") == 0)
df_val = df_dum.filter(pl.col("dumVE") == 1)

# == Early descive statistics ===
print(df_final.filter(pl.col("yd") == 1).describe().to_pandas())
print(df_final.filter(pl.col("yd") == 0).describe().to_pandas())

# 1.Boxplots and distribution plots

for column in df_final.columns:
    if column not in ["yd", "dumVE", "id"]:
        fig = px.histogram(
            df_final, x=column, color="yd", marginal="box", hover_data=df_final.columns
        )

        fig.show()

# == Compute Kurtosis and Skewness ===
results = []
for yd_value in [0, 1]:
    df_subset = df_final.filter(pl.col("yd") == yd_value).drop_nans()
    for column in df_final.columns:
        if column not in ["yd", "dumVE", "id"]:
            data = df_subset.select(df_final[column])
            kur = kurtosis(data, axis=0)
            sk = skew(data)
            print(
                f"Kurtosis = {kur} and Skewness = {sk} for variable {column} for yd = {yd_value}"
            )
            results.append(
                {"Group": yd_value, "Variable": column, "Kurtosis": kur, "Skewness": sk}
            )

df_stats = pl.DataFrame(results)
print(df_stats)

# == Conversion in latex table ===
if convert_to_latex == True:
    # Convert to LaTeX table
    latex_table = df_stats.to_pandas().to_latex(
        index=False,
        float_format="%.3f",
        caption="Skewness and Kurtosis of Variables",
        label="tab:skew_kurt",
        column_format="lrrr",
        escape=False,
    )
    latex_table = latex_table.replace(
        r"\begin{tabular}{lrrr}", r"\begin{tabular}{lrrr}" + "\n\\hline"
    ).replace(r"\end{tabular}", "\\hline\n\\end{tabular}")
    # Save to a .tex file
    with open("results/skew_kurtosis_table.tex", "w") as f:
        f.write(latex_table)

    print(latex_table)

else:
    pass

# == High leverage observations : How much a single point is different with respect to the others ? ===
skip = {"yd", "dumVE", "id"}

pdf = df_final.to_pandas()


X_cols = [c for c in pdf.select_dtypes(include=np.number).columns if c not in skip]
X = pdf[X_cols].to_numpy(dtype=float)

# Add intercept
X = np.column_stack([np.ones(len(X)), X])  # shape (n, p)
n, p = X.shape

row_ok = np.isfinite(X).all(axis=1)
Xc = X[row_ok]
idx = np.nonzero(row_ok)[0]

# Compute (X'X)^{-1}
XtX_inv = np.linalg.pinv(Xc.T @ Xc)

# Hat diagonal: h_i = x_i^T (X'X)^{-1} x_i
# Vectorized with einsum:
h = np.einsum("ij,jk,ik->i", Xc, XtX_inv, Xc)


pdf.loc[idx, "leverage"] = h

# Rule of thumb threshold: 2p/n (or use 3p/n to be stricter) ?
threshold = 2 * p / n
pdf["high_leverage"] = pdf["leverage"] > threshold

print(f"n={n}, p={p}, threshold (2p/n) = {threshold:.4f}")
print(
    pdf.loc[pdf["high_leverage"]]
    .sort_values("leverage", ascending=False)[["id"] + X_cols + ["leverage"]]
    .head(10)
)

# === Biserial test yd us categorical => default or not : see the correlation between yd and other numerical variables===


def pointbiserial_all(df: pl.DataFrame, y_col: str = "yd", skip=("yd", "id", "dumVE")):
    results_corr = []
    # y en numpy (0/1)
    y = df.select(pl.col(y_col)).to_series().to_numpy()
    for col in df.columns:
        if col in skip:
            continue

        x = df.select(pl.col(col)).to_series().to_numpy()

        mask = np.isfinite(x) & np.isfinite(y)
        x_, y_ = x[mask], y[mask]

        if x_.size < 3 or np.nanstd(x_) == 0 or np.nanstd(y_) == 0:
            r, p = np.nan, np.nan
        else:
            r, p = stats.pointbiserialr(x_, y_)
            r = float(r) if r is not None else np.nan
            p = float(p) if p is not None else np.nan
            t = r * np.sqrt((len(x_) - 2) / (1 - r**2))

        results_corr.append(
            {
                "variable": col,
                "r": r,
                "t_stat": None if np.isnan(r) else round(t, 3),
                "p_value": None if np.isnan(p) else round(p, 3),
            }
        )
    return pl.DataFrame(results_corr).sort("p_value", descending=False, nulls_last=True)


df_results_corr = pointbiserial_all(df_final)

print("=== Point-biserial (global estimation sample) ===")
print(df_results_corr)

if convert_to_latex == True:
    latex_table = df_results_corr.to_pandas().to_latex(
        index=False,
        float_format="%.3f",
        caption="Test of Point-Biserial Correlation",
        label="tab:test_corr",
    )

    # Save to a .tex file
    with open("results/test_corr.tex", "w") as f:
        f.write(latex_table)

    print(latex_table)
else:
    pass

#  == Most Correlated Variables with yd ===
print(df_results_corr.sort(by=pl.col("r").abs(), descending=True))
print(df_results_corr.sort(by=pl.col("t_stat").abs(), descending=True))

print("Sort by absolute value for t-stat and coefficient give the same result")


# === 7  Test with our variable reta: tyhe 4 tests give the same t-stat ===

# 1 - t-test for reta
g0 = df_final.filter(pl.col("yd") == 0)["tdta"]
g1 = df_final.filter(pl.col("yd") == 1)["tdta"]
t_stat, p_val = stats.ttest_ind(g0, g1, equal_var=False)
print(f"T-test: t={t_stat:.3f}, p={p_val:.3f}")

# 2 - biserial for reta
x = df_final["tdta"]
y = df_final["yd"]
r, p_val = stats.pointbiserialr(y, x)
print(f"Correlation test: r={r:.3f}, p={p_val:.3f}")


# Calcul de la t-stat à partir de r
n = len(x)
t_stat = r * np.sqrt((n - 2) / (1 - r**2))

print(f"r = {r:.3f}, t = {t_stat:.3f}, n = {n}")

# 3 ANIOVA regression for reta
y = df_final["tdta"].to_numpy()
X = sm.add_constant(df_final["yd"].to_numpy())  # YD en variable explicative
model_an = sm.OLS(y, X).fit()

print(model_an.summary())

# 4 Linear probability model for reta
y = df_final["yd"].to_numpy()
X = sm.add_constant(df_final["tdta"].to_numpy())  # TDTA en explicative
model = sm.OLS(y, X).fit()

print(model.summary())

# 8 === Bivariate correlation ===
columns = []
for column in df_final.drop_nans().columns:
    for column_2 in df_final.columns:
        if column_2 not in ["yd", "dumVE", "id"]:
            if column != column_2:
                corr = df_final.select(pl.corr(column, column_2)).item()
                if abs(corr) > 0.8:
                    print(f"Correlation between {column} and {column_2} is {corr}")
                    columns.append((column))
print(columns)
sns.pairplot(df_final.select(c for c in df_final.columns if c in columns).to_pandas())
df_corr = df_final.corr()

# === Bivariate clouds for the 6 variables :

cor_col = ["yd", "tdta", "reta", "opita", "ebita", "gempl"]
sns.pairplot(
    df_final.select(cor_col).to_pandas(),
    kind="reg",  # trace une droite de régression
    diag_kind="kde",  # densité (Kernel Density Estimation) sur la diagonale,
    plot_kws={"line_kws": {"color": "orange"}, "scatter_kws": {"alpha": 0.6, "s": 25}},
)
plt.suptitle("Pairplot with Regression Lines and Density on Diagonal", y=1.02)
plt.show()

# === Regression Models ===


y = df_final["yd"].to_pandas()
X = sm.add_constant(df_final["tdta"].to_pandas())

# === Linear regression (OLS) ===
model = sm.OLS(y, X).fit()

# === Logit ===
model_log = Logit(y, X).fit()
p_hat_logit = model_log.predict(X)

# === Probit ===
model_prob = Probit(y, X).fit()
p_hat_probit = model_prob.predict(X)

# === Combine predictions (Polars) ===
df_score = (
    df_final["yd", "tdta"]
    .with_columns(
        pl.Series("score_probit", p_hat_probit),
        pl.Series("score_logit", p_hat_logit),
    )
    .to_pandas()
)


# === Compute pseudo-R² for Logit/Probit ===
def pseudo_r2(model):
    """McFadden's pseudo R²"""
    if hasattr(model, "llf") and hasattr(model, "llnull"):
        return 1 - (model.llf / model.llnull)
    return None


r2_ols = model.rsquared
r2_logit = pseudo_r2(model_log)
r2_probit = pseudo_r2(model_prob)

# === Summary table ===
results_table = summary_col(
    [model, model_log, model_prob],
    stars=True,
    model_names=["OLS", "Logit", "Probit"],
    info_dict={
        "N": lambda x: f"{int(x.nobs)}",
        "R²": lambda x: (
            f"{x.rsquared:.3f}"
            if hasattr(x, "rsquared")  # OLS
            else f"{pseudo_r2(x):.3f}"
            if hasattr(x, "llf")  # Logit / Probit
            else "—"
        ),
    },
)

# === Clean output (remove redundant R-squared rows) ===
tbl = results_table.tables[0]
tbl.drop(
    index=[i for i in tbl.index if "R-squared" in i],
    inplace=True,
    errors="ignore",
)

# === Display ===
print(results_table)
print("\nCompare to default outputs")
print("\n=== OLS ===\n", model.summary())
print("\n=== Logit ===\n", model_log.summary())
print("\n=== Probit ===\n", model_prob.summary())

# 12 === Concordance pairs == hand made
p_hat = df_score["score_probit"]

concordant = 0
discordant = 0
ties = 0


for i in range(len(y)):
    for j in range(i + 1, len(y)):
        if y[i] != y[j]:
            if y[i] == 1 and y[j] == 0:
                if p_hat[i] > p_hat[j]:
                    concordant += 1
                elif p_hat[i] < p_hat[j]:
                    discordant += 1
                else:
                    ties += 1

            elif y[i] == 0 and y[j] == 1:
                if p_hat[j] > p_hat[i]:
                    concordant += 1
                elif p_hat[j] < p_hat[i]:
                    discordant += 1
                else:
                    ties += 1


total_pairs = concordant + discordant + ties

percent_concordant = concordant / total_pairs * 100 if total_pairs > 0 else 0


print(f"Percentage Concordance : {percent_concordant:.2f}%")
# === with sklearn ===

c_stat = roc_auc_score(y, p_hat)
percent_concordant_sk = c_stat * 100

print(f"Perrcentage Concordance : {percent_concordant:.2f}%")

if percent_concordant == percent_concordant_sk:
    print("same results for hand made and sklearn: ", percent_concordant)
else:
    print(
        "different results for hand made and sklearn: ",
        percent_concordant,
        percent_concordant_sk,
    )

# 13 === ols , probit and logit with prefered variables ===
y = df_final["yd"].to_pandas()
X = sm.add_constant(df_final["tdta", "gempl", "opita", "invsls"].to_pandas())

# result = stats.linregress(df_pd["tdta"], df_pd["yd"])
# === linear regression ===
model = sm.OLS(y, X).fit()


# === logit ====
model_log = Logit(y, X).fit()
p_hat_logit = model_log.predict(X)  # predicted probabilities

# === probit ===

model_prob = Probit(y, X).fit()
p_hat_probit = model_prob.predict(X)


df_score = (
    df_final["yd", "tdta"]
    .with_columns(
        pl.Series(p_hat_probit).alias("score_probit"),
        pl.Series(p_hat_logit).alias("score_logit"),
    )
    .to_pandas()
)


p_ols = model.predict(X)
p_logit = p_hat_logit
p_probit = p_hat_probit


def safe_auc(y_true, scores):
    try:
        return roc_auc_score(y_true, scores)
    except Exception:
        return np.nan


auc_ols = safe_auc(y, p_ols)
auc_logit = safe_auc(y, p_logit)
auc_probit = safe_auc(y, p_probit)

results_table = summary_col(
    [model, model_log, model_prob],
    stars=True,
    model_names=["OLS", "Logit", "Probit"],
    info_dict={
        "N": lambda x: f"{int(x.nobs)}",
        "R²": lambda x: (
            f"{x.rsquared:.3f}"
            if hasattr(x, "rsquared")
            else f"{x.prsquared:.3f}"
            if hasattr(x, "prsquared")
            else "—"
        ),
        "AUC": lambda x: (
            f"{auc_ols:.3f}"
            if x is model
            else f"{auc_logit:.3f}"
            if x is model_log
            else f"{auc_probit:.3f}"
        ),
    },
)

results_table.tables[0].drop(
    index=[i for i in results_table.tables[0].index if "R-squared" in i],
    inplace=True,
    errors="ignore",
)
print(results_table)
print("Compare to default outputs")
print(model.summary())
print(model_log.summary())
print(model_prob.summary())

df_results_table = results_table.tables[0]
df_results_table = pd.DataFrame(df_results_table)
df_results_table.reset_index(inplace=True)
df_results_table.rename(columns={"index": ""}, inplace=True)
# == Convert to LaTeX table ===
if convert_to_latex == True:
    latex_table = df_results_table.to_latex(
        index=False,
        float_format="%.3f",
        caption="Test of Point-Biserial Correlation",
        label="tab:test_corr",
    )

    # Save to a .tex file
    with open("results/reg.tex", "w") as f:
        f.write(latex_table)

    print(latex_table)
else:
    pass

# == Plot ROC Curve for each regression ==

regressions = ["ols", "logit", "probit"]
p_ols = model.predict(X)

# --- compute ROC and AUC for each model
fpr_ols, tpr_ols, _ = roc_curve(y, p_ols)
auc_ols = auc(fpr_ols, tpr_ols)
fpr_logit, tpr_logit, _ = roc_curve(y, p_hat_logit)
auc_logit = auc(fpr_logit, tpr_logit)
fpr_probit, tpr_probit, _ = roc_curve(y, p_hat_probit)
auc_probit = auc(fpr_probit, tpr_probit)
for regression in regressions:
    if regression == "ols":
        plt.figure(figsize=(6, 5))
        plt.plot(fpr_ols, tpr_ols, label=f"OLS (AUC = {auc_ols:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random guess")
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("ROC Curve: OLS")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    if regression == "logit":
        plt.figure(figsize=(6, 5))
        plt.plot(fpr_logit, tpr_logit, label=f"Logit (AUC = {auc_logit:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random guess")
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("ROC Curve: Logit")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    if regression == "probit":
        plt.figure(figsize=(6, 5))
        plt.plot(fpr_probit, tpr_probit, label=f"Probit (AUC = {auc_probit:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random guess")
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("ROC Curve: Probit")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

#  === Do the same for the validation sample ===
n_0 = len(df_val.filter(pl.col("yd") == 0))
n_1 = len(df_val.filter(pl.col("yd") == 1))
print(f"Number of defaulting firms {n_0}: vs non defaulting firms {n_1}")

# === ols , probit and logit with prefered variables ===
y = df_val["yd"].to_pandas()
X = sm.add_constant(df_val["tdta", "gempl", "opita", "invsls"].to_pandas())

# result = stats.linregress(df_pd["tdta"], df_pd["yd"])
# === linear regression ===
model = sm.OLS(y, X).fit()


# === logit ====
model_log = Logit(y, X).fit()
p_hat_logit = model_log.predict(X)  # predicted probabilities

# === probit ===

model_prob = Probit(y, X).fit()
p_hat_probit = model_prob.predict(X)


df_score = (
    df_final["yd", "tdta"]
    .with_columns(
        pl.Series(p_hat_probit).alias("score_probit"),
        pl.Series(p_hat_logit).alias("score_logit"),
    )
    .to_pandas()
)


p_ols = model.predict(X)
p_logit = p_hat_logit
p_probit = p_hat_probit


def safe_auc(y_true, scores):
    try:
        return roc_auc_score(y_true, scores)
    except Exception:
        return np.nan


auc_ols = safe_auc(y, p_ols)
auc_logit = safe_auc(y, p_logit)
auc_probit = safe_auc(y, p_probit)

results_table = summary_col(
    [model, model_log, model_prob],
    stars=True,
    model_names=["OLS", "Logit", "Probit"],
    info_dict={
        "N": lambda x: f"{int(x.nobs)}",
        "R²": lambda x: (
            f"{x.rsquared:.3f}"
            if hasattr(x, "rsquared")
            else f"{x.prsquared:.3f}"
            if hasattr(x, "prsquared")
            else "—"
        ),
        "AUC": lambda x: (
            f"{auc_ols:.3f}"
            if x is model
            else f"{auc_logit:.3f}"
            if x is model_log
            else f"{auc_probit:.3f}"
        ),
    },
)

results_table.tables[0].drop(
    index=[i for i in results_table.tables[0].index if "R-squared" in i],
    inplace=True,
    errors="ignore",
)
print(results_table)
print("Compare to default outputs")
print(model.summary())
print(model_log.summary())
print(model_prob.summary())

df_results_table = results_table.tables[0]
df_results_table = pd.DataFrame(df_results_table)
df_results_table.reset_index(inplace=True)
df_results_table.rename(columns={"index": ""}, inplace=True)
# == Convert to LaTeX table ===
if convert_to_latex == True:
    latex_table = df_results_table.to_latex(
        index=False,
        float_format="%.3f",
        caption="Test of Point-Biserial Correlation for Validation Sample",
        label="tab:test_corr",
    )

    # Save to a .tex file
    with open("results/reg_val.tex", "w") as f:
        f.write(latex_table)

    print(latex_table)
else:
    pass

# == Plot ROC Curve for each regression ==

regressions = ["ols", "logit", "probit"]
p_ols = model.predict(X)

# --- compute ROC and AUC for each model
fpr_ols, tpr_ols, _ = roc_curve(y, p_ols)
auc_ols = auc(fpr_ols, tpr_ols)
fpr_logit, tpr_logit, _ = roc_curve(y, p_hat_logit)
auc_logit = auc(fpr_logit, tpr_logit)
fpr_probit, tpr_probit, _ = roc_curve(y, p_hat_probit)
auc_probit = auc(fpr_probit, tpr_probit)
for regression in regressions:
    if regression == "ols":
        plt.figure(figsize=(6, 5))
        plt.plot(fpr_ols, tpr_ols, label=f"OLS (AUC = {auc_ols:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random guess")
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("ROC Curve: OLS")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    if regression == "logit":
        plt.figure(figsize=(6, 5))
        plt.plot(fpr_logit, tpr_logit, label=f"Logit (AUC = {auc_logit:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random guess")
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("ROC Curve: Logit")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    if regression == "probit":
        plt.figure(figsize=(6, 5))
        plt.plot(fpr_probit, tpr_probit, label=f"Probit (AUC = {auc_probit:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random guess")
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("ROC Curve: Probit")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

# 15 == Plots of the distribution of Standardized Person Residuals ==
p_ols = np.clip(p_ols, 1e-6, 1 - 1e-6)

# === Compute standardized Pearson residuals ===
r_ols = (y - p_ols) / np.sqrt(p_ols * (1 - p_ols))
r_logit = (y - p_logit) / np.sqrt(p_logit * (1 - p_logit))
r_probit = (y - p_probit) / np.sqrt(p_probit * (1 - p_probit))

df_resid = pd.DataFrame({"OLS": r_ols, "Logit": r_logit, "Probit": r_probit}).melt(
    var_name="Model", value_name="Pearson Residual"
)

# === plot pearson standazdized residuals ===
g = sns.FacetGrid(df_resid, col="Model", sharex=True, sharey=True, height=4, aspect=1.2)
g.map_dataframe(
    sns.kdeplot, x="Pearson Residual", fill=True, color="#1f77b4", alpha=0.4
)
g.set_titles("{col_name}")
g.set_axis_labels("Pearson Standardized Residuals", "Density")

for ax in g.axes.flat:
    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.set_xlim(-4, 4)

plt.tight_layout()
plt.show()
# == 16 Do per group : default and non default ===

y = df_val["yd"].to_pandas().reset_index(drop=True)
df_resid["yd"] = np.tile(y, 3)
# === Pearson residuals for defaulting firms ===
g = sns.FacetGrid(
    df_resid[df_resid["yd"] == 1],
    col="Model",
    sharex=True,
    sharey=True,
    height=4,
    aspect=1.2,
)
g.map_dataframe(
    sns.kdeplot, x="Pearson Residual", fill=True, color="#1f77b4", alpha=0.4
)
g.set_titles("{col_name}")
g.set_axis_labels("Pearson Standardized Residuals yd=1", "Density")
for ax in g.axes.flat:
    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.set_xlim(-4, 4)
plt.tight_layout()
plt.show()


# === Pearson standardized versus predites : test === p_ols
n = len(y)
assert len(df_resid) == 3 * n, (len(df_resid), n)
df_resid["y"] = np.tile(np.asarray(y), 3)

# 1) map from model -> predicted probabilities (make them numpy arrays)
eps = 1e-9
preds = {
    "logit": np.asarray(p_logit),
    "probit": np.asarray(p_probit),
    "ols": np.asarray(p_ols),
}

# OLS probabilities might be outside [0,1] -> clip
preds["ols"] = preds["ols"].clip(eps, 1 - eps)

# 2) iterate and plot
for model in ["Logit", "Probit", "OLS"]:
    key = model.lower()  # normalized key for the dict and mask

    # subset rows for the current model
    mask = df_resid["Model"].str.strip().str.lower().eq(key)
    d = df_resid.loc[mask].copy()

    # attach p_hat for this subset (length must equal n)
    assert len(d) == n, (len(d), n)
    d["p_hat"] = preds[key]

    # outlier flag
    d["outlier"] = d["Pearson Residual"].abs() > 2

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8, 4.5))

    d0 = d[d["y"] == 0]
    d1 = d[d["y"] == 1]
    ax.scatter(
        d0["p_hat"], d0["Pearson Residual"], s=30, alpha=0.9, label="0", color="blue"
    )
    ax.scatter(
        d1["p_hat"], d1["Pearson Residual"], s=30, alpha=0.9, label="1", color="red"
    )

    ax.axhline(0, color="black", ls="--", lw=1, label="y=0")
    ax.axhline(2, color="orange", ls="--", lw=1, label="y=2 (Outlier threshold)")
    ax.axhline(-2, color="orange", ls="--", lw=1, label="y=-2 (Outlier threshold)")

    out = d[d["outlier"]]
    ax.scatter(
        out["p_hat"],
        out["Pearson Residual"],
        s=80,
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
        label="Outliers",
    )

    ax.set_xlabel("Predicted Probability of Default")
    ax.set_ylabel("Standardized Pearson Residuals")
    ax.set_title(
        f"Standardized Pearson Residuals vs Predicted Probability of Default ({model})"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

# 17 === Loss function == Redaction cf slide 261
"""
type 1 : lend to a bankrupt next period 
type 2 : loss of profit next period
"""

# 19 == dummy trap ===

# === Create new dummy ===
df_trap = df_final.with_columns((1 - pl.col("yd")).alias("ynd").cast(pl.Int8))

# # => default : 0
# # => non default : 1
# # === Run OLS ==
# y = df_trap["ydn"].to_pandas()
# X = sm.add_constant(df_final["tdta"].to_pandas())

# result = stats.linregress(df_pd["tdta"], df_pd["yd"])
# === linear regression on yd et ynd ===
# model_dum = sm.OLS(y, X).fit()
m1 = smf.ols("tdta ~ yd + ynd", data=df_trap.to_pandas()).fit()
print(m1.summary())

# comment perfect colinearity => " The smallest eigenvalue is 8.12e-31. This might indicate that there are
# strong multicollinearity problems or that the design matrix is singular."

# === linear regression on yd ===
m2 = smf.ols("tdta ~ yd", data=df_trap.to_pandas()).fit()
print(m2.summary())

# === linear regression on ynd ===
m3 = smf.ols("tdta ~ ynd", data=df_trap.to_pandas()).fit()
print(m3.summary())


# ==== linear regression on ynd and yd and ynd+yd = 1 ===
df_trap = df_trap.with_columns((pl.col("yd") - pl.col("ynd")).alias("z"))
m4 = smf.ols("tdta ~ z", data=df_trap.to_pandas()).fit()
print(m4.summary())
print(m4.params)


# 2) GLM ~ Gaussian(identity) == OLS in point estimates
#    (and the constraint resolves the dummy trap)
glm_model = smf.glm(
    "tdta ~ yd + ynd", data=df_trap.to_pandas(), family=sm.families.Gaussian()
)

# 3) Fit with the linear restriction: coef(yd) + coef(ynd) = 0
glm_constrained = glm_model.fit_constrained("yd + ynd = 0")

print(glm_constrained.summary())
print("Params:", glm_constrained.params)
