# ============================================================
# Traduction du script Python vers R
# Auteur: Juliette (adapté par l'assistant)
# Objet : Analyse scoring défaut — descriptifs, tests, modèles, ROC, résidus
# ============================================================

# ---- Packages ----
suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(GGally)       # pour ggpairs
  library(pROC)         # AUC / ROC
  library(moments)      # skewness, kurtosis
  library(car)          # leveneTest
  library(modelsummary) # jolis tableaux (optionnel)
  library(corrplot)     # heatmap de corrélation
})

# ---- Paramètres ----
convert_to_latex <- FALSE
if (!convert_to_latex) {
  warning("Latex conversion skipped. Set convert_to_latex <- TRUE to enable it.")
} else {
  warning("Latex conversion not skipped. Set convert_to_latex <- FALSE to stop it.")
}

# ---- Import & nettoyage (équiv. Polars) ----
dt <- fread(
  "C:/Users/Juliette/OneDrive/A Master FE/Scoring/defaut2000.csv",
  sep = ";",
  na.strings = c("-99.99", "NA", "")
)

dt <- na.omit(dt)

# Tri (yd, reta)
setorder(dt, yd, reta)

# Index & split estimation/validation (pairs/impairs) ----
dt[, id := .I - 1L]
# dumVE = 0 -> impairs (Python: odd rows => Estimation)
# dumVE = 1 -> pairs   (Python: even rows => Validation)
dt[, dumVE := as.integer(id %% 2L == 0L)]

# Cibles & jeux
TARGET <- "yd"
train <- dt[dumVE == 0]
val   <- dt[dumVE == 1]

# ==== Statistiques descriptives par classe yd ====
cat("\n--- Descriptifs (yd = 1) ---\n")
print(train[yd == 1, lapply(.SD, \(x) c(N=.N, mean=mean(x, na.rm=TRUE), sd=sd(x, na.rm=TRUE))),
            .SDcols = is.numeric])
cat("\n--- Descriptifs (yd = 0) ---\n")
print(train[yd == 0, lapply(.SD, \(x) c(N=.N, mean=mean(x, na.rm=TRUE), sd=sd(x, na.rm=TRUE))),
            .SDcols = is.numeric])

# ==== 1) Histogrammes + boxplots superposés par yd ====
num_cols <- names(train)[sapply(train, is.numeric)]
num_cols <- setdiff(num_cols, c("yd", "dumVE", "id"))

for (col in num_cols) {
  p <- ggplot(train, aes(x = .data[[col]], fill = factor(yd))) +
    geom_histogram(alpha = 0.6, bins = 30, position = "identity") +
    facet_wrap(~ yd, nrow = 2, scales = "free_y") +
    theme_minimal() +
    labs(title = paste("Distribution de", col), fill = "yd")
  print(p)
  
  p2 <- ggplot(train, aes(x = factor(yd), y = .data[[col]], fill = factor(yd))) +
    geom_boxplot(alpha = 0.6) +
    theme_minimal() +
    labs(title = paste("Boxplot de", col, "par yd"), x = "yd", y = col, fill = "yd")
  print(p2)
}

# ==== 2) Kurtosis & Skewness par groupe ====
res_stats <- rbindlist(lapply(c(0,1), function(g){
  sub <- train[yd == g]
  r <- lapply(num_cols, function(col){
    x <- sub[[col]]
    data.table(Group=g, Variable=col,
               Kurtosis = suppressWarnings(kurtosis(x, na.rm=TRUE)),
               Skewness = suppressWarnings(skewness(x, na.rm=TRUE)))
  })
  rbindlist(r)
}))
print(res_stats)

if (convert_to_latex) {
  msummary(list("Skewness/Kurtosis" = res_stats), output = "results/skew_kurtosis_table.tex")
}

# ==== 3) Leverage (valeurs de levier) ====
skip <- c("yd", "dumVE", "id")
Xcols <- setdiff(names(train)[sapply(train, is.numeric)], skip)
X <- as.matrix(cbind(`(Intercept)`=1, train[, ..Xcols]))
n <- nrow(X); p <- ncol(X)
XtX_inv <- solve(crossprod(X))
h <- rowSums((X %*% XtX_inv) * X)
train[, leverage := h]
threshold <- 2 * p / n
train[, high_leverage := leverage > threshold]
cat(sprintf("n=%d, p=%d, threshold (2p/n)=%.4f\n", n, p, threshold))
print(train[high_leverage == TRUE][order(-leverage)][, c("id", Xcols, "leverage"), with=FALSE][1:10])

# ==== 4) Tests de normalité & d'égalité des variances (Shapiro + Bartlett/Levene) ====

homog_var_one <- function(df, value_col, group_col = "yd", alpha = 0.05){
  sub <- df[, c(group_col, value_col), with=FALSE]
  sub <- sub[complete.cases(sub)]
  g0 <- sub[get(group_col)==0][[value_col]]
  g1 <- sub[get(group_col)==1][[value_col]]
  n0 <- length(g0); n1 <- length(g1)
  if (n0 < 3 || n1 < 3) return(data.table(Variable=value_col, test=NA, pvalue=NA,
                                          decision_normality=NA, p_shapiro_0=NA, p_shapiro_1=NA,
                                          equal_variances=NA, n0=n0, n1=n1))
  sh0 <- shapiro.test(g0)
  sh1 <- shapiro.test(g1)
  decision_normality <- (sh0$p.value > alpha) && (sh1$p.value > alpha)
  var0 <- var(g0); var1 <- var(g1)
  if (var0 == 0 || var1 == 0) return(data.table(Variable=value_col, test="undefined (zero variance)",
                                                pvalue=NA, decision_normality=decision_normality,
                                                p_shapiro_0=sh0$p.value, p_shapiro_1=sh1$p.value,
                                                equal_variances=(var0==var1), n0=n0, n1=n1))
  if (decision_normality) {
    bt <- bartlett.test(x = list(g0, g1))
    data.table(Variable=value_col, test="bartlett", pvalue=round(bt$p.value,3),
               decision_normality=decision_normality, p_shapiro_0=round(sh0$p.value,3),
               p_shapiro_1=round(sh1$p.value,3), equal_variances = (bt$p.value >= alpha),
               n0=n0, n1=n1)
  } else {
    # centre = median -> Levene robuste
    grp <- factor(sub[[group_col]])
    lv <- car::leveneTest(sub[[value_col]], grp, center = median)
    p <- lv$`Pr(>F)`[1]
    data.table(Variable=value_col, test="levene", pvalue=round(p,3),
               decision_normality=decision_normality, p_shapiro_0=round(sh0$p.value,3),
               p_shapiro_1=round(sh1$p.value,3), equal_variances = (p >= alpha),
               n0=n0, n1=n1)
  }
}

homog_var_all <- function(df, group_col = "yd", alpha = 0.05, skip = c("yd","id","dumVE")){
  out <- rbindlist(lapply(names(df), function(col){
    if (col %in% skip) return(NULL)
    if (is.numeric(df[[col]])) homog_var_one(df, col, group_col, alpha) else NULL
  }), fill = TRUE)
  if (nrow(out)) setorder(out, equal_variances, pvalue)
  out
}

res_homog <- homog_var_all(train, group_col = "yd", alpha = 0.05)
print(res_homog)
cat("\nVariables rejetant H0 d'égalité des variances (p < 0.05):\n")
print(res_homog[pvalue < 0.05, .(Variable, test, pvalue, equal_variances)])

if (convert_to_latex) {
  modelsummary::datasummary_df(res_homog, output = "results/table_normality_variances.tex")
}

# ==== 5) Tests d'égalité de moyennes (t-test) ====
force_equal_var <- c("tdta", "reta", "nwcta")  # comme dans le script Python
res_means <- rbindlist(lapply(num_cols, function(col){
  rvs0 <- train[yd==0][[col]]
  rvs1 <- train[yd==1][[col]]
  eqv <- col %in% force_equal_var
  tt <- t.test(rvs0, rvs1, var.equal = eqv)
  data.table(Variable = col, t_statistic = unname(tt$statistic), p_value = unname(tt$p.value))
}))
print(res_means[order(p_value)])
cat("\nSi p_value < 0.05 on rejette l'égalité des moyennes.\n")
print(res_means[p_value < 0.05])

if (convert_to_latex) {
  modelsummary::datasummary_df(res_means, output = "results/test_means.tex")
}

# ==== 6) Corrélation bisérielle point (yd binaire ~ numérique) ====
pointbiserial_all <- function(df, y_col = "yd", skip = c("yd","id","dumVE")){
  y <- as.numeric(df[[y_col]])
  rbindlist(lapply(names(df), function(col){
    if (col %in% skip) return(NULL)
    x <- df[[col]]
    if (!is.numeric(x)) return(NULL)
    ok <- is.finite(x) & is.finite(y)
    x <- x[ok]; y0 <- y[ok]
    if (length(x) < 3 || sd(x)==0 || sd(y0)==0) return(data.table(variable=col, r=NA_real_, t_stat=NA_real_, p_value=NA_real_))
    ct <- suppressWarnings(cor.test(x, y0, method = "pearson"))
    r <- unname(ct$estimate)
    t <- r * sqrt((length(x) - 2) / (1 - r^2))
    data.table(variable=col, r=r, t_stat=round(t,3), p_value=round(unname(ct$p.value),3))
  }))[, .SD[order(p_value)]]
}

res_corr <- pointbiserial_all(train)
cat("\n=== Corrélation point-bisérielle (échantillon estimation) ===\n")
print(res_corr)

# Top variables
print(res_corr[order(-abs(r))])
print(res_corr[order(-abs(t_stat))])
cat("\nTri par |t| et |r| donne le même ordre.\n")

# ==== 7) Vérif sur une variable (ex: tdta) ====
# t-test
g0 <- train[yd==0][["tdta"]]
g1 <- train[yd==1][["tdta"]]
cat(sprintf("T-test tdta: t=%.3f, p=%.3f\n", t.test(g0, g1, var.equal = FALSE)$statistic, t.test(g0, g1, var.equal = FALSE)$p.value))
# corrélation
tdta <- train[["tdta"]]
yd_v <- train[["yd"]]
ct <- cor.test(tdta, yd_v)
r <- unname(ct$estimate); n <- sum(complete.cases(tdta, yd_v)); tstat <- r * sqrt((n-2)/(1-r^2))
cat(sprintf("Correlation tdta~yd: r=%.3f, p=%.3f, t=%.3f, n=%d\n", r, ct$p.value, tstat, n))
# ANOVA (yd explicative, tdta réponse)
cat("\n=== ANOVA OLS (tdta ~ yd) ===\n")
print(summary(lm(tdta ~ yd, data = train)))
# LPM: yd ~ tdta
cat("\n=== OLS (yd ~ tdta) ===\n")
print(summary(lm(yd ~ tdta, data = train)))

# ==== 8) Corrélations bivariées fortes (> 0.8) ====
nums <- train[, ..num_cols]
C <- suppressWarnings(cor(nums, use = "pairwise.complete.obs"))
thr <- 0.8

# Indices des corrélations |r| > thr (hors diagonale)
above <- which(abs(C) > thr & abs(C) < 1, arr.ind = TRUE)

if (nrow(above)) {
  # Tableau des paires triées par |r| décroissant
  pairs_dt <- data.table(
    var1 = colnames(C)[above[,1]],
    var2 = colnames(C)[above[,2]],
    r    = C[above]
  )[order(-abs(r))]
  print(pairs_dt)
  
  # Colonnes à représenter dans le pairplot
  cols_high <- sort(unique(c(pairs_dt$var1, pairs_dt$var2)))
  if (length(cols_high) >= 2) {
    print(GGally::ggpairs(train[, ..cols_high]))
  }
} else {
  message(sprintf("Aucune paire avec |r| > %.2f", thr))
}

# Corrplot absolu (triangle inférieur)
CA <- abs(C)
CA[upper.tri(CA, diag = TRUE)] <- NA
corrplot(CA, method = "color", tl.col = "black", na.label = " ", tl.cex = 0.7)

# ==== 9) Modèles (OLS, Logit, Probit) — 1 variable tdta ==== (OLS, Logit, Probit) — 1 variable tdta ====
y <- train$yd
X <- train$tdta

m_ols   <- lm(yd ~ tdta, data = train)
m_logit <- glm(yd ~ tdta, data = train, family = binomial(link = "logit"))
m_probit<- glm(yd ~ tdta, data = train, family = binomial(link = "probit"))

p_ols    <- pmin(pmax(predict(m_ols, type = "response"), 1e-6), 1-1e-6) # clip [0,1]
p_logit  <- predict(m_logit, type = "response")
p_probit <- predict(m_probit, type = "response")

# Pseudo-R2 de McFadden
mcfadden <- function(mod){
  if (inherits(mod, "glm")) {
    1 - (as.numeric(logLik(mod)) / as.numeric(logLik(update(mod, . ~ 1))))
  } else if (inherits(mod, "lm")) {
    summary(mod)$r.squared
  } else NA_real_
}

r2_ols <- mcfadden(m_ols); r2_logit <- mcfadden(m_logit); r2_probit <- mcfadden(m_probit)

# Tableau comparatif (corrigé: gof_map en data.frame)
gof_map_df <- data.frame(
  raw   = c("nobs", "r.squared", "ll"),
  clean = c("N", "R² (OLS)", "LogLik"),
  fmt   = c(0, 3, 3),
  stringsAsFactors = FALSE
)
modelsummary(list(OLS = m_ols, Logit = m_logit, Probit = m_probit),
             gof_map = gof_map_df,
             stars = TRUE)

cat(sprintf("R² OLS = %.3f | McFadden Logit = %.3f | McFadden Probit = %.3f\n", r2_ols, r2_logit, r2_probit))

# ROC & AUC
roc_ols   <- roc(y, p_ols);   auc_ols   <- as.numeric(auc(roc_ols))
roc_logit <- roc(y, p_logit); auc_logit <- as.numeric(auc(roc_logit))
roc_probit<- roc(y, p_probit);auc_probit<- as.numeric(auc(roc_probit))

plot(roc_ols, main = sprintf("ROC OLS (AUC=%.3f)", auc_ols))
plot(roc_logit, main = sprintf("ROC Logit (AUC=%.3f)", auc_logit))
plot(roc_probit, main = sprintf("ROC Probit (AUC=%.3f)", auc_probit))

# ==== 10) Concordance (à la main) vs AUC ====
concordance_manual <- function(y, p){
  y <- as.integer(y)
  n <- length(y)
  conc <- disc <- ties <- 0L
  for (i in 1:(n-1)){
    for (j in (i+1):n){
      if (y[i] != y[j]){
        if (y[i]==1 && y[j]==0){
          if (p[i] > p[j]) conc <- conc + 1L else if (p[i] < p[j]) disc <- disc + 1L else ties <- ties + 1L
        } else if (y[i]==0 && y[j]==1){
          if (p[j] > p[i]) conc <- conc + 1L else if (p[j] < p[i]) disc <- disc + 1L else ties <- ties + 1L
        }
      }
    }
  }
  tot <- conc + disc + ties
  if (tot == 0) return(0)
  100 * conc / tot
}

pc_manual <- concordance_manual(y, p_probit)
pc_sklearn <- as.numeric(auc(roc(y, p_probit))) * 100
cat(sprintf("Pourcentage de concordance (manuel)=%.2f%% | via AUC=%.2f%%\n", pc_manual, pc_sklearn))

# ==== 11) Modèles multivariés préférés (tdta + gempl + opita + invsls) ====
form_pref <- as.formula("yd ~ tdta + gempl + opita + invsls")

m2_ols    <- lm(form_pref, data = train)
m2_logit  <- glm(form_pref, data = train, family = binomial("logit"))
m2_probit <- glm(form_pref, data = train, family = binomial("probit"))

p2_ols    <- pmin(pmax(predict(m2_ols, type = "response"), 1e-6), 1-1e-6)
p2_logit  <- predict(m2_logit, type = "response")
p2_probit <- predict(m2_probit, type = "response")

safe_auc <- function(y, s){
  tryCatch(as.numeric(auc(roc(y, s))), error = function(e) NA_real_)
}
AUC_ols <- safe_auc(y, p2_ols); AUC_log <- safe_auc(y, p2_logit); AUC_prob <- safe_auc(y, p2_probit)

modelsummary(list(OLS=m2_ols, Logit=m2_logit, Probit=m2_probit),
             add_rows = data.frame(term = c("N", "AUC"),
                                   OLS = c(nobs(m2_ols), sprintf("%.3f", AUC_ols)),
                                   Logit = c(nobs(m2_logit), sprintf("%.3f", AUC_log)),
                                   Probit = c(nobs(m2_probit), sprintf("%.3f", AUC_prob))))

# ROC plots
plot(roc(y, p2_ols),   main = sprintf("ROC OLS (AUC=%.3f)",  AUC_ols))
plot(roc(y, p2_logit), main = sprintf("ROC Logit (AUC=%.3f)",AUC_log))
plot(roc(y, p2_probit),main = sprintf("ROC Probit (AUC=%.3f)",AUC_prob))

# ==== 12) Validation sur l'échantillon de validation ====
yv <- val$yd
mV_ols    <- lm(form_pref, data = val)
mV_logit  <- glm(form_pref, data = val, family = binomial("logit"))
mV_probit <- glm(form_pref, data = val, family = binomial("probit"))

pv_ols    <- pmin(pmax(predict(mV_ols, type = "response"), 1e-6), 1-1e-6)
pv_logit  <- predict(mV_logit, type = "response")
pv_probit <- predict(mV_probit, type = "response")

AUCv_ols <- safe_auc(yv, pv_ols); AUCv_log <- safe_auc(yv, pv_logit); AUCv_prob <- safe_auc(yv, pv_probit)

modelsummary(list(OLS=mV_ols, Logit=mV_logit, Probit=mV_probit),
             add_rows = data.frame(term = c("N", "AUC"),
                                   OLS = c(nobs(mV_ols), sprintf("%.3f", AUCv_ols)),
                                   Logit = c(nobs(mV_logit), sprintf("%.3f", AUCv_log)),
                                   Probit = c(nobs(mV_probit), sprintf("%.3f", AUCv_prob))))

plot(roc(yv, pv_ols),   main = sprintf("ROC OLS (val) AUC=%.3f",  AUCv_ols))
plot(roc(yv, pv_logit), main = sprintf("ROC Logit (val) AUC=%.3f",AUCv_log))
plot(roc(yv, pv_probit),main = sprintf("ROC Probit (val) AUC=%.3f",AUCv_prob))

# ==== 13) Résidus de Pearson standardisés (selon p-hat) ====
resid_df <- data.table(
  OLS    = (y  - p2_ols)   / sqrt(p2_ols*(1-p2_ols)),
  Logit  = (y  - p2_logit) / sqrt(p2_logit*(1-p2_logit)),
  Probit = (y  - p2_probit)/ sqrt(p2_probit*(1-p2_probit))
)
resid_long <- melt(resid_df, measure.vars = c("OLS","Logit","Probit"),
                   variable.name = "Model", value.name = "PearsonResidual")

g <- ggplot(resid_long, aes(x = PearsonResidual)) +
  geom_density(alpha = .4, fill = "#1f77b4") +
  facet_wrap(~ Model, nrow = 1, scales = "free_y") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  coord_cartesian(xlim = c(-4,4)) +
  theme_minimal() + labs(title = "Pearson Residuals (standardisés)")
print(g)

# par groupe yd (sur validation comme dans Python)
resid_df$yd <- yv[seq_len(nrow(resid_df))] # aligne tailles si besoin
resid_long2 <- melt(resid_df[yd == 1], measure.vars = c("OLS","Logit","Probit"),
                    variable.name = "Model", value.name = "PearsonResidual")
print(ggplot(resid_long2, aes(x = PearsonResidual)) +
        geom_density(alpha = .4, fill = "#1f77b4") +
        facet_wrap(~ Model, nrow = 1) +
        geom_vline(xintercept = 0, linetype = "dashed") +
        coord_cartesian(xlim = c(-4,4)) + theme_minimal() +
        labs(title = "Pearson Residuals (yd=1)"))

# Résidus vs p_hat + outliers |r|>2
plot_resid_vs_phat <- function(y, p_hat, model_name){
  d <- data.table(y=y, p_hat=p_hat,
                  resid = (y - p_hat)/sqrt(p_hat*(1-p_hat)))
  d[, outlier := abs(resid) > 2]
  ggplot(d, aes(x = p_hat, y = resid, color = factor(y))) +
    geom_point(alpha = 0.8) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_hline(yintercept = c(-2,2), linetype = "dashed", color = "orange") +
    geom_point(data = d[outlier==TRUE], size = 2, shape = 21, stroke = 0.8, fill = NA, color = "red") +
    theme_minimal() + labs(title = paste("Résidus vs p_hat (", model_name, ")"),
                           x = "Probabilité prédite", y = "Résidus de Pearson")
}
print(plot_resid_vs_phat(y, p2_logit,  "Logit"))
print(plot_resid_vs_phat(y, p2_probit, "Probit"))
print(plot_resid_vs_phat(y, p2_ols,    "OLS"))

# ==== 14) Dummy trap ====
# Crée la contre-dummy
dt[, ynd := as.integer(1 - yd)]
# Régression tdta ~ yd + ynd : R détecte colinéarité parfaite et supprime une colonne
m_dum <- lm(tdta ~ yd + ynd, data = dt)
print(summary(m_dum))

# Régressions séparées
print(summary(lm(tdta ~ yd,  data = dt)))
print(summary(lm(tdta ~ ynd, data = dt)))

# Re-paramétrisation z = yd - ynd (contrainte coef(yd)+coef(ynd)=0)
dt[, z := yd - ynd]
print(summary(lm(tdta ~ z, data = dt)))

# GLM gaussien avec test de contrainte (via car::linearHypothesis)
glm_gauss <- glm(tdta ~ yd + ynd, data = dt, family = gaussian())
print(summary(glm_gauss))
# Tester la contrainte yd + ynd = 0 (équivalente à coef(yd) = -coef(ynd))
print(car::linearHypothesis(glm_gauss, c("yd + ynd = 0")))

# (Option : pour imposer la contrainte, on utilise la re-paramétrisation z ci-dessus)

# ============================================================
# Fin du script
# ============================================================
