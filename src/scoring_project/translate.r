# translation in R of the Python code in scoring.py using LLM
suppressPackageStartupMessages({
  base_pkgs <- c("data.table","dplyr","ggplot2","e1071","car","MASS","pROC","broom","stringr")
  opt_pkgs  <- c("GGally","ggExtra")
  
  need <- setdiff(c(base_pkgs, opt_pkgs), rownames(installed.packages()))
  if (length(need)) install.packages(need, dependencies = TRUE)
  
  lapply(base_pkgs, require, character.only = TRUE)
  
  has_GGally  <- requireNamespace("GGally",  quietly = TRUE)
  if (has_GGally)  library(GGally)
  
  has_ggExtra <- requireNamespace("ggExtra", quietly = TRUE)
  if (has_ggExtra) library(ggExtra)
})

#%% === Load & clean
df_ju <- fread(
  "./defaut2000.csv",
  sep = ";",
  na.strings = c("-99.99"),
  showProgress = FALSE
) |>
  tidyr::drop_na()

#%% === Sort
df_sort <- df_ju |> arrange(yd, reta)

# dumVE = 0 => odd rows => Estimation ; dumVE = 1 => even rows => Validation
df_dum <- df_sort |>
  mutate(id = row_number(),
         dumVE = as.integer((id %% 2) == 0))

# Train/Test split (deterministic like in Python)
target <- "yd"
X_train <- subset(df_dum, dumVE == 0)[ , setdiff(names(df_dum), target), drop = FALSE]
X_test  <- subset(df_dum, dumVE == 1)[ , setdiff(names(df_dum), target), drop = FALSE]

y_train <- subset(df_dum, dumVE == 0)[ , c("yd","id"), drop = FALSE]
y_test  <- subset(df_dum, dumVE == 1)[ , c("yd","id"), drop = FALSE]

# Data for estimation/validation
df_final <- df_dum |> filter(dumVE == 0)
df_val   <- df_dum |> filter(dumVE == 1)

#%% 1) Histogrammes + Boxplots par variable
num_cols  <- names(df_final)
skip_cols <- c("yd","dumVE","id")

for (col in setdiff(num_cols, skip_cols)) {
  if (!is.numeric(df_final[[col]])) next
  
  p_hist <- ggplot(df_final, aes(x = .data[[col]], fill = factor(yd))) +
    geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
    labs(title = paste("Histogram of", col), x = col, fill = "yd") +
    theme_minimal()
  print(p_hist)
  
  p_box <- ggplot(df_final, aes(x = factor(yd), y = .data[[col]], fill = factor(yd))) +
    geom_boxplot(alpha = 0.7, outlier.alpha = 0.3) +
    labs(title = paste("Boxplot of", col, "by yd"), x = "yd", y = col) +
    theme_minimal() + guides(fill = "none")
  print(p_box)
}

# Descriptifs rapides
print(summary(df_final))

#%% 2) Skewness & Kurtosis by yd group (console only)
results <- list()
for (yd_val in c(0, 1)) {
  df_sub <- df_final |> filter(yd == yd_val)
  for (col in setdiff(names(df_final), skip_cols)) {
    x <- df_sub[[col]]
    if (is.numeric(x)) {
      kur <- tryCatch(kurtosis(x, na.rm = TRUE, type = 2), error = function(e) NA_real_)
      sk  <- tryCatch(skewness(x, na.rm = TRUE, type = 2), error = function(e) NA_real_)
      cat(sprintf("Kurtosis = %s and Skewness = %s for %s | yd=%d\n",
                  round(kur, 3), round(sk, 3), col, yd_val))
      results[[length(results) + 1]] <- data.frame(
        Group = yd_val, Variable = col, Kurtosis = kur, Skewness = sk
      )
    }
  }
}
df_stats <- dplyr::bind_rows(results)
print(df_stats)

#%% High leverage observations (OLS “hat values”)
pdf <- df_final
num_X_cols <- pdf |>
  dplyr::select(dplyr::where(is.numeric), -dplyr::all_of(c("yd", "dumVE", "id"))) |>
  names()

lm_data <- pdf |>
  dplyr::select(dplyr::all_of(c("id","yd", num_X_cols))) |>
  tidyr::drop_na()

form_leverage <- as.formula(paste("yd ~", paste(num_X_cols, collapse = " + ")))
fit_leverage  <- lm(form_leverage, data = lm_data)
h <- hatvalues(fit_leverage)
lm_data$leverage <- as.numeric(h)

Xmat <- model.matrix(fit_leverage)
n <- nrow(Xmat); p <- ncol(Xmat)
threshold <- 2 * p / n
lm_data$high_leverage <- lm_data$leverage > threshold

cat(sprintf("n=%d, p=%d, threshold (2p/n)=%0.4f\n", n, p, threshold))
print(
  lm_data |>
    arrange(desc(leverage)) |>
    dplyr::select(id, leverage) |>
    head(10)
)

# Plots densités par high_leverage
for (col in num_X_cols) {
  dfp <- lm_data |> dplyr::select(dplyr::all_of(c(col, "high_leverage"))) |>
    mutate(high_leverage = factor(high_leverage))
  g <- ggplot(dfp, aes(x = .data[[col]], fill = high_leverage)) +
    geom_histogram(aes(y = after_stat(density)), position = "identity", alpha = 0.4, bins = 30) +
    geom_density(linewidth = 1) +
    labs(title = paste("Distribution of", col, "by high_leverage")) +
    theme_minimal()
  print(g)
}

#%% Normality & equality of variances tests (Shapiro + Bartlett/Levene)
test_homogeneite_variances_variable <- function(df, value_col, group_col = "yd", alpha = 0.05) {
  df <- df |> tidyr::drop_na()
  g0 <- df |> dplyr::filter(.data[[group_col]] == 0) |> dplyr::pull(!!value_col)
  g1 <- df |> dplyr::filter(.data[[group_col]] == 1) |> dplyr::pull(!!value_col)
  if (length(g0) < 3 || length(g1) < 3) {
    return(data.frame(Variable = value_col, test = NA, pvalue = NA, decision_normalite = NA))
  }
  sh0 <- shapiro.test(g0); sh1 <- shapiro.test(g1)
  decision_normalite <- (sh0$p.value > alpha) && (sh1$p.value > alpha)
  if (decision_normalite) {
    bt <- bartlett.test(list(g0, g1)); test_name <- "bartlett"; p <- bt$p.value
  } else {
    grp <- factor(c(rep(0, length(g0)), rep(1, length(g1))))
    val <- c(g0, g1)
    lv <- car::leveneTest(val ~ grp, center = median)
    test_name <- "levene"; p <- lv$`Pr(>F)`[1]
  }
  data.frame(Variable = value_col, test = test_name,
             pvalue = round(p, 3),
             decision_normalite = decision_normalite)
}

test_all_variables <- function(df, group_col = "yd", alpha = 0.05, skip = c("yd","id","dumVE")) {
  res <- lapply(setdiff(names(df), skip), function(col) {
    if (is.numeric(df[[col]])) test_homogeneite_variances_variable(df, col, group_col, alpha)
  })
  dplyr::bind_rows(res)
}

df_results <- test_all_variables(df_final, group_col = "yd", alpha = 0.05)
print(df_results)
message("For variables with p-value < 0.05, reject equal variances.")
print(df_results |> dplyr::filter(pvalue < 0.05))

#%% Equality of means (t-tests)
results_means <- list()
for (col in setdiff(names(df_final), c("yd","dumVE","id"))) {
  rvs0 <- df_final |> dplyr::filter(yd == 0) |> dplyr::pull(!!col)
  rvs1 <- df_final |> dplyr::filter(yd == 1) |> dplyr::pull(!!col)
  if (!is.numeric(rvs0)) next
  if (col %in% c("tdta","reta","nwcta")) {
    tt <- t.test(rvs0, rvs1, var.equal = FALSE)
  } else {
    tt <- t.test(rvs0, rvs1, var.equal = TRUE)
  }
  results_means[[length(results_means)+1]] <-
    data.frame(Variable = col, t_statistic = unname(tt$statistic), p_value = unname(tt$p.value))
}
df_results_means <- dplyr::bind_rows(results_means)
print(df_results_means |> arrange(p_value))
message("If p_val < 0.05, means differ significantly.")
print(df_results_means |> dplyr::filter(p_value < 0.05))

#%% Point-biserial correlations (Pearson between binary yd and numeric x)
pointbiserial_all <- function(df, y_col = "yd", skip = c("yd","id","dumVE")) {
  out <- list(); y <- df[[y_col]]
  for (col in setdiff(names(df), skip)) {
    x <- df[[col]]
    if (is.numeric(x)) {
      ok <- is.finite(x) & is.finite(y)
      x_ <- x[ok]; y_ <- y[ok]
      if (length(x_) >= 3 && sd(x_) > 0 && sd(y_) > 0) {
        ct <- suppressWarnings(cor.test(x_, y_, method = "pearson"))
        r <- as.numeric(ct$estimate)
        n <- length(x_)
        t_stat <- r * sqrt((n - 2) / (1 - r^2))
        p <- ct$p.value
      } else {
        r <- NA; t_stat <- NA; p <- NA
      }
      out[[length(out)+1]] <- data.frame(variable = col, r = r,
                                         t_stat = ifelse(is.na(r), NA, round(t_stat,3)),
                                         p_value = ifelse(is.na(p), NA, round(p,3)))
    }
  }
  dplyr::bind_rows(out) |> arrange(p_value)
}
df_results_corr <- pointbiserial_all(df_final)
cat("=== Point-biserial (global estimation sample) ===\n")
print(df_results_corr)
print(df_results_corr |> arrange(desc(abs(r))))
print(df_results_corr |> arrange(desc(abs(t_stat))))
message("Sort by |t| and |r| should coincide.")

#%% 7) Four equivalent tests for 'tdta'
g0 <- df_final |> filter(yd == 0) |> pull(tdta)
g1 <- df_final |> filter(yd == 1) |> pull(tdta)
tt <- t.test(g0, g1, var.equal = FALSE)
cat(sprintf("T-test: t=%.3f, p=%.3f\n", tt$statistic, tt$p.value))

ct <- cor.test(df_final$yd, df_final$tdta, method = "pearson")
r  <- as.numeric(ct$estimate); p_val <- ct$p.value
cat(sprintf("Correlation test: r=%.3f, p=%.3f\n", r, p_val))
n <- sum(complete.cases(df_final$yd, df_final$tdta))
t_stat <- r * sqrt((n - 2)/(1 - r^2))
cat(sprintf("r = %.3f, t = %.3f, n = %d\n", r, t_stat, n))

model_an  <- lm(tdta ~ yd, data = df_final);  print(summary(model_an))
model_lpm <- lm(yd ~ tdta, data = df_final);  print(summary(model_lpm))

#%% 8) Bivariate correlations > 0.8 (absolute)
cols_over <- c()
all_cols <- names(df_final)
for (c1 in all_cols) {
  for (c2 in setdiff(all_cols, c("yd","dumVE","id"))) {
    if (c1 != c2 && is.numeric(df_final[[c1]]) && is.numeric(df_final[[c2]])) {
      corr <- suppressWarnings(cor(df_final[[c1]], df_final[[c2]], use = "complete.obs"))
      if (!is.na(corr) && abs(corr) > 0.8) {
        cat(sprintf("Correlation between %s and %s is %.3f\n", c1, c2, corr))
        cols_over <- unique(c(cols_over, c1))
      }
    }
  }
}
print(cols_over)
if (length(cols_over) >= 2 && has_GGally) {
  print(GGally::ggpairs(df_final |> dplyr::select(dplyr::all_of(cols_over))))
} else if (length(cols_over) >= 2 && !has_GGally) {
  message("GGally non dispo : matrice de corrélation affichée plus haut.")
}
df_corr <- df_final |> dplyr::select(dplyr::where(is.numeric)) |> cor(use = "pairwise.complete.obs")

#%% 10) Bivariate clouds for 6 variables
cor_col <- c("yd", "tdta", "reta", "opita", "ebita", "gempl")
if (has_GGally) {
  print(GGally::ggpairs(df_final |> dplyr::select(dplyr::all_of(intersect(cor_col, names(df_final))))))
} else {
  message("GGally non dispo : j'affiche la matrice de corrélation pour les colonnes choisies.")
  sel <- intersect(cor_col, names(df_final))
  print(round(cor(dplyr::select(df_final, dplyr::all_of(sel)), use = "pairwise.complete.obs"), 3))
}

#%% 11) OLS, Logit, Probit (one variable: tdta)
y <- df_final$yd
ols1   <- lm(yd ~ tdta, data = df_final)
logit1 <- glm(yd ~ tdta, data = df_final, family = binomial(link = "logit"))
probit1<- glm(yd ~ tdta, data = df_final, family = binomial(link = "probit"))

p_ols1    <- predict(ols1, type = "response")
p_logit1  <- predict(logit1, type = "response")
p_probit1 <- predict(probit1, type = "response")

cat("\n--- One-variable models (console) ---\n")
print(broom::tidy(ols1));   print(broom::glance(ols1))
print(broom::tidy(logit1)); print(broom::glance(logit1))
print(broom::tidy(probit1));print(broom::glance(probit1))

#%% 12) Concordant pairs (manual) + AUC equivalence (with Probit scores)
p_hat <- p_probit1; y_vec <- y
concordant <- 0; discordant <- 0; ties <- 0
n_y <- length(y_vec)
for (i in 1:(n_y - 1)) {
  for (j in (i + 1):n_y) if (y_vec[i] != y_vec[j]) {
    if (y_vec[i] == 1 && y_vec[j] == 0) {
      if (p_hat[i] > p_hat[j]) concordant <- concordant + 1
      else if (p_hat[i] < p_hat[j]) discordant <- discordant + 1
      else ties <- ties + 1
    } else {
      if (p_hat[j] > p_hat[i]) concordant <- concordant + 1
      else if (p_hat[j] < p_hat[i]) discordant <- discordant + 1
      else ties <- ties + 1
    }
  }
}
total_pairs <- concordant + discordant + ties
percent_concordant <- ifelse(total_pairs > 0, 100 * concordant / total_pairs, 0)
cat(sprintf("\nPercentage Concordance (manual): %.2f%%\n", percent_concordant))

roc_obj <- pROC::roc(response = y_vec, predictor = p_hat)
c_stat <- as.numeric(pROC::auc(roc_obj))
cat(sprintf("Percentage Concordance (via AUC): %.2f%%\n", 100 * c_stat))
cat("same results\n")

#%% 13) Multi-var models with preferred variables
keep_cols <- intersect(c("tdta","ebita","invsls"), names(df_final))
form3 <- as.formula(paste("yd ~", paste(keep_cols, collapse = " + ")))

ols3   <- lm(form3, data = df_final)
logit3 <- glm(form3, data = df_final, family = binomial(link = "logit"))
probit3<- glm(form3, data = df_final, family = binomial(link = "probit"))

p_ols    <- predict(ols3, type = "response")
p_logit  <- predict(logit3, type = "response")
p_probit <- predict(probit3, type = "response")

safe_auc <- function(y, s) {
  tryCatch(as.numeric(pROC::auc(pROC::roc(y, s))), error = function(e) NA_real_)
}
auc_ols    <- safe_auc(y, p_ols)
auc_logit  <- safe_auc(y, p_logit)
auc_probit <- safe_auc(y, p_probit)

cat("\n--- Multivariate models (console) ---\n")
print(broom::tidy(ols3));   print(broom::glance(ols3))
print(broom::tidy(logit3)); print(broom::glance(logit3))
print(broom::tidy(probit3));print(broom::glance(probit3))
cat(sprintf("\nAUC (OLS)=%.3f | AUC (Logit)=%.3f | AUC (Probit)=%.3f\n",
            auc_ols, auc_logit, auc_probit))

#%% ROC curves
plot_roc <- function(y, scores, label) {
  roc_obj <- pROC::roc(y, scores)
  dfp <- data.frame(
    fpr = 1 - roc_obj$specificities,
    tpr = roc_obj$sensitivities
  )
  ggplot(dfp, aes(x = fpr, y = tpr)) +
    geom_line() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(x = "False Positive Rate (1 - Specificity)",
         y = "True Positive Rate (Sensitivity)",
         title = paste("ROC Curve:", label),
         subtitle = sprintf("AUC = %.3f", as.numeric(pROC::auc(roc_obj)))) +
    theme_minimal()
}
print(plot_roc(y, p_ols,   "OLS"))
print(plot_roc(y, p_logit, "Logit"))
print(plot_roc(y, p_probit,"Probit"))

#%% Validation sample
n_0 <- nrow(df_val |> filter(yd == 0))
n_1 <- nrow(df_val |> filter(yd == 1))
cat(sprintf("\nNumber of defaulting firms %d vs non defaulting firms %d\n", n_0, n_1))

X_val_cols <- keep_cols
form3_val <- as.formula(paste("yd ~", paste(X_val_cols, collapse = " + ")))

ols3v   <- lm(form3_val, data = df_val)
logit3v <- glm(form3_val, data = df_val, family = binomial(link = "logit"))
probit3v<- glm(form3_val, data = df_val, family = binomial(link = "probit"))

y_v <- df_val$yd
p_olsv    <- predict(ols3v, type = "response")
p_logitv  <- predict(logit3v, type = "response")
p_probitv <- predict(probit3v, type = "response")

auc_olsv    <- safe_auc(y_v, p_olsv)
auc_logitv  <- safe_auc(y_v, p_logitv)
auc_probitv <- safe_auc(y_v, p_probitv)

cat("\n--- Validation models (console) ---\n")
print(broom::tidy(ols3v));   print(broom::glance(ols3v))
print(broom::tidy(logit3v)); print(broom::glance(logit3v))
print(broom::tidy(probit3v));print(broom::glance(probit3v))
cat(sprintf("\nAUC Val (OLS)=%.3f | (Logit)=%.3f | (Probit)=%.3f\n",
            auc_olsv, auc_logitv, auc_probitv))

# ROC plots (validation)
print(plot_roc(y_v, p_olsv,   "OLS (Validation)"))
print(plot_roc(y_v, p_logitv, "Logit (Validation)"))
print(plot_roc(y_v, p_probitv,"Probit (Validation)"))

#%% 15) Pearson residuals (per model)
p_olsv_clip <- pmin(pmax(p_olsv, 1e-6), 1 - 1e-6)
r_ols    <- (y_v - p_olsv_clip) / sqrt(p_olsv_clip * (1 - p_olsv_clip))
r_logit  <- residuals(logit3v,  type = "pearson")
r_probit <- residuals(probit3v, type = "pearson")

df_resid <- data.frame(
  Model = rep(c("OLS", "Logit", "Probit"), each = length(y_v)),
  Pearson_Residual = c(r_ols, r_logit, r_probit),
  yd = rep(y_v, times = 3)
)

ggplot(df_resid, aes(x = Pearson_Residual)) +
  geom_density(fill = NA) +
  facet_wrap(~ Model, scales = "free") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  coord_cartesian(xlim = c(-4, 4)) +
  labs(title = "Pearson Standardized Residuals", x = "Pearson Residual", y = "Density") +
  theme_minimal()

#%% 16) Residuals per group (yd)
ggplot(df_resid |> dplyr::filter(yd == 1), aes(x = Pearson_Residual)) +
  geom_density(fill = NA) +
  facet_wrap(~ Model, scales = "free") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  coord_cartesian(xlim = c(-4, 4)) +
  labs(title = "Pearson Residuals (yd=1)", x = "Pearson Residual", y = "Density") +
  theme_minimal()

ggplot(df_resid |> dplyr::filter(yd == 0), aes(x = Pearson_Residual)) +
  geom_density(fill = NA) +
  facet_wrap(~ Model, scales = "free") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  coord_cartesian(xlim = c(-4, 4)) +
  labs(title = "Pearson Residuals (yd=0)", x = "Pearson Residual", y = "Density") +
  theme_minimal()

#%% 17) Loss function — (write-up)

#%% 19) Dummy trap illustration
df_trap <- df_final |> mutate(ydn = as.integer(1 - yd))
model_dum <- lm(ydn ~ tdta, data = df_trap)
print(summary(model_dum))

#%% 20) Practical changes vs templates — (write-up)

#%% PART 2 — Random split using base R
set.seed(42)
idx <- sample(seq_len(nrow(df_dum)), size = floor(0.7 * nrow(df_dum)))
X_train_ran <- df_dum[idx, ]
X_test_ran  <- df_dum[-idx, ]
y_train_ran <- df_dum[idx, c("id","yd")]
y_test_ran  <- df_dum[-idx, c("id","yd")]
cat("\nRandom split done.\n")
