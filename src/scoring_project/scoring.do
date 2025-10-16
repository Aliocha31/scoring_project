*******************************************************
* scoring.do — Stata version of scoring.py
* Requirements: Stata 16+, dataset defaut2000.csv
* Converted from Python / sklearn / statsmodels workflow
*******************************************************

clear all
set more off
version 17

*******************************************************
* 0) Parameters & options
*******************************************************
global CSV_PATH      "defaut2000.csv"
global SEP           ";"
global MISSVAL_CODE  -99.99
global OUTDIR        "results"
cap mkdir "$OUTDIR"

*******************************************************
* 1) Import CSV (semicolon-separated) + handle missing values
*******************************************************
import delimited using "$CSV_PATH", delimiter("$SEP") varnames(1) case(preserve) ///
    bindquote(strict) encoding("UTF-8") stringcols(_all) clear

* Try to convert string variables into numeric when possible
foreach v of varlist _all {
    cap destring `v', replace force
}

* Replace coded missing values (-99.99) with Stata missing (.)
foreach v of varlist _all {
    capture confirm numeric variable `v'
    if !_rc {
        quietly replace `v' = . if `v' == $MISSVAL_CODE
    }
}

* Check that the binary target exists
capture confirm variable yd
if _rc {
    di as error "Target column 'yd' not found."
    exit 198
}

*******************************************************
* 2) Create Estimation / Validation split (dumVE)
*    Alternate rows even/odd to mimic deterministic hold-out
*    Note: do NOT sort before splitting to avoid bias
*******************************************************
gen long id = _n - 1
gen byte dumVE = mod(id, 2)==0    // 0 = Estimation, 1 = Validation

*******************************************************
* 3) Summary statistics by class (yd)
*******************************************************
di as txt "===== Descriptive statistics by yd ====="
quietly levelsof yd, local(yvals)
foreach y of local yvals {
    di as txt "---- yd == `y' ----"
    quietly ds yd dumVE id
    local numvars : varlist _all - yd dumVE id
    quietly tabstat `numvars', by(yd) stat(n mean sd min p25 p50 p75 max skewness kurtosis) columns(statistics)
}

*******************************************************
* 4) Histograms & boxplots by class
*******************************************************
capture graph drop _all
quietly ds yd dumVE id
local plotvars : varlist _all - yd dumVE id
foreach v of local plotvars {
    * Histogram per class
    quietly histogram `v', discrete if !missing(`v') , ///
        by(yd, note("") title("Histogram: `v' by yd")) name(h_`v', replace)
    * Boxplot per class
    quietly graph box `v', over(yd) title("Boxplot: `v' by yd") name(b_`v', replace)
}

*******************************************************
* 5) Create Estimation (dumVE==0) and Validation (dumVE==1) datasets
*******************************************************
preserve
keep if dumVE==0
tempfile est
save `est'
restore

preserve
keep if dumVE==1
tempfile val
save `val'
restore

*******************************************************
* 6) Point-biserial correlation (yd vs numeric variables)
*    (Standard correlation since yd ∈ {0,1})
*******************************************************
use `est', clear
ds yd dumVE id, not
local numvars `r(varlist)'

tempname corrmat
matrix define `corrmat' = J(0,4,.)
* Columns: variable | r | t_stat | p_value

postutil clear
tempname holder
postfile `holder' str32 variable double r double t_stat double p_value using "$OUTDIR/pointbiserial_est.dta", replace

foreach x of local numvars {
    quietly corr `x' yd if !missing(`x', yd), cov sig
    matrix R = r(C)
    scalar rxy = R[1,2]
    scalar n   = r(N)
    * Compute t-statistic of r: t = r * sqrt((n-2)/(1-r^2))
    if (abs(rxy)<1) & (n>=3) {
        scalar tstat = rxy * sqrt((n-2)/(1-rxy^2))
        scalar pval = 2*ttail(n-2, abs(tstat))
    }
    else {
        scalar tstat = .
        scalar pval  = .
    }
    post `holder' ("`x'") (rxy) (tstat) (pval)
}
postclose `holder'

use "$OUTDIR/pointbiserial_est.dta", clear
sort p_value
export delimited using "$OUTDIR/pointbiserial_est.csv", replace

*******************************************************
* 7) Tests on variable tdta (t-test, corr, regressions)
*******************************************************
use `est', clear

capture confirm variable tdta
if _rc {
    di as err "Variable 'tdta' not found — skipping section 7."
}
else {
    * t-test of tdta across yd classes
    ttest tdta, by(yd)

    * Simple correlation
    corr tdta yd, sig

    * ANOVA-style OLS (tdta ~ yd)
    regress tdta i.yd
    estimates store m_anova

    * Linear Probability Model (yd ~ tdta)
    regress yd tdta
    estimates store m_lpm

    * Logit and Probit (yd ~ tdta)
    logit  yd tdta
    estimates store m_logit
    predict phat_logit if e(sample), pr

    probit yd tdta
    estimates store m_probit
    predict phat_probit if e(sample), pr

    * Export results table if esttab is available
    cap which esttab
    if !_rc {
        esttab m_lpm m_logit m_probit using "$OUTDIR/reg_est.rtf", replace ///
            se r2 aic bic stats(N, fmt(%9.0g) labels("N")) ///
            title("OLS / Logit / Probit (Estimation sample)")
    }
}

*******************************************************
* 8) ROC & AUC (OLS, Logit, Probit)
****
