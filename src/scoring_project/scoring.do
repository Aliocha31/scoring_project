*******************************************************
* Default scoring — Traduction complète Python -> Stata (corrigée)
* Requis : Stata 16+ ; (optionnel) estout pour LaTeX : ssc install estout
*******************************************************

clear all
set more off
version 16

*******************************************************
* 0) Import & nettoyage (CSV ; ;  -99.99 -> .)
*******************************************************
import delimited using "defaut2000.csv", delimiter(";") varnames(1) case(preserve) clear

* Remplacer -99.99 par manquant pour toutes les numériques
quietly ds, has(type numeric)
foreach v of varlist `r(varlist)' {
    quietly replace `v' = . if `v' == -99.99
}

* Drop lignes avec au moins un manquant (≈ drop_nans)
egen __rowmiss = rowmiss(_all)
drop if __rowmiss > 0
drop __rowmiss

*******************************************************
* 1) Tri & split impairs/pairs (comme Python)
*******************************************************
capture confirm variable reta
if _rc==0 {
    sort yd reta
}
else {
    sort yd
}

gen long id   = _n - 1
gen byte dumVE = mod(id, 2)==0     // 0 -> impair (Estimation), 1 -> pair (Validation)

preserve
    keep if dumVE==0
    tempfile TRAIN
    save `TRAIN', replace
restore

preserve
    keep if dumVE==1
    tempfile VALID
    save `VALID', replace
restore

*******************************************************
* 2) Descriptifs par classe yd
*******************************************************
use `TRAIN', clear

* Variables numériques hors yd id dumVE
ds yd id dumVE, not
local others `r(varlist)'
ds `others', has(type numeric)
local numvars `r(varlist)'

levelsof yd, local(yvals)
foreach y of local yvals {
    di as text "---- Descriptive stats for yd = `y' ----"
    foreach v of varlist `numvars' {
        quietly summarize `v' if yd==`y'
        di as res "`v' : N=" %9.0f r(N) "  mean=" %9.4f r(mean) "  sd=" %9.4f r(sd)
    }
}

*******************************************************
* 3) Histogrammes & boxplots (exemple sur tdta si dispo)
*******************************************************
capture confirm variable tdta
if _rc==0 {
    histogram tdta, by(yd, note("") legend(off)) fraction title("tdta by yd")
    graph box tdta, over(yd) title("Boxplot tdta by yd")
}

*******************************************************
* 4) Skewness & Kurtosis par groupe (table)
*******************************************************
ds yd id dumVE, not
local others `r(varlist)'
ds `others', has(type numeric)
local numvars `r(varlist)'

tempname memSK
tempfile SKTAB
postfile `memSK' str32 Variable byte Group double Skewness double Kurtosis using `SKTAB', replace

levelsof yd, local(yvals)
foreach g of local yvals {
    foreach v of varlist `numvars' {
        quietly summarize `v' if yd==`g', detail
        post `memSK' ("`v'") (`g') (r(skewness)) (r(kurtosis))
    }
}
postclose `memSK'
use `SKTAB', clear
sort Variable Group
* list in 1/20

*******************************************************
* 5) Normalité + égalité des variances (Shapiro -> Bartlett/Levene)
*******************************************************
use `TRAIN', clear
ds yd id dumVE, not
local others `r(varlist)'
ds `others', has(type numeric)
local numvars `r(varlist)'

tempname memHOV
tempfile HOVTAB
postfile `memHOV' str32 Variable str10 test double p_shap0 double p_shap1 ///
                 byte normal_both double pvalue byte equal_variances using `HOVTAB', replace

foreach v of varlist `numvars' {
    * Shapiro par groupe
    capture noisily swilk `v' if yd==0
    scalar p0 = cond(c(rc)==0, r(p), .)
    capture noisily swilk `v' if yd==1
    scalar p1 = cond(c(rc)==0, r(p), .)
    scalar both = (p0>0.05 & p1>0.05)

    if (both) {
        capture noisily bartlett `v', by(yd)
        scalar p = cond(c(rc)==0, r(p), .)
        post `memHOV' ("`v'") ("bartlett") (p0) (p1) (both) (p) (p>=0.05)
    }
    else {
        capture noisily robvar `v', by(yd) w0
        scalar p = cond(c(rc)==0, r(p), .)
        post `memHOV' ("`v'") ("levene") (p0) (p1) (both) (p) (p>=0.05)
    }
}
postclose `memHOV'
use `HOVTAB', clear
gsort +equal_variances +pvalue
* list in 1/20

*******************************************************
* 6) Tests d'égalité des moyennes (Welch sauf vars connues égales)
*******************************************************
use `TRAIN', clear
ds yd id dumVE, not
local others `r(varlist)'
ds `others', has(type numeric)
local numvars `r(varlist)'

local equalvars tdta reta nwcta

tempname memTT
tempfile TTTAB
postfile `memTT' str32 Variable double t_stat double p_value using `TTTAB', replace

foreach v of varlist `numvars' {
    capture noisily ttest `v', by(yd) unequal
    if strpos(" `equalvars' ", " `v' ") {
        capture noisily ttest `v', by(yd) equal
    }
    post `memTT' ("`v'") (cond(c(rc)==0, r(t), .)) (cond(c(rc)==0, r(p), .))
}
postclose `memTT'
use `TTTAB', clear
gsort +p_value
* list in 1/20

*******************************************************
* 7) Corrélation point-bisérielle (Pearson avec yd binaire)
*******************************************************
use `TRAIN', clear
ds yd id dumVE, not
local others `r(varlist)'
ds `others', has(type numeric)
local numvars `r(varlist)'

tempname memPB
tempfile PBTAB
postfile `memPB' str32 Variable double r double t_stat double p_value using `PBTAB', replace

foreach v of varlist `numvars' {
    capture noisily corr `v' yd
    if (c(rc)==0) {
        matrix C = r(C)
        scalar rr = C[1,2]
        scalar n  = r(N)
        scalar t  = rr*sqrt((n-2)/(1-rr^2))
        scalar p  = 2*ttail(n-2, abs(t))
        post `memPB' ("`v'") (rr) (t) (p)
    }
    else {
        post `memPB' ("`v'") (.) (.) (.)
    }
}
postclose `memPB'
use `PBTAB', clear
gsort +p_value
* list in 1/20

*******************************************************
* 8) Corrélations fortes & paires (approx)
*******************************************************
use `TRAIN', clear
ds yd id dumVE, not
local others `r(varlist)'
ds `others', has(type numeric)
local numvars `r(varlist)'

corr `numvars'
graph matrix `numvars', half title("Pairs of numeric variables (train)")

*******************************************************
* 9) Modèles 1 variable (tdta) : OLS, Logit, Probit + ROC
*******************************************************
use `TRAIN', clear
capture confirm variable tdta
if _rc==0 {
    regress yd tdta
    estimates store OLS1

    logit yd tdta
    estimates store LOG1
    predict p_logit1 if e(sample), pr

    probit yd tdta
    estimates store PROB1
    predict p_probit1 if e(sample), pr

    * ROC/AUC pour logit
    lroc
}

*******************************************************
* 10) Modèles multivariés préférés (tdta gempl opita invsls)
*******************************************************
use `TRAIN', clear
capture noisily regress yd tdta gempl opita invsls
estimates store OLS2

capture noisily logit   yd tdta gempl opita invsls
estimates store LOG2
capture noisily predict p_logit2 if e(sample), pr

capture noisily probit  yd tdta gempl opita invsls
estimates store PROB2
capture noisily predict p_probit2 if e(sample), pr

*******************************************************
* 11) Leverage (hat) & outliers (règle 2p/n) avec OLS
*******************************************************
capture noisily regress yd tdta gempl opita invsls
predict hat, hat
summarize hat
local p = e(df_m) + 1
local n = e(N)
local thr = 2*`p'/`n'
gen byte high_leverage = hat > `thr'
di as text "n=" `n' " p=" `p' " threshold(2p/n)=" %6.4f `thr'
list id hat in 1/10 if high_leverage==1, noobs sepby(high_leverage)

*******************************************************
* 12) Concordance « faite à la main » vs AUC (Probit préférés)
*******************************************************
probit yd tdta gempl opita invsls
predict phat, pr

quietly lroc
scalar auc_lroc = r(area)

* --- MATA: fonction concordance propre ---
mata:
real scalar concordance(real colvector y, real colvector p)
{
    real scalar n, conc, disc, ties, i, j, tot;
    n = rows(y);
    conc = 0; disc = 0; ties = 0;
    for (i=1; i<=n-1; i++) {
        for (j=i+1; j<=n; j++) {
            if (y[i]!=y[j]) {
                if (y[i]==1 & y[j]==0) {
                    if (p[i]>p[j])      conc = conc + 1;
                    else if (p[i]<p[j]) disc = disc + 1;
                    else                ties = ties + 1;
                }
                else if (y[i]==0 & y[j]==1) {
                    if (p[j]>p[i])      conc = conc + 1;
                    else if (p[j]<p[i]) disc = disc + 1;
                    else                ties = ties + 1;
                }
            }
        }
    }
    tot = conc + disc + ties;
    if (tot==0) return(.);
    return(100*conc/tot);
}
end

mata: st_numscalar("pc_manual", concordance(st_data(., "yd"), st_data(., "phat")))
display as text "Percent Concordance (manual) = " %6.2f scalar(pc_manual) "%"
display as text "AUC from lroc * 100        = " %6.2f (scalar(auc_lroc)*100) "%"

*******************************************************
* 13) Résidus de Pearson (GLM), densités par groupe & scatter vs p_hat
*******************************************************
glm yd tdta gempl opita invsls, family(binomial) link(logit)
predict phat_glm, mu
* >>> CORRECTION : Pearson residuals <<<
predict rpear, pearson

kdensity rpear, title("Pearson residuals (GLM Logit)")

twoway (kdensity rpear if yd==0) (kdensity rpear if yd==1), ///
       legend(order(1 "yd=0" 2 "yd=1")) ///
       title("Pearson residuals by group (train)")

gen byte outlier = abs(rpear)>2
twoway (scatter rpear phat_glm if yd==0, mcolor(blue)) ///
       (scatter rpear phat_glm if yd==1, mcolor(red))  ///
       (scatter rpear phat_glm if outlier, msymbol(Oh) mcolor(none) mlcolor(red) msize(large)) ///
       , yline(0 -2 2, lpattern(dash) lcolor(black*0.6)) ///
         legend(order(1 "yd=0" 2 "yd=1" 3 "outliers |r|>2")) ///
         title("Pearson residuals vs predicted probability (logit)")

*******************************************************
* 14) Validation : refit sur l'échantillon de validation
*******************************************************
use `VALID', clear

capture noisily regress yd tdta gempl opita invsls
capture noisily logit   yd tdta gempl opita invsls
capture noisily probit  yd tdta gempl opita invsls

* ROC exemple (logit)
capture noisily predict p_val, pr
capture noisily lroc

*******************************************************
* 15) Pseudo-R² de McFadden (logit/probit)
*******************************************************
use `TRAIN', clear
logit  yd tdta gempl opita invsls
display "McFadden R2 (logit) = " 1 - e(ll)/e(ll_0)

probit yd tdta gempl opita invsls
display "McFadden R2 (probit)= " 1 - e(ll)/e(ll_0)

*******************************************************
* 16) Sauvegardes .dta (train/val)
*******************************************************
use `TRAIN', clear
save "defaut2000_train.dta", replace
use `VALID', clear
save "defaut2000_val.dta", replace

*******************************************************
* 17) (Option) Exports LaTeX via esttab
*******************************************************
* eststo clear
* use `TRAIN', clear
* regress yd tdta gempl opita invsls
* eststo OLS
* logit  yd tdta gempl opita invsls
* eststo LOG
* probit yd tdta gempl opita invsls
* eststo PROB
* esttab OLS LOG PROB using "results/regression_table.tex", replace ///
*       se star(* 0.10 ** 0.05 *** 0.01) label booktabs title("OLS/Logit/Probit")

*******************************************************
* 18) Fonction de perte & seuil coût-sensible (Validation)
*******************************************************
scalar C_FN = 10   // prêter à un futur défaillant (Type 1 / FN)
scalar C_FP = 1    // refuser un emprunteur sain (Type 2 / FP)

use `VALID', clear
logit yd tdta gempl opita invsls
predict phat_val, pr

scalar t_star = C_FP/(C_FN + C_FP)
display "Theoretical threshold t* = " %6.4f t_star

gen byte yhat_t = (phat_val >= t_star)
tab yd yhat_t, matcell(M)
scalar TN = M[1,1]
scalar FP = M[1,2]
scalar FN = M[2,1]
scalar TP = M[2,2]
scalar N  = TN + FP + FN + TP
scalar COST_tstar = (C_FN*FN + C_FP*FP)/N
display "Empirical average cost at t*  = " %9.5f COST_tstar

tempname memC
tempfile COSTGRID
postfile `memC' double thr double cost using `COSTGRID', replace

forvalues i = 1/199 {
    scalar thr = `i'/200
    gen byte yhat_tmp = (phat_val >= thr)
    quietly tab yd yhat_tmp, matcell(Mg)
    drop yhat_tmp
    scalar TNg = Mg[1,1]
    scalar FPg = Mg[1,2]
    scalar FNg = Mg[2,1]
    scalar TPg = Mg[2,2]
    scalar Ng  = TNg + FPg + FNg + TPg
    scalar COSTg = (C_FN*FNg + C_FP*FPg)/Ng
    post `memC' (thr) (COSTg)
}
postclose `memC'
use `COSTGRID', clear
gsort +cost
list in 1/5, abbrev(16)

sum thr if cost == cost[1]
scalar t_emp = r(mean)
display "Best empirical threshold (min cost) t_emp = " %6.4f t_emp " ; cost = " %9.5f cost[1]
display "Δ cost (t_emp - t*) = " %9.5f (cost[1] - COST_tstar)

twoway line cost thr, sort ytitle("Average cost") xtitle("Threshold") ///
       title("Cost curve vs threshold (validation)") yline(`=COST_tstar', lpattern(dash))

*******************************************************
* 19) Dummy trap (colinéarité parfaite)
*******************************************************
use `TRAIN', clear
gen byte ynd = 1 - yd

regress tdta yd ynd           // colinéarité -> Stata droppe une variable
regress tdta yd
regress tdta ynd

gen double z = yd - ynd
regress tdta z
matrix list e(b)

constraint define 1 _b[yd] + _b[ynd] = 0
cnsreg tdta yd ynd, constraints(1)

*******************************************************
di as text "Done."
