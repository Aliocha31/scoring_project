/*==============================================================
  scoring.sas — SAS translation of the Python scoring pipeline
  Goals:
   - Deterministic hold-out split (Estimation/Validation)
   - Preprocessing: median imputation + standardization (train-only params)
   - Models: Logistic Regression (base). Optional: HPFOREST if available
   - Validation metrics: ROC AUC, PR AUC (approx), F1/ACC @0.5 and @best F1
   - Leaderboard + save best model
================================================================*/

options nodate nonumber;
ods listing; ods graphics on;

/* ------------------------------
   0) Config
--------------------------------*/
%let csv_path   = defaut2000.csv;
%let sep        = %str(;);
%let miss_code  = -99.99;
%let outdir     = results;
%let target     = yd;
/* Feature list (like Python FEATURES) */
%let features   = ebita opita reta gempl;

/* Create output dir (Windows/Linux safe) */
options noxwait noxsync;
%sysexec mkdir &outdir;

/* ------------------------------
   1) Import CSV ; handle missing codes
--------------------------------*/
proc import datafile="&csv_path" out=work.raw dbms=dlm replace;
  delimiter="&sep";
  guessingrows=max;
  getnames=yes;
run;

/* Replace custom missing code for numeric variables */
proc contents data=work.raw out=_vars(keep=name type) noprint; run;

data work.raw2;
  set work.raw;
  array _n_vars _numeric_;
  do _i=1 to dim(_n_vars);
    if _n_vars[_i] = &miss_code then _n_vars[_i] = .;
  end;
  drop _i;
run;

/* Ensure target exists and is binary int */
%macro assert_target;
  %local dsid varnum rc;
  %let dsid=%sysfunc(open(work.raw2,i));
  %let varnum=%sysfunc(varnum(&dsid,&target));
  %if &varnum=0 %then %do;
    %put ERROR: Target &target not found.;
    %abort cancel;
  %end;
  %let rc=%sysfunc(close(&dsid));
%mend; %assert_target

/* Force integer 0/1 if needed */
data work.raw2;
  set work.raw2;
  if not missing(&target) then &target = ( &target ne 0 );
run;

/* ------------------------------
   2) Hold-out split (no sorting): dumVE
      even row -> 0 (Estimation), odd -> 1 (Validation)
--------------------------------*/
data work.base;
  set work.raw2;
  id = _n_-1;
  dumVE = (mod(id,2)=1);  /* 0: Estimation, 1: Validation */
run;

/* ------------------------------
   3) Build Estimation/Validation datasets
--------------------------------*/
data work.est work.val;
  set work.base;
  if dumVE=0 then output work.est;
  else output work.val;
run;

/* Keep only rows with non-missing target */
data work.est; set work.est; if not missing(&target); run;
data work.val; set work.val; if not missing(&target); run;

/* ------------------------------
   4) Separate numeric and categorical features
      (use PROC CONTENTS to derive types)
--------------------------------*/
proc contents data=work.base out=_meta noprint; run;

proc sql noprint;
  /* keep only selected FEATURES present in the data */
  create table _feat as
  select lowcase(name) as name,
         type
  from _meta
  where lowcase(name) in (%sysfunc(tranwrd(%sysfunc(lowcase(&features)),%str( ),%str(,")))) )
  ;
quit;

/* Build macro lists num_features / cat_features */
%macro build_lists;
  %global num_features cat_features;
  %let num_features=;
  %let cat_features=;
  data _null_;
    set _feat end=last;
    length accumn accumc $32000;
    retain accumn accumc "";
    if type=1 then accumn=catx(' ',accumn, name);
    else if type=2 then accumc=catx(' ',accumc, name);
    if last then do;
      call symputx('num_features',accumn);
      call symputx('cat_features',accumc);
    end;
  run;
%mend; %build_lists

%put NOTE: Numeric = &num_features;
%put NOTE: Categorical = &cat_features;

/* ------------------------------
   5) Preprocessing on Estimation:
      - Median imputation for numeric
      - Standardization (mean=0, std=1)
      Apply SAME parameters to Validation
--------------------------------*/

/* 5.1 Median imputation (train) + capture stats */
%let numlist=&num_features;
%if %length(&numlist)>0 %then %do;
  proc stdize data=work.est out=work.est_numimp method=median reponly outstat=work._meds;
    var &numlist;
  run;
%end;
%else %do; data work.est_numimp; set work.est; run; %end;

/* 5.2 Standardize with train parameters (mean 0 / std 1) */
%if %length(&numlist)>0 %then %do;
  /* Compute means/std on the already imputed train data */
  proc stdize data=work.est_numimp out=work.est_std method=std outstat=work._stdpars;
    var &numlist;
  run;
%end;
%else %do; data work.est_std; set work.est_numimp; run; %end;

/* -- For Validation: apply SAME imputation and scaling params from train -- */
%if %length(&numlist)>0 %then %do;
  /* Impute using train medians */
  proc stdize data=work.val out=work.val_numimp method=in(_meds) reponly;
    var &numlist;
  run;
  /* Standardize using train mean/std */
  proc stdize data=work.val_numimp out=work.val_std method=in(_stdpars);
    var &numlist;
  run;
%end;
%else %do; data work.val_std; set work.val; run; %end;

/* Treat missing levels of categoricals as a valid level (like one-hot with 'Missing') */
%macro fill_cat(ds);
  %if %length(&cat_features)>0 %then %do;
    data &ds;
      set &ds;
      %local i tok;
      %let i=1; %let tok=%scan(&cat_features,&i,%str( ));
      %do %while(%length(&tok));
        length &tok $256;
        if missing(&tok) then &tok = "Missing";
        %let i=%eval(&i+1);
        %let tok=%scan(&cat_features,&i,%str( ));
      %end;
    run;
  %end;
%mend;
%fill_cat(work.est_std)
%fill_cat(work.val_std)

/* ------------------------------
   6) Train models on Estimation
      A) Logistic Regression (base SAS)
      (Optional) B) Random Forest (HPFOREST) if available
--------------------------------*/

/* A) Logistic Regression */
ods select none;
proc logistic data=work.est_std plots(only)=none;
  %if %length(&cat_features)>0 %then %do;
    class &cat_features / param=glm ref=first MISSING;
  %end;
  model &target(event='1') = &num_features &cat_features;
  /* store model for scoring */
  store out=work.store_logit;
  /* score on validation immediately */
  score data=work.val_std out=work.val_sc_logit(rename=(p_1=phat_logit)) outroc=work.roc_logit;
run;
ods select all;

/* (Optional) B) Random Forest (requires SAS High-Performance) */
%macro try_hpforest;
  %local available;
  %let available=0;
  /* crude probe: try to compile HPFOREST silently */
  ods exclude all; ods noresults;
  proc hpforest data=work.est_std ntrees=300 maxtrees=300 seed=42; target &target / level=interval; run; quit;
  %let available = 1;
  ods results; ods exclude none;
%mend;
%macro train_hpforest;
  %try_hpforest
  %if &available = 1 %then %do;
    %put NOTE: HPFOREST available — training Random Forest.;
    proc hpforest data=work.est_std seed=42 ntrees=600 maxtrees=600 
                  leafsize=1 vars_to_try=.; /* defaults similar to sklearn */
      /* Categorical handled automatically if character */
      target &target / level=interval;
      input &num_features / level=interval;
      %if %length(&cat_features)>0 %then %do;
        input &cat_features / level=nominal;
      %end;
      /* score on validation */
      score data=work.val_std out=work.val_sc_rf;
    run; quit;

    /* Conform predicted prob name to phat_rf */
    data work.val_sc_rf;
      set work.val_sc_rf;
      /* HPFOREST creates P_target1; harmonize to phat_rf */
      phat_rf = p_&target.1;
    run;
  %end;
  %else %do;
    %put WARNING: PROC HPFOREST not available — skipping Random Forest.;
  %end;
%mend; %train_hpforest

/* ------------------------------
   7) Metrics on Validation
      - Common utility macros
--------------------------------*/

/* Utility: build confusion matrix + F1/Accuracy at a given threshold */
%macro cm_metrics(in=, score=, thr=, out=);
data _pred_;
  set &in;
  p = &score;
  y = &target;
  yhat = (p >= &thr);
run;

proc sql noprint;
  /* counts */
  select sum(y=0 & yhat=0), sum(y=0 & yhat=1),
         sum(y=1 & yhat=0), sum(y=1 & yhat=1)
    into :TN, :FP, :FN, :TP
  from _pred_;
quit;

%let TN=%sysfunc(coalescec(&TN,0));
%let FP=%sysfunc(coalescec(&FP,0));
%let FN=%sysfunc(coalescec(&FN,0));
%let TP=%sysfunc(coalescec(&TP,0));

/* Precision/Recall/F1/Accuracy */
%let precision = %sysevalf(&TP / %sysfunc(max(1,%eval(&TP+&FP))));
%let recall    = %sysevalf(&TP / %sysfunc(max(1,%eval(&TP+&FN))));
%let f1        = %sysevalf(2*&precision*&recall / %sysfunc(max(1e-12,%sysevalf(&precision+&recall))));
%let acc       = %sysevalf(%sysevalf((&TP+&TN)) / %sysfunc(max(1,%eval(&TP+&TN+&FP+&FN))));

data &out;
  length model $24;
  model = ""; /* fill later */
  threshold = &thr;
  f1 = &f1; accuracy = &acc;
  TN=&TN; FP=&FP; FN=&FN; TP=&TP;
run;

proc datasets lib=work nolist; delete _pred_; quit;
%mend;

/* Utility: ROC AUC from a scored table (uses PROC LOGISTIC just to compute AUC) */
%macro auc_from_scores(in=, score=, outroc=, outaucds=);
ods select none;
proc logistic data=&in plots(only)=none;
  model &target(event='1') = &score / nofit; /* trick to get ROC on a single score */
  score data=&in out=__tmp__ outroc=&outroc;
run;
ods select all;

/* Pull AUC from the OUTROC (contains _ROC_ data; AUC in &SYSLAST? safer: re-run with association) */
ods select none;
ods output Association=__assoc__;
proc logistic data=&in;
  model &target(event='1') = &score / nofit;
  roc 'score' &score;
run;
ods select all;

data &outaucds;
  set __assoc__;
  where Label2 = "c";
  /* c = concordance index; AUC = c */
  keep nValue2;
  rename nValue2 = auc;
run;

proc datasets lib=work nolist; delete __tmp__ __assoc__; quit;
%mend;

/* Utility: Precision-Recall curve + AP (approx) + best F1 over all unique thresholds */
%macro pr_f1_best(in=, score=, outbest=, outpr=);
proc sort data=&in out=_s; by descending &score; run;

/* Compute cumulative TP/FP as threshold moves down */
data _pr;
  set _s end=last;
  by descending &score;
  retain tp fp pprev nprev;
  if _n_=1 then do; tp=0; fp=0; end;
  if &target=1 then tp+1; else fp+1;

  /* At each distinct threshold (score), output a point */
  if first.&score or last then do;
    thresh = &score;
    /* totals needed for recall denominator */
    /* Compute totals once (lazy): we'll merge later */
    output;
  end;
run;

/* Totals of positives/negatives */
proc sql noprint;
  select sum(&target=1), sum(&target=0) into :P, :N from &in;
quit;

/* Compute precision, recall; F1; accumulate AP (interp as step-wise) */
data _pr2;
  set _pr;
  retain prev_recall 0 ap 0 best_f1 0 best_t .;
  P = &P; N = &N;
  recall    = tp / max(1,P);
  precision = tp / max(1,(tp+fp));
  f1 = ifn(precision+recall>0, 2*precision*recall/(precision+recall), 0);

  /* Average Precision (step-wise in recall space) */
  ap + precision * (recall - prev_recall);
  prev_recall = recall;

  /* Track best F1 and threshold */
  if f1 > best_f1 then do; best_f1 = f1; best_t = thresh; end;
run;

/* Export PR points (optional) */
data &outpr; set _pr2; run;

/* Single-row dataset with best F1 threshold + AP */
proc sql;
  create table &outbest as
  select best_t as best_threshold,
         best_f1 as best_f1,
         max(ap) as pr_auc
  from _pr2;
quit;

proc datasets lib=work nolist; delete _s _pr _pr2; quit;
%mend;

/* ------------------------------
   8) Evaluate models on Validation
--------------------------------*/

/* 8.A) Logistic Regression metrics */
%auc_from_scores(in=work.val_sc_logit, score=phat_logit, outroc=work.rocpts_logit, outaucds=work.auc_logit)
%pr_f1_best(in=work.val_sc_logit, score=phat_logit, outbest=work.best_logit, outpr=work.pr_logit)

/* F1/ACC at 0.5 and at best threshold */
%cm_metrics(in=work.val_sc_logit, score=phat_logit, thr=0.5, out=work.cm05_logit)
data work.cm05_logit; set work.cm05_logit; model="logit"; run;

data _bt; set work.best_logit; call symputx('t_logit',best_threshold); run;
%cm_metrics(in=work.val_sc_logit, score=phat_logit, thr=&t_logit, out=work.cmbest_logit)
data work.cmbest_logit; set work.cmbest_logit; model="logit"; run;

/* 8.B) Random Forest (only if table exists) */
%macro eval_rf;
  %if %sysfunc(exist(work.val_sc_rf)) %then %do;
    %auc_from_scores(in=work.val_sc_rf, score=phat_rf, outroc=work.rocpts_rf, outaucds=work.auc_rf)
    %pr_f1_best(in=work.val_sc_rf, score=phat_rf, outbest=work.best_rf, outpr=work.pr_rf)

    %cm_metrics(in=work.val_sc_rf, score=phat_rf, thr=0.5, out=work.cm05_rf)
    data work.cm05_rf; set work.cm05_rf; model="rf"; run;

    data _btrf; set work.best_rf; call symputx('t_rf',best_threshold); run;
    %cm_metrics(in=work.val_sc_rf, score=phat_rf, thr=&t_rf, out=work.cmbest_rf)
    data work.cmbest_rf; set work.cmbest_rf; model="rf"; run;
  %end;
%mend; %eval_rf

/* ------------------------------
   9) Leaderboard assembly
--------------------------------*/
data work.lb_part;
  length model $24;
  /* Logistic */
  if _n_=1 then do;
    set work.auc_logit(rename=(auc=val_roc_auc)) nobs=n1;
    set work.best_logit(rename=(pr_auc=val_pr_auc best_f1=val_f1_best best_threshold=best_threshold_val));
    set work.cm05_logit(keep=f1 accuracy rename=(f1=val_f1_05 accuracy=val_acc_05));
    model='logit'; output;
  end;
run;

%macro add_rf_to_lb;
  %if %sysfunc(exist(work.auc_rf)) %then %do;
    data work.lb_rf;
      length model $24;
      set work.auc_rf(rename=(auc=val_roc_auc));
      set work.best_rf(rename=(pr_auc=val_pr_auc best_f1=val_f1_best best_threshold=best_threshold_val));
      set work.cm05_rf(keep=f1 accuracy rename=(f1=val_f1_05 accuracy=val_acc_05));
      model='rf';
    run;
    data work.lb_all; set work.lb_part work.lb_rf; run;
  %end;
  %else %do;
    data work.lb_all; set work.lb_part; run;
  %end;
%mend; %add_rf_to_lb

proc sort data=work.lb_all out=work.leaderboard;
  by descending val_roc_auc;
run;

proc print data=work.leaderboard label noobs;
  title "Leaderboard (sorted by Validation ROC AUC)";
  label val_roc_auc = "Val ROC AUC"
        val_pr_auc  = "Val PR AUC"
        val_f1_05   = "F1 @0.5"
        val_acc_05  = "ACC @0.5"
        val_f1_best = "F1 @best_t"
        best_threshold_val = "best_t";
run;
title;

/* Save CSV */
proc export data=work.leaderboard outfile="&outdir./leaderboard.csv" dbms=csv replace; run;

/* ------------------------------
   10) Save the best model (only models we trained/stored)
       Here: Logistic was stored; RF model persistence varies by product
--------------------------------*/
data _null_;
  set work.leaderboard(obs=1);
  call symputx('best_model', model);
run;

%put NOTE: Best model on Validation = &best_model;

%macro save_best;
  %if "&best_model" = "logit" %then %do;
    /* LOGISTIC stored earlier as work.store_logit */
    /* Save to a permanent item store */
    proc plm restore=work.store_logit;
      /* Store to file-like catalog */
      show all;
      code file="&outdir./best_model_logit.sas";
    quit;
    %put NOTE: Saved best model scoring code to &outdir./best_model_logit.sas ;
  %end;
  %else %do;
    %put NOTE: No persistent store implemented for &best_model (skipped).;
  %end;
%mend; %save_best

/* ------------------------------
   11) Quick report: thresholds and metrics
--------------------------------*/
%macro dump_model_block(name=);
  %if "&name"="logit" %then %do;
    data _null_; set work.best_logit; put "LOGIT best_t=" best_threshold 8.6 " best_F1=" best_f1 8.6 " PR_AUC=" pr_auc 8.6; run;
    data _null_; set work.cm05_logit; put "LOGIT @0.5  F1=" f1 8.6 " ACC=" accuracy 8.6; run;
  %end;
  %else %if "&name"="rf" %then %do;
    %if %sysfunc(exist(work.best_rf)) %then %do;
      data _null_; set work.best_rf; put "RF best_t=" best_threshold 8.6 " best_F1=" best_f1 8.6 " PR_AUC=" pr_auc 8.6; run;
      data _null_; set work.cm05_rf; put "RF @0.5  F1=" f1 8.6 " ACC=" accuracy 8.6; run;
    %end;
  %end;
%mend;
%dump_model_block(name=&best_model)

ods graphics off;
