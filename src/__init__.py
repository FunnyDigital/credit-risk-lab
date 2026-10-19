"""Credit Risk Lab - an end-to-end credit default modelling pipeline.

Modules
-------
data        synthetic portfolio generator + real CSV loader
features    Weight-of-Evidence (WoE) binning and Information Value (IV)
scorecard   logistic-regression scorecard with points scaling
challenger  gradient-boosted challenger model + explainability
validation  KS, Gini/AUC, PSI, calibration, lift/gains
decision    score<->PD conversion, cut-off selection, expected loss
plots       matplotlib figures for the validation report
pipeline    orchestrates the whole thing and writes artifacts
"""

__version__ = "1.0.0"
