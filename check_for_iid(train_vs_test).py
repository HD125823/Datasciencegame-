"""Evaluate if a classifier can distinguish between the training and test sample. If yes (high performance) then the 
samples come from different distributions. """


import pandas as pd
import numpy as np
import load_data

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


if __name__ == "__main__":

  X, y, X_sub, _ = load_data.load()
  X, X_sub = pd.DataFrame(X), pd.DataFrame(X_sub)
  X["TARGET"], X_sub["TARGET"] = 1, 0
  data = pd.concat((X, X_sub))
  X = data.drop("TARGET", axis=1).values
  y = data["TARGET"].values

  skf = StratifiedKFold(5, random_state=1000, shuffle=True)


  reg = LogisticRegression()
  print("5 fold CV Logreg...")
  cv_score = cross_val_score(reg, X, y, cv=skf, scoring="roc_auc")
  print("mean CV score: {} +/- {}".format(cv_score.mean(), cv_score.std()))
  # 0.80


  clf = lgb.LGBMClassifier(colsample_bytree=0.8, learning_rate=0.1,
        min_child_samples=10, n_estimators=120, num_leaves=1000,
        seed=1000, silent=False, subsample=0.8)

  print("5 fold CV LGB...")
  cv_score = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")
  print("mean CV score: {} +/- {}".format(cv_score.mean(), cv_score.std()))
  # 0.98




