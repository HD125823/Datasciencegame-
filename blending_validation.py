import numpy as np
import load_data

from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



if __name__ == "__main__":


  X, y, X_sub, _ = load_data.load()


  validx = int(X.shape[0] * 0.3)

  X_train = X[:-validx]
  y_train = y[:-validx]
  X_val = X[-validx:]
  y_val = y[-validx:]



  # n_estimators=51 and 500
  xg = XGBClassifier(colsample_bytree=0.9, gamma=5, learning_rate=0.07, max_depth=30,
                     n_estimators=51, reg_alpha=0.01, silent=False)


  gb = LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=1100,
                      colsample_bytree=.9, subsample=1, silent=False,
                      min_child_weight=1, seed=1000, min_child_samples=10,
                      reg_alpha=0.01, max_bin=5000)

  clfs = [xg, gb]


  print("\nStage 1: Creating train and test blend...")

  dataset_blend_train = np.zeros((X_val.shape[0], len(clfs)))
  #dataset_blend_test = np.zeros((X_sub.shape[0], len(clfs)))

  for j, clf in enumerate(clfs):
    print(j, clf)
    clf.fit(X_train, y_train)
    dataset_blend_train[:, j] = clf.predict_proba(X_val)[:, 1]
    #dataset_blend_test[:, j] = clf.predict_proba(X_sub)[:, 1]
  print("\nDone!")

  dataset_blend_train = np.hstack((X_val, dataset_blend_train))
  #dataset_blend_test = np.hstack((X_sub, dataset_blend_test))


  print()
  print("\nStage 2: Blending validation...")
  validx = int(dataset_blend_train.shape[0] * 0.3)

  X_train = dataset_blend_train[:-validx]
  y_train = y_val[:-validx]
  X_val = dataset_blend_train[-validx:]
  y_val = y_val[-validx:]


  clf = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=5000,
                       colsample_bytree=.85, silent=False, seed=42, max_bin=5500,
                       reg_alpha=.012)

  clf.fit(X_train, y_train)
  y_pred = clf.predict_proba(X_val)[:, 1]
  print("\nDone!")
  print(roc_auc_score(y_val, y_pred))


