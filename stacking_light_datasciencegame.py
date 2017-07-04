import numpy as np
import load_data
import time

from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



if __name__ == "__main__":


  np.random.seed(1000)


  n_folds = 5
  verbose = True
  shuffle = True

  X, y, X_sub, feature_names = load_data.load()


  if shuffle:
      idx = np.random.permutation(y.size)
      X = X[idx]
      y = y[idx]


  skf = list(StratifiedKFold(n_folds).split(X, y))


  # First stage models
  xg1 = XGBClassifier(colsample_bytree=0.9, gamma=5, learning_rate=0.07, max_depth=30,
                      n_estimators=51, reg_alpha=0.01, silent=False)

  xg2 = XGBClassifier(colsample_bytree=0.8, gamma=6, learning_rate=0.1, max_depth=33,
                      n_estimators=55, reg_alpha=0.0001, silent=False)

  xg3 = XGBClassifier(colsample_bytree=0.98, gamma=3, learning_rate=0.045, max_depth=28,
                      n_estimators=70, silent=False)

  xg4 = XGBClassifier(colsample_bytree=0.5, gamma=4, learning_rate=0.05, max_depth=38,
                      n_estimators=45, silent=False)

  gb1 = LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=1100,
                       colsample_bytree=.9, subsample=1, silent=False,
                       min_child_weight=1, seed=1000, min_child_samples=10,
                       reg_alpha=0.01, max_bin=5000)

  gb2 = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=5000,
                       colsample_bytree=.9, subsample=.95, silent=False,
                       min_child_weight=2, min_child_samples=20,
                       max_bin=6000)

  clfs = [xg1, xg2, xg3, xg4, gb1, gb2]

  #Stage 1: Creating train and test sets for second level
  dataset_stack_train = np.zeros((X.shape[0], len(clfs)))
  dataset_stack_test = np.zeros((X_sub.shape[0], len(clfs)))

  start = time.time()
  for j, clf in enumerate(clfs):
      print(j, clf)
      for i, (train, val) in enumerate(skf):
          print("Fold", i)
          X_train = X[train]
          y_train = y[train]
          X_val = X[val]
          y_val = y[val]
          clf.fit(X_train, y_train)
          y_submission = clf.predict_proba(X_val)[:, 1]
          dataset_stack_train[val, j] = y_submission
      clf.fit(X, y)
      dataset_stack_test[:, j] = clf.predict_proba(X_sub)[:, 1]



  print("Stage 2: Stacking...")
  clf = XGBClassifier(colsample_bytree=0.9, gamma=5, learning_rate=0.07, max_depth=30,
                      n_estimators=51, reg_alpha=0.01, silent=False)
  clf.fit(dataset_stack_train, y)
  y_submission = clf.predict_proba(dataset_stack_test)[:, 1]


  print("Saving Results...")
  tmp = np.vstack([range(0, len(y_submission)), y_submission]).T
  np.savetxt(fname='submission_stacking.csv', X=tmp, fmt='%d, %0.9f',
             header='sample_id,is_listened', comments='')
  print("Runtime of script: {}". format(time.time() - start))
