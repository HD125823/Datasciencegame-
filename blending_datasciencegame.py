import numpy as np
import load_data

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier




if __name__ == "__main__":


  np.random.seed(1000)

  X, y, X_sub, feature_names = load_data.load()

  validx = int(X.shape[0] * 0.3)

  X_train = X[:-validx]
  y_train = y[:-validx]
  X_val = X[-validx:]
  y_val = y[-validx:]


  xg = XGBClassifier(colsample_bytree=0.9, gamma=5, learning_rate=0.07, max_depth=30,
                     n_estimators=51, reg_alpha=0.01, silent=False)


  gb = LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=1100,
                      colsample_bytree=.9, subsample=1, silent=False,
                      min_child_weight=1, seed=1000, min_child_samples=10,
                      reg_alpha=0.01, max_bin=5000)

  clfs = [xg, gb]

  print("Stage 1: Creating train and test sets for second level...")

  dataset_blend_train = np.zeros((X_val.shape[0], len(clfs)))
  dataset_blend_test = np.zeros((X_sub.shape[0], len(clfs)))

  start = time.time()
  for j, clf in enumerate(clfs):
    print(j, clf)
    clf.fit(X_train, y_train)
    dataset_blend_train[:, j] = clf.predict_proba(X_val)[:, 1]
    dataset_blend_test[:, j] = clf.predict_proba(X_sub)[:, 1]
  print("Done!")

  dataset_blend_train = np.hstack((X_val, dataset_blend_train))
  dataset_blend_test = np.hstack((X_sub, dataset_blend_test))


  print()
  print("Stage 2: Blending...")
  clf = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=5000,
                       colsample_bytree=.85, silent=False, seed=42, max_bin=5500,
                       reg_alpha=.012)
  clf.fit(dataset_blend_train, y_val)
  y_submission = clf.predict_proba(dataset_blend_test)[:, 1]
  print("Done!")


  print("Saving Results...")
  tmp = np.vstack([range(0, len(y_submission)), y_submission]).T
  np.savetxt(fname='submission_blending.csv', X=tmp, fmt='%d, %0.9f',
             header='sample_id,is_listened', comments='')
  print("Runtime of script: {}". format(time.time() - start))
