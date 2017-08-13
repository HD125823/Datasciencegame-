from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import numpy as np
import load_data



if __name__ == "__main__":



  X, y, X_sub, feature_names = load_data.load()


  valSize = int(X.shape[0] * 0.3)

  X_train = X[:-valSize]
  y_train = y[:-valSize]
  X_val = X[-valSize:]
  y_val = y[-valSize:]


  clf = lgb.LGBMClassifier(n_estimators=170, learning_rate=0.05, num_leaves=5000,
                           colsample_bytree=.9, subsample=1, silent=True,
                           min_child_weight=1, seed=1000, min_child_samples=10,
                           reg_alpha=0.01, max_bin=5000)


  print("Training...")
  clf.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric="auc",
          early_stopping_rounds=30)

  print("Validation...")
  pred = clf.predict_proba(X_val, num_iteration=clf.best_iteration)[:, 1]
  auc = roc_auc_score(y_val, pred)
  print("AUC: {}".format(auc))

