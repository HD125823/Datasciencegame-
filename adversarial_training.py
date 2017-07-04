import pandas as pd
import numpy as np
import load_data

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score



if __name__ == "__main__":


  # load V2 (best so far)
  X, y_orig, X_sub, feature_names = load_data.load()
  X, X_sub = pd.DataFrame(X), pd.DataFrame(X_sub)
  X["is_test"], X_sub["is_test"] = 0, 1
  data = pd.concat([X, X_sub], axis=0)
  data.reset_index(inplace=True, drop=True)
  X = data.drop("is_test", axis=1).values
  y = data["is_test"].values



  clf = lgb.LGBMClassifier(colsample_bytree=0.8, learning_rate=0.1,
        min_child_samples=10, n_estimators=120, num_leaves=1000,
        seed=1000, silent=False, subsample=0.8)


  skf = StratifiedKFold(5, random_state=1000, shuffle=True)
  skf = list(skf.split(X, y))


  predictions = np.zeros(y.shape)

  print("CV...")
  for i, (train, val) in enumerate(skf):
      print("Fold", i)
      X_train = X[train]
      y_train = y[train]
      X_val = X[val]
      y_val = y[val]
      clf.fit(X_train, y_train)
      y_submission = clf.predict_proba(X_val)[:, 1]
      auc = roc_auc_score(y_val, y_submission)
      print("AUC: {}".format(auc))
      predictions[val] = y_submission


  data["p"] = predictions
  i = predictions.argsort()
  # create additional random numbers only used for the
  #concatenation process, will be deleted later on
  y_tmp = np.random.rand(X_sub.shape[0])
  y = np.concatenate((y_orig, y_tmp))
  data = pd.concat([data, pd.Series(y)], axis=1)
  # data_sorted = data.iloc[i]


  # print("Save file")
  # # pull out the training samples
  # train_sorted = data_sorted[data_sorted.is_test == 0]
  # train_sorted = train_sorted.drop(["is_test", "p"], axis=1)
  # train_sorted.to_csv("df_train_sorted.csv", header=True, index=False)
  # print("Done!")


