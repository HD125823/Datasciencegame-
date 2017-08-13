import load_data2
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans



if __name__ == "__main__":


  X, X_sub, feature_names = load_data2.load()


  # concat train and test
  X_full = pd.concat([X.drop("is_listened", axis=1), X_sub], axis=0)
  X_full.reset_index(inplace=True, drop=True)

  tridx = range(0, len(X))
  teidx = range(len(X), len(X_full))

  # predict and concat cluster centers
  km = KMeans(3)
  print("Kmeans...")
  cluster_info = km.fit_predict(X_full)
  X_full = pd.concat([X_full, pd.Series(cluster_info, name="cluster_info")], axis=1)
  X = pd.concat([X_full.iloc[tridx], X.is_listened], axis=1)
  X_sub = X_full.iloc[teidx]


  clusters = [0, 1, 2]
  X_sub["prediction"] = -1


  for cluster in clusters:

    Xtrain = X[X.cluster_info == cluster].drop(["is_listened", "cluster_info"], axis=1).values
    ytrain = X[X.cluster_info == cluster].is_listened.values
    Xtest = X_sub[X_sub.cluster_info == cluster].drop(["cluster_info", "prediction"], axis=1).values


    clf = lgb.LGBMClassifier(n_estimators=170, learning_rate=0.05, num_leaves=1100,
                            colsample_bytree=.9, subsample=1, silent=True,
                            min_child_weight=1, seed=1000, min_child_samples=10,
                            reg_alpha=0.01, max_bin=5000)



    print("Fit and predict for cluster", cluster)
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict_proba(Xtest)[:, 1]
    X_sub.loc[X_sub.cluster_info == cluster, "prediction"] = ypred
    print()


  print("Saving Results...")
  y_submission = X_sub.prediction.values
  tmp = np.vstack((range(0, len(y_submission)), y_submission)).T
  np.savetxt(fname='kmeans_models.csv', X=tmp, fmt='%d, %0.9f',
             header='sample_id,is_listened', comments='')
  print("Done!")
