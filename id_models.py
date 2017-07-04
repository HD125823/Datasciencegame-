import load_data2
import lightgbm as lgb
import numpy as np



if __name__ == "__main__":


  # on original dataset (the big one)
  X, X_sub, feature_names = load_data2.load()

  idx = np.argsort(X.user_id.unique())
  unique_users = X.user_id.unique()[idx]
  X_sub["prediction"] = -1
  users_greater100 = []


  print("\nTraining...")

  for user in unique_users:
    if X[X.user_id == user].shape[0] >= 100:
      users_greater100.append(user)
      print(user)

      Xtrain = X[X.user_id == user].drop(["is_listened", "user_id"], axis=1).values
      ytrain = X[X.user_id == user].is_listened.values
      Xtest = X_sub[X_sub.user_id == user].drop(["user_id", "prediction"], axis=1).values


      clf = lgb.LGBMClassifier(n_estimators=170,
                               learning_rate=0.05,
                               num_leaves=1100,
                               colsample_bytree=.9,
                               subsample=1,
                               silent=True,
                               min_child_weight=1,
                               seed=1000,
                               min_child_samples=10,
                               reg_alpha=0.01,
                               max_bin=5000)



      clf.fit(Xtrain, ytrain)
      ypred = clf.predict_proba(Xtest)[:, 1]
      X_sub.loc[X_sub.user_id == user, "prediction"] = ypred



  Xtrain_remain = X[~X.user_id.isin(users_greater100)].drop("is_listened", axis=1).values
  ytrain_remain = X[~X.user_id.isin(users_greater100)].is_listened.values
  Xtest_remain = X_sub[~X_sub.user_id.isin(users_greater100)].drop("prediction", axis=1).values


  clf.fit(Xtrain_remain, ytrain_remain)
  ypred = clf.predict_proba(Xtest_remain)[:, 1]
  X_sub.loc[~X_sub.user_id.isin(users_greater100), "prediction"] = ypred


  print("Saving Results...")
  y_submission = X_sub.prediction.values
  tmp = np.vstack((range(0, len(y_submission)), y_submission)).T
  np.savetxt(fname='user_models.csv', X=tmp, fmt='%d, %0.9f',
             header='sample_id,is_listened', comments='')
  print("Done!")
