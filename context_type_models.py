import load_data2
import lightgbm as lgb
import numpy as np


if __name__ == "__main__":


  X, X_sub, feature_names = load_data2.load()


  context_type = [1, 5, 20, 23]
  X_sub["prediction"] = -1


  for context in context_type:

    Xtrain = X[X.context_type == context].drop(["is_listened", "context_type"], axis=1).values
    ytrain = X[X.context_type == context].is_listened.values
    Xtest = X_sub[X_sub.context_type == context].drop(["context_type", "prediction"], axis=1).values


    clf = lgb.LGBMClassifier(n_estimators=170, learning_rate=0.05, num_leaves=1100,
                            colsample_bytree=.9, subsample=1, silent=True,
                            min_child_weight=1, seed=1000, min_child_samples=10,
                            reg_alpha=0.01, max_bin=5000)



    print("Fit and predict for context_type", context)
    print(Xtrain.shape, Xtest.shape)
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict_proba(Xtest)[:, 1]
    X_sub.loc[X_sub.context_type == context, "prediction"] = ypred
    print()


  print("Saving Results...")
  y_submission = X_sub.prediction.values
  tmp = np.vstack((range(0, len(y_submission)), y_submission)).T
  np.savetxt(fname='context_models.csv', X=tmp, fmt='%d, %0.9f',
             header='sample_id,is_listened', comments='')
  print("Done!")
