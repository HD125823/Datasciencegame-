import numpy as np
import load_data

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



if __name__ == "__main__":


  X, y, X_sub, feature_names = load_data.load()


  xg = XGBClassifier(colsample_bytree=0.9,
                     gamma=5,
                     learning_rate=0.07,
                     max_depth=30,
                     n_estimators=51,
                     reg_alpha=0.01,
                     silent=True).fit(X, y)


  ypred = xg.predict_proba(X_sub)

  trind = range(0, len(X))
  meta_bag_rounds = range(1, 10)
  bag_size = len(trind)



  for i in meta_bag_rounds:
    print("\nIteration:", i)
    bag = np.random.choice(trind, size=bag_size, replace=True)
    train = list(set(trind) - set(bag))

    X_train = X[train]
    y_train = y[train]

    print("Training BaseLearner")
    gb = LGBMClassifier(n_estimators=70,
                        learning_rate=0.2,
                        num_leaves=1100,
                        colsample_bytree=.9,
                        subsample=1,
                        silent=True,
                        min_child_weight=1,
                        seed=1000,
                        min_child_samples=10,
                        reg_alpha=0.01,
                        max_bin=5000).fit(X_train, y_train)

    X_val = X[bag]
    y_val = y[bag]

    # Predicting MetaFeatures
    tmp_blend_train = gb.predict_proba(X_val)
    tmp_blend_test = gb.predict_proba(X_sub)
    blend_train = np.concatenate((X_val, tmp_blend_train), axis=1)
    blend_test = np.concatenate((X_sub, tmp_blend_test), axis=1)

    print("Training Blender")
    xg = XGBClassifier(colsample_bytree=0.8,
                       gamma=5,
                       learning_rate=0.4,
                       max_depth=30,
                       n_estimators=50,
                       reg_alpha=0.01,
                       silent=True).fit(blend_train, y_val)

    ypred0 = xg.predict_proba(blend_test)
    ypred = ypred + ypred0
    print("--"*30)

  y_submission = ypred[:, 1]/(i+1)
  print("\nDone!")


  print("Saving Results...")
  tmp = np.vstack([range(0, len(y_submission)), y_submission]).T
  np.savetxt(fname='meta_blend.csv', X=tmp, fmt='%d, %0.9f',
             header='sample_id,is_listened', comments='')

