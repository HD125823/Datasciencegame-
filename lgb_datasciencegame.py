import lightgbm as lgb
import numpy as np
import load_data


if __name__ == "__main__":


  X, y, X_sub, feature_names = load_data.load()

  gb = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=1100,
                          colsample_bytree=.9, subsample=1, silent=False,
                          min_child_weight=1, seed=1000, min_child_samples=10,
                          reg_alpha=0.01, max_bin=5000)

  print("Training...")
  gb.fit(X, y)

  print("Predicting...")
  y_submission = gb.predict_proba(X_sub)[:, 1]

  print("Saving Results...")
  tmp = np.vstack((range(0, len(y_submission)), y_submission)).T
  np.savetxt(fname='lgb_V4datasciencegame.csv', X=tmp, fmt='%d, %0.9f',
             header='sample_id,is_listened', comments='')
