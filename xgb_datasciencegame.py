import xgboost as xgb
import load_data
import numpy as np


if __name__ == "__main__":


  X, y, X_sub, feature_names = load_data.load()

  dtrain = xgb.DMatrix(X, y)
  dtest = xgb.DMatrix(X_sub)



  params = {
      "objective": "binary:logistic",
      "eval_metric": "auc",
      "colsample_bytree": 0.9,
      "max_depth": 30,
      "eta": 0.07,
      "silent": True,
      "alpha": 0.01,
      "gamma": 5
      }

  print("Training...")
  model = xgb.train(params, dtrain, 51)


  print("Predicting...")
  y_submission = model.predict(dtest)

  print("Saving Results...")
  tmp = np.vstack((range(0, len(y_submission)), y_submission)).T
  np.savetxt(fname='xgb_datasciencegame.csv', X=tmp, fmt='%d, %0.9f',
             header='sample_id,is_listened', comments='')
