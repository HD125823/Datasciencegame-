import xgboost as xgb

dtrain = xgb.DMatrix(X_train, y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val, y_val, feature_names=feature_names)
evallist  = [(dval, 'eval')]

# all with default seed of "seed": 0


# histogram
params1 = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "grow_policy": "depthwise",
    "max_bin": 2550,
    "eval_metric": "auc",
    "colsample_bytree": 0.9,
    "max_depth": 15,
    "eta": 0.07,
    "silent": True
    }
# 0.751873

# same but higher max_bin
params2 = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "grow_policy": "depthwise",
    "max_bin": 5000,
    "eval_metric": "auc",
    "colsample_bytree": 0.9,
    "max_depth": 15,
    "eta": 0.07,
    "silent": True
    }
# 0.755445

# best so far
params3 = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_leaves": 1100,
    "max_depth": 15,
    "max_bin": 5000,
    "eval_metric": "auc",
    "colsample_bytree": 0.9,
    "eta": 0.07,
    "silent": True
    }
# eval-auc:0.76048


params4 = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_leaves": 1100,
    "max_depth": 17,
    "max_bin": 5000,
    "eval_metric": "auc",
    "colsample_bytree": 0.9,
    "eta": 0.07,
    "silent": True
    }
# 0.762654


params5 = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_leaves": 1100,
    "max_depth": 16,
    "max_bin": 5000,
    "eval_metric": "auc",
    "colsample_bytree": 0.9,
    "eta": 0.07,
    "silent": True
    }
# 0.762009

params6 = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_leaves": 1100,
    "max_depth": 17,
    "max_bin": 5000,
    "eval_metric": "auc",
    "colsample_bytree": 0.9,
    "subsample": 0.95,
    "eta": 0.07,
    "silent": True
    }
# 0.760773


params7 = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_leaves": 1500,
    "max_depth": 17,
    "max_bin": 5000,
    "eval_metric": "auc",
    "colsample_bytree": 0.9,
    "eta": 0.07,
    "silent": True
    }

# Stopping. Best iteration:
# [185]   eval-auc:0.763001 !!!


params8 = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_leaves": 2000,
    "max_depth": 17,
    "max_bin": 5000,
    "eval_metric": "auc",
    "colsample_bytree": 0.9,
    "eta": 0.07,
    "silent": True
    }
# Stopping. Best iteration:
# [209]   eval-auc:0.763272


# classic
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "colsample_bytree": 0.9,
    "max_depth": 17,
    "eta": 0.07,
    "silent": True,
    "alpha": 0.01,
    "gamma": 5
    }

# Stopping. Best iteration:
# [215]   eval-auc:0.765643


params2 = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "colsample_bytree": 0.9,
    "max_depth": 20,
    "eta": 0.07,
    "silent": True,
    "alpha": 0.01,
    "gamma": 5
    }

# Stopping. Best iteration:
# [111]   eval-auc:0.767059

params3 = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "colsample_bytree": 0.9,
    "max_depth": 25,
    "eta": 0.07,
    "silent": True,
    "alpha": 0.01,
    "gamma": 5
    }

# Stopping. Best iteration:
# [93]    eval-auc:0.769731

params4 = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "colsample_bytree": 0.9,
    "max_depth": 30,
    "eta": 0.07,
    "silent": True,
    "alpha": 0.01,
    "gamma": 5
    }

# Stopping. Best iteration:
# [51]    eval-auc:0.7706

% time model = xgb.train(params4, dtrain, 500, evals=evallist, early_stopping_rounds=35)

