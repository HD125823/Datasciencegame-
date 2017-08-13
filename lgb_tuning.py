


n_estimaors = [120, 130, 150, 170]
learning_rate = [0.3, 0.1, 0.05, 0.01]

for i in n_estimaors:
  for j in learning_rate:
    print("n_estimators:", i)
    print("learning_rate:", j)
    clf = lgb.LGBMClassifier(n_estimators=i, learning_rate=j, num_leaves=1000,
                            colsample_bytree=.8, subsample=.8, silent=True,
                            min_child_weight=10, seed=1000)


    print("Training...")
    clf.fit(X_train, y_train)

    print("Validation...")
    pred = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred)
    print("AUC: {}".format(auc))


# n_estimators: 170 # or 150 (similarly good)
# learning_rate: 0.05
# Training...
# Validation...
# AUC: 0.7227027808069776


num_leaves = [800, 900, 1100]
min_child_weight = [1, 5, 10, 30]

for i in num_leaves:
  for j in min_child_weight:
    print("num_leaves:", i)
    print("min_child_weight:", j)
    clf = lgb.LGBMClassifier(n_estimators=170, learning_rate=0.05, num_leaves=i,
                            colsample_bytree=.8, subsample=.8, silent=True,
                            min_child_weight=j, seed=1000)


    print("Training...")
    clf.fit(X_train, y_train)

    print("Validation...")
    pred = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred)
    print("AUC: {}".format(auc))

# num_leaves: 1100
# min_child_weight: 1
# Training...
# Validation...
# AUC: 0.72367187


colsample_bytree = [0.5, .6, .7, .8, .9, 1]
subsample = [0.5, .6, .7, .8, .9, 1]

for i in colsample_bytree:
  for j in subsample:
    print("colsample_bytree:", i)
    print("subsample:", j)
    clf = lgb.LGBMClassifier(n_estimators=170, learning_rate=0.05, num_leaves=1100,
                            colsample_bytree=i, subsample=j, silent=True,
                            min_child_weight=1, seed=1000)


    print("Training...")
    clf.fit(X_train, y_train)

    print("Validation...")
    pred = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred)
    print("AUC: {}".format(auc))
    print()

# colsample_bytree: 0.9
# subsample: 1
# Training...
# Validation...
# AUC: 0.7266114650835803

# 0.9 and 0.9 or 0.8 and 0.8 are also strong (maybe good for ensemble)



min_child_sample = [10, 50, 100, 1000]

for i in min_child_sample:
    print("min_child_sample:", i)
    clf = lgb.LGBMClassifier(n_estimators=170, learning_rate=0.05, num_leaves=1100,
                            colsample_bytree=.9, subsample=1, silent=True,
                            min_child_weight=1, seed=1000, min_child_samples=i)


    print("Training...")
    clf.fit(X_train, y_train)

    print("Validation...")
    pred = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred)
    print("AUC: {}".format(auc))
    print()


# min_child_sample: 10
# Training...
# Validation...
# AUC: 0.7266114650835803


reg_alpha = [0.005, 0.0075, 0.01, 0.02, 0.03]

for i in reg_alpha:
    print("reg_alpha:", i)
    clf = lgb.LGBMClassifier(n_estimators=170, learning_rate=0.05, num_leaves=1100,
                            colsample_bytree=.9, subsample=1, silent=True,
                            min_child_weight=1, seed=1000, min_child_samples=10,
                            reg_alpha=i)


    print("Training...")
    clf.fit(X_train, y_train)

    print("Validation...")
    pred = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred)
    print("AUC: {}".format(auc))
    print()

# reg_alpha: 0.01
# Training...
# Validation...
# AUC: 0.7270699688019916


max_bin = [2000, 3000, 5000]

for i in max_bin:
    print("max_bin:", i)
    clf = lgb.LGBMClassifier(n_estimators=170, learning_rate=0.05, num_leaves=1100,
                            colsample_bytree=.9, subsample=1, silent=True,
                            min_child_weight=1, seed=1000, min_child_samples=10,
                            reg_alpha=0.01, max_bin=i)


    print("Training...")
    %time clf.fit(X_train, y_train)

    print("Validation...")
    pred = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred)
    print("AUC: {}".format(auc))
    print()

# max_bin: 5000
# Training...
# CPU times: user 3min 35s, sys: 1min 9s, total: 4min 44s
# Wall time: 2min 12s
# Validation...
# AUC: 0.7646433474270383


#####################

lgb_train = lgb.Dataset(X_train, y_train, max_bin=6000, free_raw_data=False, feature_name=feature_names)
lgb_val = lgb.Dataset(X_val, y_val, max_bin=6000, reference=lgb_train, free_raw_data=False, feature_name=feature_names)
# lgb_train.set_categorical_feature(["context_type", "platform_name",
#                                    "platform_family", "listen_type", "user_gender",
#                                    "ts_listen_month", "release_month", "release_year",
#                                    "genre_id", "media_id", "album_id", "user_id"])
# lgb_val.set_categorical_feature(["context_type", "platform_name",
#                                    "platform_family", "listen_type", "user_gender",
#                                    "ts_listen_month", "release_month", "release_year",
#                                     "genre_id", "media_id", "album_id", "user_id"])

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 3000,
    'learning_rate': 0.02,
    'feature_fraction': 0.9,
    'bagging_fraction': 1,
    'verbose': 0,
    "lambda_l1": 0.01,
    "min_data_in_leaf":10,
}


%time model = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_val, early_stopping_rounds=50)
# categorical feature encoding has no effect



clf = lgb.LGBMClassifier(n_estimators=170, learning_rate=0.05, num_leaves=5000,
                        colsample_bytree=.9, subsample=1, silent=True,
                        min_child_weight=1, seed=1000, min_child_samples=10,
                        reg_alpha=0.01, max_bin=5000)


print("Training...")
clf.fit(X_train, y_train,
      eval_set=[(X_val, y_val)],
      eval_metric="auc",
      early_stopping_rounds=30)
# Early stopping, best iteration is:
# [74]    valid_0's auc: 0.769561



clf = lgb.LGBMClassifier(n_estimators=210, learning_rate=0.07, num_leaves=30000,
                        colsample_bytree=.9, subsample=1, silent=True,
                        min_child_weight=1, seed=1000, min_child_samples=10,
                        reg_alpha=0.01, max_bin=5000)


print("Training...")
clf.fit(X_train, y_train,
      eval_set=[(X_val, y_val)],
      eval_metric="auc",
      early_stopping_rounds=30)
