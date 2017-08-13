# train on df_train
vw -d df_train.txt

# train on df_train, 10 passes over training data, cache
vw -d df_train.txt, -c, --passes 10

# -b 24 tells Vowpal Wabbit to use 24-bit hashes (18-bit hashes is default)
# -c -k means to use a cache for multiple passes, and kill any existing cache
# -f model.vw means “save model as ‘model.vw'”
vw df_train.txt -c -k --passes 10 -b 24 -f model.vw

# train neural net with 1000 hidden units in 1 hidden layer
vw -d df_train.txt --nn 1000

# expects labels in [-1, 1]
vw -d df_train.txt --binary
vw -d df_train.txt --loss_function logistic

# The size is 2b bits and by default b is 18. 218 is just 262144,
# so if there are more features than this you are guaranteed to get collisions.


# testing:
# -t testing only (no learning)
# -i model.vw says to use model.vw as the model
# -p preds.txt means “save predictions as ‘preds.txt'”
vw df_test.txt -t -i model.vw -p preds.txt

# run through sigmoid (predictions above are just from the linear model)
sigmoid_mc.py preds.txt p.txt


# simple logreg average loss = 0.633758
vw -d df_train.txt --loss_function logistic -f model.vw
vw df_test.txt -t -i model.vw -p y_submission.txt

import pandas as pd
import numpy as np
y_submission = pd.read_csv("y_submission.txt", header=None)
sigmoid = lambda x: 1 / (1+ np.exp(-x))
y_submission = sigmoid(y_submission)

tmp = np.vstack((range(0, len(y_submission)), y_submission.values.T)).T
np.savetxt(fname='vw.csv', X=tmp, fmt='%d, %0.9f',
           header='sample_id,is_listened', comments='')


# vw -d df_train.txt --loss_function logistic -q xx --l2 0.00000005 --passes 2
# average loss = 0.631216

vw -d df_train.txt --loss_function logistic -q xx --l2 0.00000005 -k -c --passes 2 -f model.vw
vw df_test.txt -q xx -t -i model.vw -p y_submission.txt

