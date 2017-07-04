import numpy as np
import pandas as pd

def load():

  X = pd.read_csv("data2/df_train.csv")
  X_sub = pd.read_csv("data2/df_test.csv")
  feature_names = X.columns.values[:-1]
  print(X.shape, X_sub.shape)
  return X, X_sub, feature_names




if __name__ == "__main__":

  X, X_sub, feature_names = load()
