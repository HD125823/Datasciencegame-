import numpy as np
import pandas as pd

def load():

  train = pd.read_csv("data2/df_train_new.csv")
  X_test = pd.read_csv("data2/df_test_new.csv").values
  X_train = train.drop("is_listened" , axis=1).values
  y_train = train["is_listened"].values
  feature_names = train.columns.values[:-1]
  del train
  #print(X_train.shape, y_train.shape, X_test.shape)
  return X_train, y_train, X_test, feature_names




if __name__ == "__main__":

  X, y, X_sub, feature_names = load()
