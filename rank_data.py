"""This file takes as input csv prediction files, ranks and then averages those predictions 
(described here: https://mlwave.com/kaggle-ensembling-guide/). 
Through the weight parameter, its possible to define the strength that you want to assign to an individual model.
"""


import pandas as pd
import numpy as np
from scipy.stats import rankdata

# weights for each model
weights = [1, .8, .5, .3, .2] 
# prediction files that you want to include
files = ['xgb_datasciencegame.csv', 'lgb_val0.7696831.csv', 'lgb_val0.7683541.csv',
         'meta_blend.csv', 'lgb_val0.7646433.csv'] 

# average the predictions.
finalRank = 0
for i in range(len(files)):
    temp_df = pd.read_csv(files[i])
    finalRank = finalRank + rankdata(temp_df.is_listened, method='ordinal') * weights[i]
finalRank = finalRank / (max(finalRank) + 1.0)

df = temp_df.copy()
df['is_listened'] = finalRank
df.to_csv('ensembleByRanking2.csv', index=False)
