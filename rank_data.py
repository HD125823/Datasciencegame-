import pandas as pd
import numpy as np
from scipy.stats import rankdata

weights = [1, .8, .5, .3, .2] # your weights for each model
files = ['xgb_datasciencegame.csv', 'lgb_val0.7696831.csv', 'lgb_val0.7683541.csv',
         'meta_blend.csv', 'lgb_val0.7646433.csv'] # your prediction files

finalRank = 0
for i in range(len(files)):
    temp_df = pd.read_csv(files[i])
    finalRank = finalRank + rankdata(temp_df.is_listened, method='ordinal') * weights[i]
finalRank = finalRank / (max(finalRank) + 1.0)

df = temp_df.copy()
df['is_listened'] = finalRank
df.to_csv('ensembleByRanking2.csv', index = False)
