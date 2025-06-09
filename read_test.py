import pandas as pd
print(pd.read_feather("levels.feather").head())
print(pd.read_feather("gammas.feather").head())
print(pd.read_feather("q.feather").head())
