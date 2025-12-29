""" import pandas as pd

df = pd.DataFrame({"Col1": [10, 20, 15, 30, 45],
                   "Col2": [13, 23, 18, 33, 48],
                   "Col3": [17, 27, 22, 37, 52]},
                  index=pd.date_range("2020-01-01", "2020-01-05"))

print(df)
df.shift(periods=1, axis="columns")
print(df.shift(periods=1, axis="columns")) """

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

df = pd.read_csv("procura.csv")
print(df)
df['lag_1'] = df['procura'].shift(1)
print(df['lag_1'])
df['lag_2'] = df['procura'].shift(2)
print(df['lag_2'])
df['mm3'] = df['procura'].rolling(3).mean()
print(df['mm3'])
df = df.dropna().reset_index(drop=True)

train = df.iloc[:7] # semanas 3..9
test = df.iloc[7:] # semanas 10..12

X_tr, y_tr = train[['lag_1', 'lag_2', 'mm3']], train['procura']
X_te, y_te = test[['lag_1', 'lag_2', 'mm3']], test['procura']

# Baseline (naive): usar último valor conhecido
y_pred_naive = test['lag_1'].values
mae_naive = mean_absolute_error(y_te, y_pred_naive)
mape_naive = np.mean(np.abs((y_te - y_pred_naive)/y_te))

lr = LinearRegression().fit(X_tr, y_tr)
y_pred = lr.predict(X_te)
mae = mean_absolute_error(y_te, y_pred)
mape = np.mean(np.abs((y_te - y_pred)/y_te))

print(f"Baseline -> MAE:{mae_naive:.1f} MAPE:{mape_naive:.3f}")
print(f"Modelo -> MAE:{mae:.1f} MAPE:{mape:.3f}")