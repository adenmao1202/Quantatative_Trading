import pandas as pd

df = pd.read_csv("website_data.csv")

df.info()
df.plot()
## train test split
msk = df.index < len(df) - 30

print(msk)

df_train = df[msk].copy()  # 挑選後30 天 以外的
df_test = df[~msk].copy()  # 挑選後30 天

# Step 1 : Check for stationarity of time series
## ARIMA only suits for stationary time series

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF :
# ACF (autocorrelation function) is the correlation of the time series
# with its lags. It measures the linear relationship between lagged values of the time series.

acf_original = plot_acf(df_train)

# PACF :
# PACF (partial autocorrelation function) shows the
# partial correlation of the time series with its lags,
# after removing the effects of lower-order-lags between them.
pacf_original = plot_pacf(df_train)

# based on the plot, we can assume that the time series is not stationary

# OR we can use statsmodels to see the P-value
# with the null hypothesis  null hypothesis :
# " there is a unit root (non-stationary). "
from statsmodels.tsa.stattools import adfuller

adf_test = adfuller(df_train)
print(f"p-value: {adf_test[1]}")


# Transform to stationary: differencing
df_train_diff = df_train.diff().dropna()  # drop the first NaN
# diff():  find the difference between consecutive elements
df_train_diff.plot()

# Check ACF and PACF again
# CF plot (left) drops in value more quickly.
# While the PACF plot (right) also shows a less strong spike at lag 1.
# These are signs of the series being more stationary.

acf_diff = plot_acf(df_train_diff)

pacf_diff = plot_pacf(df_train_diff)

# Check ADF again

adf_test = adfuller(df_train_diff)
print(f"p-value: {adf_test[1]}")

# Step2 ;  Determine ARIMA models parameters p, q
""" 1. If the PACF plot has a significant spike at lag p, but not beyond; 
the ACF plot decays more gradually. This may suggest an ARIMA(p, d, 0) model
2. If the ACF plot has a significant spike at lag q, but not beyond;
the PACF plot decays more gradually. This may suggest an ARIMA(0, d, q) model """

## This is because the PACF measures the "balance variance" of the lags;
## it helps tell us whether we should include such lag within the auto-regressive (AR) models.
## While the ACF measures the "correlations with the lags", it helps judge the moving average (MA) models.
## Most of the time, we should focus on either the AR or the MA models, not mixed

## For our differenced series, the PACF has a large spike at lag 1, and still shows more minor but significant lags at 2, 4, and 5.
## In contrast, the ACF shows a more gradual decay.  we’ll fit an ARIMA(2, 1, 0) model.

# Step 3 : Fit an ARIMA model
from statsmodels.tsa.arima.model import ARIMA

# 這邊產出我們要的 model
model = ARIMA(df_train, order=(2, 1, 0))
model_fit = model.fit()
print(model_fit.summary())

# Step 4 : Forecast

import matplotlib
import matplotlib.pyplot as plt

## ifferences between observed values and the values predicted by the model
residuals = model_fit.resid[1:]  # 第一筆不看
print(residuals)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

residuals.plot(title="Residuals", ax=ax[0])  # left
residuals.plot(title="Density", kind="kde", ax=ax[1])  # right
plt.show()

## These show that the residuals are close to white noise.
## to make sure is ready to forecast with this model ARIMA(2, 1, 0).
acf_res = plot_acf(residuals)
pacf_res = plot_pacf(residuals)


# Forecasting
forecast_test = model_fit.forecast(len(df_test))

# adding a new column to the dataframe, and filling it with the forecast
df["forecast_manual"] = [None] * len(df_train) + list(forecast_test)

df.plot()


# AUTO ARIMA
import pmdarima as pm

auto_arima = pm.auto_arima(df_train, stepwise=False, seasonal=False)
auto_arima.summary

# Compare two models

forecast_test_auto = auto_arima.predict(n_periods=len(df_test))
df["forecast_auto"] = [None] * len(df_train) + list(forecast_test_auto)

df.plot()
