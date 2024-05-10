import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

# 設定隨機種子，以保證結果可重現
np.random.seed(42)

# 創建模擬的時間序列數據
n_samples = 100
time = pd.date_range(start="2021-01-01", periods=n_samples, freq="D")
data = (
    0.5 * np.arange(n_samples)
    + 10 * np.sin(np.linspace(0, 3.14 * 2, n_samples))
    + np.random.normal(size=n_samples, scale=10)
)
ts = pd.Series(data, index=time)

# 擬合 ARIMA 模型，選擇 (1,1,1) 作為參數
model = ARIMA(ts, order=(1, 1, 1))
model_fit = model.fit()

# 獲取模型的殘差
residuals = model_fit.resid

# 繪製殘差圖
plt.figure(figsize=(10, 4))
plt.plot(residuals)
plt.title("Residuals from ARIMA Model")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.show()

# 輸出殘差的統計描述
print("殘差的統計描述:")
print(residuals.describe())

# 殘差的正態性檢驗
jb_test = sm.stats.jarque_bera(residuals)
print("Jarque-Bera 檢驗結果:")
print(f"統計量: {jb_test[0]}, p-value: {jb_test[1]}")

# 確保殘差的隨機性和無自相關性
ljung_box_result = sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True)
print("隆格-盒檢定結果:")
print(ljung_box_result)
