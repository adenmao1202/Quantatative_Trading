import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 讀取數據
data = pd.read_csv("processed_stock_data_combined.csv")

# 設定特徵列表，包括所有需要的特徵
features = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "MACD",
    "+DI",
    "RSI",
    "Bias",
    "OffsetLoanAndShort",
    "spread_shifted",
    "MACD Histogram_shifted",
    "-DI_shifted",
    "%K_shifted",
    "Bias_shifted",
    "MarginPurchaseLimit_shifted",
    "ShortSaleLimit_shifted",
    "ForeignInvestmentShares_shifted",
    "ForeignInvestmentSharesRatio_shifted",
    "NumberOfSharesIssued_shifted",
]

# 構建特徵和標籤
X = data[features]
y = data["target"]

# 切分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 轉換數據為DMatrix格式（XGBoost特定的數據格式）
train_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
test_dmatrix = xgb.DMatrix(data=X_test, label=y_test)

# 設定模型參數
params = {
    "objective": "reg:squarederror",  # 迴歸任務
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# 訓練模型
model = xgb.train(params, train_dmatrix, num_boost_round=100)

# 預測
preds = model.predict(test_dmatrix)

# 計算均方誤差（MSE）
mse = mean_squared_error(y_test, preds)
print(f"Mean Squared Error: {mse}")

# 準備用於未來10天預測的數據
future_data = data.tail(10)[features]
future_dmatrix = xgb.DMatrix(data=future_data)

# 預測未來10天的收盤價
future_preds = model.predict(future_dmatrix)
print("Predicted future 10 days closing prices:", future_preds)
