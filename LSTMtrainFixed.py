import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import random
import matplotlib.pyplot as plt

# 固定随机种子
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# 读取数据
data = pd.read_csv("processed_stock_data_combined.csv")

# 将日期列转换为日期时间格式
data["date"] = pd.to_datetime(data["date"])

# 设置目标变量和特征变量
X = data.drop(columns=["Returns", "date"])
y = data["Returns"]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# LSTM模型
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 将数据转换为LSTM需要的格式
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 构建LSTM模型
lstm_model = build_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))

# 训练LSTM模型
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=2)

# 预测LSTM模型
y_pred_lstm = lstm_model.predict(X_test_lstm)
lstm_mse = mean_squared_error(y_test, y_pred_lstm)
lstm_rmse = np.sqrt(lstm_mse)
lstm_mae = mean_absolute_error(y_test, y_pred_lstm)
lstm_r2 = r2_score(y_test, y_pred_lstm)

print("LSTM Performance:")
print("Mean Squared Error (MSE):", lstm_mse)
print("Root Mean Squared Error (RMSE):", lstm_rmse)
print("Mean Absolute Error (MAE):", lstm_mae)
print("R2 Score:", lstm_r2)

# SVR模型
svr_model = SVR()
svr_model.fit(X_train, y_train)

# 预测SVR模型
y_pred_svr = svr_model.predict(X_test)
svr_mse = mean_squared_error(y_test, y_pred_svr)
svr_rmse = np.sqrt(svr_mse)
svr_mae = mean_absolute_error(y_test, y_pred_svr)
svr_r2 = r2_score(y_test, y_pred_svr)

print("SVR Performance:")
print("Mean Squared Error (MSE):", svr_mse)
print("Root Mean Squared Error (RMSE):", svr_rmse)
print("Mean Absolute Error (MAE):", svr_mae)
print("R2 Score:", svr_r2)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lstm, alpha=0.3)
plt.xlabel("Actual Returns")
plt.ylabel("Predicted Returns")
plt.title("LSTM Model: Actual vs Predicted Returns")
plt.show()

# 绘制残差图
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred_lstm.flatten()
plt.scatter(y_pred_lstm, residuals, alpha=0.3)
plt.xlabel("Predicted Returns")
plt.ylabel("Residuals")
plt.title("LSTM Model: Residuals vs Predicted Returns")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
