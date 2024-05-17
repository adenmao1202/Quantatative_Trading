import pandas as pd
import numpy as np
import requests
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import time

# Alpha Vantage API配置
ALPHA_VANTAGE_API_KEY = 'Z68LNMIA5Y7T9NB2'
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

# 特征列表
features = ['MACD', '+DI', 'RSI', 'Bias', 'OffsetLoanAndShort', 'spread_shifted',
            'MACD Histogram_shifted', '-DI_shifted', '%K_shifted', 'Bias_shifted',
            'MarginPurchaseLimit_shifted', 'ShortSaleLimit_shifted',
            'ForeignInvestmentShares_shifted', 'ForeignInvestmentSharesRatio_shifted',
            'NumberOfSharesIssued_shifted']

# 加载并预处理数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    data["date"] = pd.to_datetime(data["date"])
    X = data[features]
    y = data["Returns"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# 构建LSTM模型
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# 训练LSTM模型
def train_lstm_model(X_train, y_train):
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    model = build_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
    model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=2)
    return model

# 获取实时数据
def get_realtime_data(symbol):
    data, meta_data = ts.get_quote_endpoint(symbol=symbol)
    return data

# 预测未来市场走势
def predict_future(model, X_last, scaler, n_future_steps):
    predictions = []
    X_input = X_last.reshape((1, 1, X_last.shape[0]))
    for _ in range(n_future_steps):
        y_pred = model.predict(X_input)
        predictions.append(y_pred.flatten()[0])
        # 更新输入数据
        X_input = np.roll(X_input, -1)
        X_input[0, 0, -1] = y_pred.flatten()[0]
    return predictions

# 加载数据并训练模型
X_scaled, y, scaler = load_data("processed_stock_data_combined.csv")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
lstm_model = train_lstm_model(X_train, y_train)

# 获取最新的特征数据
X_last = X_scaled[-1, :]

# 实时预测示例
symbol = '2330.TW'
realtime_data = get_realtime_data(symbol)
# 假设我们提取并预处理实时数据到合适的特征格式
# 这里需要根据实际情况调整

# 预测未来10步
n_future_steps = 10
future_predictions = predict_future(lstm_model, X_last, scaler, n_future_steps)

print("Future Predictions:", future_predictions)

# 绘制未来预测结果
plt.figure(figsize=(10, 6))
plt.plot(range(len(future_predictions)), future_predictions, marker='o')
plt.xlabel("Future Steps")
plt.ylabel("Predicted Returns")
plt.title("Future Predictions for Next 10 Days")
plt.show()
