import talib as ta
from FinMind.data import DataLoader
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

"""
融资融券数据 (df_Mg)。
机构投资者数据 (df_Ins)。
外资持股数据 (df_FS)。
持股数据 (df_HS)。
日线数据 (df_kbar)。
"""

# 登入並獲取數據
url = "https://api.finmindtrade.com/api/v4/login"
payload = {"user_id": "109102049", "password": "Bb891202"}
data = requests.post(url, data=payload).json()
print(data)

# 初始化DataLoader並通過token登錄
api = DataLoader()
api.login_by_token(
    api_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyNC0wNS0wMSAxMjozMTozMiIsInVzZXJfaWQiOiIxMDkxMDIwNDkiLCJpcCI6IjExOC4yMzIuMTA4LjIxMyJ9.5beUa38IFaKsgw9QhD9lrF_kOVCritDIDI74591UN-E"
)

# 獲取融資融券數據
df_Mg = api.taiwan_stock_margin_purchase_short_sale(
    stock_id="2330", start_date="2019-04-30", end_date="2024-04-30"
)
df_Mg.drop(columns=["stock_id", "Note"], inplace=True)
df_Mg["date"] = pd.to_datetime(df_Mg["date"])  # 将日期列转换为 datetime 类型
print(df_Mg.head())

# 獲取機構投資者數據並進行One-Hot Encoding
df_Ins = api.taiwan_stock_institutional_investors(
    stock_id="2330", start_date="2019-04-30", end_date="2024-04-30"
)
df_Ins.drop(columns=["stock_id"], inplace=True)
df_Ins["date"] = pd.to_datetime(df_Ins["date"])  # 将日期列转换为 datetime 类型

encoder = OneHotEncoder(sparse_output=False)
encoded_categories = encoder.fit_transform(df_Ins[["name"]])
encoded_df = pd.DataFrame(
    encoded_categories, columns=encoder.get_feature_names_out(["name"])
)
df_Ins = df_Ins.drop("name", axis=1).join(encoded_df)
print(df_Ins.head())

pivot_Ins = df_Ins.pivot_table(index="date", aggfunc="first")
pivot_Ins.fillna(0, inplace=True)
pivot_Ins.columns = [
    "_".join(map(str, col)).strip() for col in pivot_Ins.columns.values
]
print(pivot_Ins.head())

# 獲取外資持股數據
df_FS = api.taiwan_stock_shareholding(
    stock_id="2330", start_date="2019-04-30", end_date="2024-04-30"
)
df_FS.drop(
    columns=[
        "stock_id",
        "stock_name",
        "InternationalCode",
        "note",
        "ChineseInvestmentUpperLimitRatio",
        "ForeignInvestmentUpperLimitRatio",
    ],
    inplace=True,
)
df_FS["date"] = pd.to_datetime(df_FS["date"])  # 将日期列转换为 datetime 类型
print(df_FS.head())

# 獲取持股數據並進行轉換
df_HS = api.taiwan_stock_holding_shares_per(
    stock_id="2330", start_date="2019-04-30", end_date="2024-04-30"
)
df_HS.drop(columns=["stock_id"], inplace=True)
df_HS["date"] = pd.to_datetime(df_HS["date"])  # 将日期列转换为 datetime 类型

mapping_dict = {
    "1-999": 500,
    "1,000-5,000": 2500,
    "10,001-15,000": 12500,
    "100,001-200,000": 150000,
    "15,001-20,000": 17500,
    "20,001-30,000": 25000,
    "200,001-400,000": 300000,
    "30,001-40,000": 35000,
    "40,001-50,000": 45000,
    "400,001-600,000": 500000,
    "5,001-10,000": 7500,
    "50,001-100,000": 75000,
    "600,001-800,000": 700000,
    "800,001-1,000,000": 900000,
}
df_HS["HSLevel_Mid"] = df_HS["HoldingSharesLevel"].apply(
    lambda x: mapping_dict.get(x, 0)
)
df_HS.drop(columns=["HoldingSharesLevel"], inplace=True)
print(df_HS.head())

pivot_HS = df_HS.pivot_table(
    index="date",
    columns="HSLevel_Mid",
    values=["people", "percent", "unit"],
    aggfunc="first",
)
pivot_HS.fillna(0, inplace=True)
pivot_HS.columns = ["_".join(map(str, col)).strip() for col in pivot_HS.columns.values]
print(pivot_HS.head())

# 獲取日線數據並計算技術指標
df_kbar = api.taiwan_stock_daily(
    stock_id="2330", start_date="2019-04-30", end_date="2024-04-30"
)
df_kbar.drop(columns=["stock_id"], inplace=True)
df_kbar["date"] = pd.to_datetime(df_kbar["date"])  # 将日期列转换为 datetime 类型
print("数据点数量:", len(df_kbar))
print("缺失数据:", df_kbar["close"].isnull().sum())
print(df_kbar.head())

# MACD
macd, signal, hist = ta.MACD(
    df_kbar["close"], fastperiod=12, slowperiod=26, signalperiod=9
)
df_kbar["MACD"] = macd
df_kbar["Signal Line"] = signal
df_kbar["MACD Histogram"] = hist

# DMI
adx = ta.ADX(df_kbar["max"], df_kbar["min"], df_kbar["close"], timeperiod=14)
plus_di = ta.PLUS_DI(df_kbar["max"], df_kbar["min"], df_kbar["close"], timeperiod=14)
minus_di = ta.MINUS_DI(df_kbar["max"], df_kbar["min"], df_kbar["close"], timeperiod=14)
df_kbar["ADX"] = adx
df_kbar["+DI"] = plus_di
df_kbar["-DI"] = minus_di

# KD
k, d = ta.STOCH(
    df_kbar["max"],
    df_kbar["min"],
    df_kbar["close"],
    fastk_period=14,
    slowk_period=3,
    slowk_matype=0,
    slowd_period=3,
    slowd_matype=0,
)
df_kbar["%K"] = k
df_kbar["%D"] = d

# RSI
rsi = ta.RSI(df_kbar["close"], timeperiod=14)
df_kbar["RSI"] = rsi

# BIAS
MA = ta.SMA(df_kbar["close"], timeperiod=20)
bias = (df_kbar["close"] - MA) / MA * 100
df_kbar["Bias"] = bias

print(df_kbar.tail())

# 合併數據框
df_merged = pd.merge(
    pd.merge(
        pd.merge(
            pd.merge(df_kbar, df_Mg, on="date", how="inner"),
            pivot_Ins,
            on="date",
            how="inner",
        ),
        df_FS,
        on="date",
        how="inner",
    ),
    pivot_HS,
    on="date",
    how="inner",
)
print(df_merged.tail())
print(df_merged.info())

# 移動列
f = 1
columns_to_shift = [
    "Trading_Volume",
    "Trading_money",
    "open",
    "max",
    "min",
    "close",
    "spread",
    "Trading_turnover",
    "MACD",
    "Signal Line",
    "MACD Histogram",
    "ADX",
    "+DI",
    "-DI",
    "%K",
    "%D",
    "RSI",
    "Bias",
    "MarginPurchaseBuy",
    "MarginPurchaseCashRepayment",
    "MarginPurchaseLimit",
    "MarginPurchaseSell",
    "MarginPurchaseTodayBalance",
    "MarginPurchaseYesterdayBalance",
    "OffsetLoanAndShort",
    "ShortSaleBuy",
    "ShortSaleCashRepayment",
    "ShortSaleLimit",
    "ShortSaleSell",
    "ShortSaleTodayBalance",
    "ShortSaleYesterdayBalance",
    "buy_0",
    "buy_1",
    "buy_2",
    "buy_3",
    "buy_4",
    "sell_0",
    "sell_1",
    "sell_2",
    "sell_3",
    "sell_4",
    "ForeignInvestmentRemainingShares",
    "ForeignInvestmentShares",
    "ForeignInvestmentRemainRatio",
    "ForeignInvestmentSharesRatio",
    "NumberOfSharesIssued",
    "people_0",
    "people_500",
    "people_2500",
    "people_7500",
    "people_12500",
    "people_17500",
    "people_25000",
    "people_35000",
    "people_45000",
    "people_75000",
    "people_150000",
    "people_300000",
    "people_500000",
    "people_700000",
    "people_900000",
    "percent_0",
    "percent_500",
    "percent_2500",
    "percent_7500",
    "percent_12500",
    "percent_17500",
    "percent_25000",
    "percent_35000",
    "percent_45000",
    "percent_75000",
    "percent_150000",
    "percent_300000",
    "percent_500000",
    "percent_700000",
    "percent_900000",
    "unit_0",
    "unit_500",
    "unit_2500",
    "unit_7500",
    "unit_12500",
    "unit_17500",
    "unit_25000",
    "unit_35000",
    "unit_45000",
    "unit_75000",
    "unit_150000",
    "unit_300000",
    "unit_500000",
    "unit_700000",
    "unit_900000",
]

for column in columns_to_shift:
    if column in df_merged.columns:
        df_merged[f"{column}_shifted"] = df_merged[column].shift(f)

df_merged.info()

df_merged["Returns"] = np.log(df_merged["close"] / df_merged["close"].shift(1))
df_merged.fillna(0, inplace=True)
print(df_merged["Returns"].head())

# 将处理好的数据集保存为CSV文件
df_merged.to_csv("processed_stock_data.csv", index=False)
