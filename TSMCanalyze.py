import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import numpy as np

# 读取数据并将日期列转换为日期时间格式
data = pd.read_csv("processed_stock_data.csv")
data["date"] = pd.to_datetime(data["date"])

# 查看数据集
print(data.head())

# 检查数据类型
print(data.dtypes)

# 分离日期列
dates = data["date"]
data_without_dates = data.drop(columns=["date"])

# 确保数据集中只有数值列
data_numeric = data_without_dates.select_dtypes(include=[np.number])

# 假设数据最后一列是 'Returns'
X = data_numeric.drop(columns=["Returns"])
y = data_numeric["Returns"]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 进行PCA分析
pca = PCA(n_components=80)  # 选择80个主成分
X_pca = pca.fit_transform(X_scaled)

# 获取PCA分析结果并转换为DataFrame
pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
pca_df["date"] = dates.reset_index(drop=True)

# 输出每个主成分的解释方差比
explained_variance_ratio = pca.explained_variance_ratio_
print("每个主成分的解释方差比:", explained_variance_ratio)

# 输出累积解释方差比
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
print("累积解释方差比:", cumulative_explained_variance_ratio)

# 绘制解释方差比图表
plt.figure(figsize=(10, 6))
plt.bar(
    range(1, len(explained_variance_ratio) + 1),
    explained_variance_ratio,
    alpha=0.5,
    align="center",
    label="Individual explained variance",
)
plt.step(
    range(1, len(cumulative_explained_variance_ratio) + 1),
    cumulative_explained_variance_ratio,
    where="mid",
    label="Cumulative explained variance",
)
plt.ylabel("Explained variance ratio")
plt.xlabel("Principal component index")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# 使用LassoCV进行特征选择
lasso = LassoCV(cv=5)
lasso.fit(X_scaled, y)

# 打印系数
print("Lasso Coefficients:", lasso.coef_)

# 选择非零系数的特征
selected_features = X.columns[lasso.coef_ != 0]
print("Selected Features:", selected_features)

# 保留选择的特征，并重新添加日期列
lasso_selected_data = data[["date"] + selected_features.tolist() + ["Returns"]]

# 将PCA和LASSO结果合并，并重新添加日期列
combined_df = pd.concat([pca_df, lasso_selected_data.drop(columns=["date"])], axis=1)

# 将处理好的数据集保存为CSV文件
combined_df.to_csv("processed_stock_data_combined.csv", index=False)

# 查看保存的文件
print("保存的文件内容:")
print(combined_df.head())
