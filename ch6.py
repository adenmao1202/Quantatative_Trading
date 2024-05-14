import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style and increase default resolution for saving figures
sns.set(style="seaborn")
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"

# Set display precision for pandas and numpy
pd.set_option("precision", 4)
np.set_printoptions(suppress=True, precision=4)

# URL of the dataset
url = "http://hilpisch.com/aiif_eikon_eod_data.csv"

# Read the dataset, parse dates for the index, and drop any missing values
try:
    data = pd.read_csv(url, index_col=0, parse_dates=True).dropna()
    print(data.head())  # Display the first few rows of the dataframe
except Exception as e:
    print(f"An error occurred: {e}")
