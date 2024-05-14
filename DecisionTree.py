import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

loans = pd.read_csv("loan_data.csv")
"""
Project Description: Loan Default Prediction Using Machine Learning Models

Overview:
This project aims to build predictive models to identify borrowers likely to default on their loans. Using data from LendingClub.com, we analyze various borrower attributes and loan features to develop models that predict loan repayment status. The goal is to assist in credit underwriting decisions by identifying high-risk borrowers.

Data Description:
The dataset includes the following features:
- credit.policy: Binary variable indicating if the customer meets LendingClub.com credit underwriting criteria.
- purpose: Categorical variable representing the loan purpose.
- int.rate: Interest rate of the loan as a proportion.
- installment: Monthly installment amount.
- log.annual.inc: Natural log of the self-reported annual income of the borrower.
- dti: Debt-to-income ratio.
- fico: FICO credit score.
- days.with.cr.line: Number of days the borrower has had a credit line.
- revol.bal: Revolving balance.
- revol.util: Revolving line utilization rate.
- inq.last.6mths: Number of inquiries by creditors in the last 6 months.
- delinq.2yrs: Number of 30+ days past due payments in the past 2 years.
- pub.rec: Number of derogatory public records.
- not.fully.paid: Binary target variable indicating if the loan is not fully paid.

Exploratory Data Analysis (EDA):
- Descriptive Statistics: Summary of basic statistics.
- Missing Values: Analysis confirming no missing values.
- Distribution Plots: Histograms of numerical features.
- Correlation Matrix: Heatmap to identify relationships between numerical features.
- Categorical Feature Distribution: Analysis of the 'purpose' feature using count plots.

Data Preprocessing:
- One-Hot Encoding: Convert the categorical 'purpose' feature into dummy variables.
- Train-Test Split: Split the data into training and testing sets (70-30 ratio).

Modeling:
1. Decision Tree Classifier:
   - Train a decision tree model on the training set.
   - Evaluate model performance using a classification report and confusion matrix.

2. Random Forest Classifier:
   - Train a random forest model with 300 estimators.
   - Evaluate model performance using a classification report and confusion matrix.

Results:
- Model performance was assessed based on accuracy, precision, recall, and F1-score.
- The confusion matrix provided insights into true positives, true negatives, false positives, and false negatives.

Conclusion:
This project demonstrates the use of decision tree and random forest classifiers to predict loan defaults. Analyzing borrower attributes and loan features helps identify patterns for informed predictions about loan repayment. These insights can help financial institutions make better credit underwriting decisions and manage risk more effectively.
"""

## Basic EDA
loans.head()
loans.info()
loans.describe()
# Missing values
print("\nMissing values in the dataset:")
print(loans.isnull().sum())  # the data has no missing values

# Visualizing the distribution of numerical features
numerical_features = loans.select_dtypes(include=[np.number]).columns.tolist()

print("\nDistribution of numerical features:")
loans[numerical_features].hist(bins=30, figsize=(15, 10), layout=(5, 3))
plt.tight_layout()
plt.show()

# Correlation analysis using HeatMap
numerical_data = loans[numerical_features]
correlation_matrix = numerical_data.corr()

print("\nCorrelation matrix:")
print(correlation_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Numerical Features")
plt.show()

# Distribution of the categorical feature 'purpose'
print("\nDistribution of the 'purpose' column:")
print(loans["purpose"].value_counts())

plt.figure(figsize=(10, 6))
sns.countplot(x="purpose", data=loans, order=loans["purpose"].value_counts().index)
plt.title("Distribution of Loan Purposes")
plt.xticks(rotation=45)
plt.show()


## Processing data for the usage in the model
# setting dummies for purpose column
cat_feats = ["purpose"]
loans[cat_feats].head()
final_data = pd.get_dummies(
    data=loans, columns=cat_feats
)  # original "purpose" column is removed auto by get_dummies
final_data.head()  # expanded purpose to several columns, and the val is BOOL

# Train Test Split
from sklearn.model_selection import train_test_split

X = final_data.drop("not.fully.paid", axis=1)
y = final_data["not.fully.paid"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=101
)

## Decision Tree Model
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Analysis of Decision Tree
predictions = dtree.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

## Random Forest Model
rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)

# Analysis of Random Forest
predictions = rfc.predict(X_test)


print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
