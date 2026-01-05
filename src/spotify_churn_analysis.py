import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


# --- Load data ---
df = pd.read_csv("data/spotify_churn_dataset.csv")

print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# --- Z-test: churn differs by subscription type (Premium vs Free) ---
premium_df = df[df["subscription_type"] == "Premium"]
free_df = df[df["subscription_type"] == "Free"]

n_premium = len(premium_df)
n_free = len(free_df)

churn_premium = premium_df["is_churned"].sum()
churn_free = free_df["is_churned"].sum()

subcount = [churn_free, churn_premium]
subnobs = [n_free, n_premium]

zstat, pval = proportions_ztest(subcount, subnobs, alternative="two-sided")
print("\nSubscription churn z-test")
print("z-stat:", zstat, "p-value:", pval)

# --- Z-test: churn differs by age group (>30 vs <=30) ---
above_30_df = df[df["age"] > 30]
below_30_df = df[df["age"] <= 30]

n_above_30 = len(above_30_df)
n_below_30 = len(below_30_df)

churn_above_30 = above_30_df["is_churned"].sum()
churn_below_30 = below_30_df["is_churned"].sum()

agecount = [churn_above_30, churn_below_30]
agenobs = [n_above_30, n_below_30]

zstat_age, pval_age = proportions_ztest(agecount, agenobs, alternative="two-sided")
print("\nAge group churn z-test")
print("z-stat:", zstat_age, "p-value:", pval_age)

# --- Chi-square: churn differs by country ---
print("\nNumber of unique countries:", df["country"].nunique())
print("\nCountry distribution:\n", df["country"].value_counts())

churn_by_country = (
    df.groupby("country")
    .agg(
        Total_churned=("is_churned", "sum"),
        Avg=("is_churned", "mean"),
        Sample_size=("is_churned", "count"),
    )
    .sort_values(by="Avg", ascending=False)
)
print("\nChurn by country (top 10 by churn rate):\n", churn_by_country.head(10))

country_contingency = pd.crosstab(df["country"], df["is_churned"])
chi2, pval_country, _, _ = chi2_contingency(country_contingency)
print("\nCountry churn chi-square")
print("chi2-stat:", chi2, "p-value:", pval_country)

# --- Logistic regression: behavioral predictors ---
behavioral_predictors = [
    "listening_time",
    "songs_played_per_day",
    "skip_rate",
    "ads_listened_per_week",
]

X = sm.add_constant(df[behavioral_predictors])
y = df["is_churned"]

logit_model_behavioral = sm.Logit(y, X)
results = logit_model_behavioral.fit()
print("\nBehavioral logistic regression results:\n")
print(results.summary())

# --- Train/test evaluation (ROC-AUC) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

logit = sm.Logit(y_train, X_train).fit(disp=False)
y_pred_prob = logit.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nEvaluation")
print("ROC AUC:", roc_auc_score(y_test, y_pred_prob))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
