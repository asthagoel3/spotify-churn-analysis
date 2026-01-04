# Spotify Churn Capstone

## Overview
This project analyzes user churn (whether a user discontinued the service) using statistical hypothesis testing and logistic regression.

## Dataset
The dataset includes user demographics, subscription type, engagement metrics, and a churn indicator (`is_churned`).

## Methods
- Data validation and missingness checks
- Two-proportion z-tests:
  - churn by subscription type (Premium vs Free)
  - churn by age group (>30 vs ≤30)
- Chi-square test:
  - churn by country
- Logistic regression using behavioral engagement features
- Model evaluation using ROC–AUC

## Key Findings
Behavioral engagement metrics (listening time, skip rate, ad exposure) did not meaningfully predict churn in this dataset (ROC–AUC ≈ 0.5), suggesting churn may be driven by factors beyond short-term listening behavior.

## How to Run
```bash
pip install -r requirements.txt
python src/spotify_churn_analysis.py
