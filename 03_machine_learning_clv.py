# BUSINESS SCIENCE -----
# LAB 91: CUSTOMER LIFETIME VALUE ----
# Part 3: PREDICTIVE CLV MODELS (ADVANCED) ----
# *** ----

# BUSINESS GOALS: 
# 1. HOW MUCH CAN WE SPEND TO ACQUIRE A CUSTOMER? LIFETIME VALUE - COST TO ACQUIRE CUSTOMER (LTV - CAC)
# 2. WHICH CUSTOMERS ARE MOST VALUABLE? SEGMENTATION
# 3. HOW CAN WE INCREASE CUSTOMER LIFETIME VALUE? MARKETING STRATEGY

# 3 PARTS:
# 1. Descriptive CLV Models (covered in this script)
# 2. Probabilistic CLV Models (covered in Part 2)
# 3. Predictive CLV Models (covered in Part 3)
# * Stick around for Part 3, that's where we'll save the MOST MONEY!


# LIBRARIES ----
import pandas as pd
import pytimetk as tk

import pycaret.classification as clf
import pycaret.regression as reg

# CONSTANTS ----
profit_margin = 0.15  # 15% Profit on Products

# DATA ----
transactions_df = pd.read_csv('data/transactions_processed.csv')

df = transactions_df.copy()

df['timestamp'] = pd.to_datetime(df['timestamp'])




# MACHINE LEARNING ----
#  Frame the problem:
#  - What will the customers spend in the next 90-Days? (Regression)
#  - What is the probability of a customer to make a purchase in next 90-days? (Classification)

n_days   = 90
max_date = df['timestamp'].max() 
cutoff   = max_date - pd.to_timedelta(n_days, unit = "d")

# Train-Test Split
temporal_in_df = df[df['timestamp'] < cutoff]

temporal_out_df = df[df['timestamp'] >= cutoff] \
    .query('household_key in @temporal_in_df.household_key')

temporal_out_df.glimpse()

# FEATURE ENGINEERING ----
#   - Most challenging part
#   - 2-Stage Process
#   - Need to frame the problem
#   - Need to think about what features to include

# Make Targets from out data ----

targets_df = temporal_out_df[['household_key', 'timestamp', 'sales_value']] \
    .groupby('household_key') \
    .sum() \
    .rename({'sales_value': 'sales_90_value'}, axis = 1) \
    .assign(sales_90_flag = 1) 
    
targets_df

# Make Recency (Date) Features from in data ----

max_date = temporal_in_df['timestamp'].max()

recency_features_df = temporal_in_df \
    [['household_key', 'timestamp']] \
    .groupby('household_key') \
    .apply(
        lambda x: int((x['timestamp'].max() - max_date) / pd.to_timedelta(1, "day"))
    ) \
    .to_frame() \
    .set_axis(["recency"], axis=1) 

recency_features_df

# Make Frequency (Count) Features from in data ----

frequency_features_df = temporal_in_df \
    [['household_key', 'timestamp']] \
    .groupby('household_key') \
    .count() \
    .set_axis(['frequency'], axis=1) 

frequency_features_df

# Make Monetary Features from in data ----

monetary_features_df = temporal_in_df \
    .groupby('household_key') \
    .aggregate(
        {
            'sales_value': ["sum", "mean"]
        }
    ) \
    .set_axis(['sales_value_sum', 'sales_value_mean'], axis = 1)

monetary_features_df

# OTHER FEATURES ----

temporal_in_df.glimpse()

# Transactions Last Month

cutoff_28d   = cutoff - pd.to_timedelta(28, unit = "d")

transactions_last_month_df = temporal_in_df[['household_key', 'timestamp']] \
    .drop_duplicates() \
    .query('timestamp >= @cutoff_28d') \
    .groupby('household_key') \
    .size() \
    .to_frame() \
    .set_axis(['transactions_last_month'], axis = 1)

transactions_last_month_df 

# Transactions Last 2 Weeks

cutoff_14d   = cutoff - pd.to_timedelta(14, unit = "d")

transactions_last_2weeks_df = temporal_in_df[['household_key', 'timestamp']] \
    .drop_duplicates() \
    .query('timestamp >= @cutoff_14d') \
    .groupby('household_key') \
    .size() \
    .to_frame() \
    .set_axis(['transactions_last_2weeks'], axis = 1)

transactions_last_2weeks_df 

# Spend Last 2 Weeks

cutoff_14d   = cutoff - pd.to_timedelta(14, unit = "d")

sales_last_2weeks_df = temporal_in_df[['household_key', 'timestamp', 'sales_value']] \
    .drop_duplicates() \
    .query('timestamp >= @cutoff_14d') \
    .groupby('household_key') \
    .sum() \
    .set_axis(['sales_value_last_2weeks'], axis = 1)

sales_last_2weeks_df

# COMBINE FEATURES ----

features_df = pd.concat(
    [recency_features_df, frequency_features_df, monetary_features_df, transactions_last_month_df,
    transactions_last_2weeks_df, sales_last_2weeks_df], axis = 1
) \
    .merge(
        targets_df, 
        left_index  = True, 
        right_index = True, 
        how         = "left"
    ) \
    .fillna(0)
    
features_df

# MACHINE LEARNING ----
# - Will use pycaret to quickly build a predictive model

# REGRESSION ----

reg_setup = reg.setup(  
    data = features_df.drop('sales_90_flag', axis = 1), 
    target = 'sales_90_value',  
    train_size = 0.8,
    normalize = True,
    session_id = 123,
    verbose = True,
    log_experiment=False
)

xgb_reg_model = reg.create_model('xgboost')

reg_predictions_df = reg.predict_model(xgb_reg_model, data = features_df) \
    .sort_values('prediction_label', ascending = False)

# CLASSIFICATION (SPEND PROBABILITY) ----

clf_setup = clf.setup(
    data = features_df.drop('sales_90_value', axis = 1), 
    target = 'sales_90_flag', 
    train_size = 0.8,
    session_id = 123,
    verbose = True,
    log_experiment=False
)

xgb_clf_model = clf.create_model('xgboost')

clf_predictions_df = clf.predict_model(xgb_clf_model, data = features_df, raw_score=True) \
    .sort_values('prediction_score_1', ascending = False)
    
# EXTRACTING INSIGHTS ----
# - Explainable AI (XAI) can be used to understand the model

reg.interpret_model(xgb_reg_model)
    
clf.interpret_model(xgb_clf_model)

# BUSINESS VALUE ----
# - What would happen if you could increase revenue by 10%?

reg_predictions_df \
    ['prediction_label'] \
    .sum() 

1_175_561 * 0.10 * 4

# It's easier than you think
top_20_customers = reg_predictions_df.head(20).index.to_list()

# *Increase *FREQUENCY* of purchases: sell them these
transactions_df \
    .query('household_key in @top_20_customers') \
    .groupby('commodity_desc') \
    .size() \
    .to_frame() \
    .set_axis(['count'], axis = 1) \
    .sort_values('count', ascending = False)

# *Increase *SIZE* of purchases: sell them these
transactions_df \
    [['household_key', 'commodity_desc', 'sales_value']] \
    .query('household_key in @top_20_customers') \
    .groupby('commodity_desc') \
    .sum() \
    .sort_values('sales_value', ascending = False)

# KEY BUSINESS INSIGHTS ----

# 1. Increasing CLV 90-Day Sales Value - If we want to increase sales value, we should focus on the customers with the highest historical sales value and get them to spend more (i.e. sales_value_sum).

reg_predictions_df

reg.interpret_model(xgb_reg_model)


# 2. Increasing CLV 90-Day Sales Probability - If we want to increase the probability of a customer making a purchase in the next 90 days, we should focus on the customers with the highest transactions last month and ensure they keep buying. Recency is the most important to increasing probability of a customer making a purchase in the next 90 days.

clf_predictions_df

clf.interpret_model(xgb_clf_model)

# CONCLUSIONS ----
# - SEE SLIDES
