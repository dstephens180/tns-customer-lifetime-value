# BUSINESS SCIENCE -----
# LAB 91: CUSTOMER LIFETIME VALUE ----
# Part 2: PROBABILISTIC CLV MODELS (INTERMEDIATE) ----
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
import lifetimes as lf
from lifetimes.plotting import plot_probability_alive_matrix

# CONSTANTS ----
profit_margin = 0.15  # 15% Profit on Products

# DATA ----
transactions_df = pd.read_csv('data/transactions_processed.csv')

df = transactions_df.copy()

df['timestamp'] = pd.to_datetime(df['timestamp'])

df.glimpse()

# PROBABILISTIC MODELS ----
# - Probabilistic models are used to predict the future transactions and churn rate of a customer.

# Data Preparatation ----
summary_3_df = lf.utils.summary_data_from_transaction_data(
    df,
    customer_id_col    = 'household_key',
    datetime_col       = 'timestamp',
    monetary_value_col = 'sales_value',
)

# Will get an error in Gamma-Gamma model if there are non-positive values
summary_3_df[summary_3_df['monetary_value'] <= 0]

summary_3_df = summary_3_df[summary_3_df['monetary_value'] > 0]

# 1.0 BG/NBD model ----
#  - The BG/NBD model is a probabilistic model that predicts the number of repeat purchases a customer will make.
#  - BG/NBD model can only predict the future transactions and churn rate of a customer

# NOTE - If doesn't converge, increase penalizer_coef
bgf = lf.BetaGeoFitter(penalizer_coef=0.15)

bgf.fit(summary_3_df['frequency'], summary_3_df['recency'], summary_3_df['T'])

bgf.summary

# Conditional probability alive
summary_3_df['probability_alive'] = bgf.conditional_probability_alive(summary_3_df['frequency'], summary_3_df['recency'], summary_3_df['T'])

summary_3_df

# Visualizing the probability alive matrix
plot_probability_alive_matrix(bgf,cmap='viridis')

# If a customer has bought multiple times (frequency) and the time between the first & last transaction is high (recency), then his/her probability of being alive is high.
# If a customer has less frequency (bought once or twice) and the time between first & last transaction is low (recency), then his/her probability of being alive is high.

# GAMMA-GAMMA MODEL ----
# - The Gamma-Gamma model is used to predict the average transaction value for each customer.

ggf = lf.GammaGammaFitter(penalizer_coef=0.1)

ggf.fit(summary_3_df['frequency'], summary_3_df['monetary_value'])

summary_3_df['predicted_avg_sales'] = ggf.conditional_expected_average_profit(summary_3_df['frequency'], summary_3_df['monetary_value'])

summary_3_df

# Predict the Customer Lifetime Value for Next 90 Days ----

summary_3_df['predicted_clv_3mo'] = ggf.customer_lifetime_value(
    bgf,
    summary_3_df['frequency'],
    summary_3_df['recency'],
    summary_3_df['T'],
    summary_3_df['monetary_value'],
    time=3,  # Time in Months
    freq='D',  # Daily 
    discount_rate=0.01  # 1% discount rate
)

summary_3_df

# Don't forget to add in profit margin
summary_3_df['predicted_profit_3mo'] = summary_3_df['predicted_clv_3mo']*profit_margin 

summary_3_df

# CONCLUSIONS ----
# 1. Now we know how much we can spend to target these customers in the next 3 months.
# 2. Much better than the historical / descriptive models as it predicts the future transactions and churn rate of a customer.
# 3. However, we don't know what relationships exist between the features and the the lifetime value of a customer, and therefore, we can't confidently improve the business. 
# 4. The next step is to build predictive models to forecast future customer lifetime value AND to understand the relationships between the features and the lifetime value of a customer.

