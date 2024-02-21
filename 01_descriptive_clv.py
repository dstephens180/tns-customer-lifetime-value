# BUSINESS SCIENCE -----
# LAB 91: CUSTOMER LIFETIME VALUE ----
# Part 1: DESCRIPTIVE CLV MODELS (BEGINNER) ----
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



# -- 0.0 DATA IMPORT
# DATA ----
transactions_df = pd.read_csv('data/transactions_processed.csv')

transactions_df.glimpse()

df = transactions_df.copy()

df['timestamp'] = pd.to_datetime(df['timestamp'])



# EXPLORATORY DATA ANALYSIS ----
df.glimpse()

df['sales_value'].sum()

df[['timestamp', 'sales_value']] \
    .summarize_by_time(
        date_column = 'timestamp',
        value_column = 'sales_value',
        agg_func = 'sum',
        freq = 'M'
    ) \
    .plot_timeseries('timestamp', 'sales_value')


# --
# 1.0 AGGREGATION MODELS ----

# Aggregation models are used to calculate the average customer lifetime value for a group of customers or a cohort.

customer_sales_1_df = df \
    .groupby(['household_key', 'basket_id']) \
    .agg(
        total_sales_basket=('sales_value', 'sum'),
        timestamp=('timestamp', 'max')
    ) \
    .reset_index() \
    .groupby('household_key') \
    .agg(
        # Time difference in days
        time_days=('timestamp', lambda x: (x.max() - x.min()).days),  
        # Count of unique 'basket_id'
        frequency=('basket_id', 'nunique'),  
        # Sum of 'sales_value'
        total_sales=('total_sales_basket', 'sum'),
        avg_sales=('total_sales_basket', 'mean') 
    ) \
    .reset_index()
    
customer_sales_1_df


# create a dictionary including all customer info
summary_1 = {
    'average_sales': customer_sales_1_df['avg_sales'].mean(),
    'average_purchase_freq': customer_sales_1_df['frequency'].mean(),
    'churn_rate': 1 - (customer_sales_1_df['frequency'] > 5).sum() / len(customer_sales_1_df['frequency']),
    'max_days': customer_sales_1_df['time_days'].max()
}

summary_1_df = pd.DataFrame([summary_1])
summary_1_df


# Define the constants
profit_margin     = 0.15  # 15% Profit on Products
customer_lifetime = 5  # 5 years
eps_churn_rate    = 0.001  # very small value to avoid inf values


# Churn CLV Calculation
summary_1_df['clv_churn_method'] = (summary_1_df['average_sales'] * summary_1_df['average_purchase_freq'] / (summary_1_df['churn_rate'] + eps_churn_rate)) * profit_margin


# Lifetime CLV Calculation
summary_1_df['clv_lifetime_method'] = (summary_1_df['average_sales'] * summary_1_df['average_purchase_freq'] / (summary_1_df['max_days'] / 365) * customer_lifetime) * profit_margin
summary_1_df


# 2.0 COHORT MODELS ----

# Cohort models are used to calculate the average customer lifetime value for a group of customers or a cohort. Often times, the cohort is defined by the customer's first purchase date.

# Constants
profit_margin     = 0.15  # 15% Profit on Products
customer_lifetime = 5  # 5 years
eps_churn_rate    = 0.001

# Calculate start_month for each household
df['start_month'] = df.groupby('household_key')['timestamp'].transform(lambda x: x.min().strftime('%Y-%m'))

df.glimpse()

# Aggregate data by start_month and household_key
cohort_data = df \
    .groupby(['start_month', 'household_key', 'basket_id']) \
    .agg(
        total_sales_basket=('sales_value', 'sum'),
        timestamp=('timestamp', 'max')
    ) \
    .reset_index() \
    .groupby(['start_month', 'household_key']) \
    .agg(
        time_days=('timestamp', lambda x: (x.max() - x.min()).days),
        frequency=('basket_id', 'nunique'),
        total_sales=('total_sales_basket', 'sum'),
        avg_sales=('total_sales_basket', 'mean') 
    ) \
    .reset_index()

# Calculate CLV metrics by start_month
summary_2_df = cohort_data \
    .groupby('start_month') \
    .agg(
        cohort_size=('household_key', 'nunique'),
        average_sales=('avg_sales', 'mean'),
        average_purchase_freq=('frequency', 'mean'),
        churn_rate=('frequency', lambda x: 1 - (x > 5).sum() / len(x)),
        max_days=('time_days', 'max')
    ) \
    .reset_index()

# Add Churn CLV calculation
summary_2_df['clv_churn_method'] = (summary_2_df['average_sales'] * summary_2_df['average_purchase_freq'] / (summary_2_df['churn_rate'] + eps_churn_rate)) * profit_margin

# Add Lifetime CLV calculation
summary_2_df['clv_lifetime_method'] = (summary_2_df['average_sales'] * summary_2_df['average_purchase_freq'] * customer_lifetime / (summary_2_df['max_days'] / 365)) * profit_margin

summary_2_df

# CONCLUSIONS ----
# 1. I don't trust these Customer Lifetime Value (CLV) calculations. They are super optimistic for the churn calculation. Lifetime is a bit more realistic, but still has not earned my trust.
# 2. The bottom line is that as we get more granular, we can gain higher accuracy in our CLV calculations.
# 3. The next step is to build predictive models to forecast future customer lifetime value.



