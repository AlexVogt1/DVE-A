import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
# from scipy.stats import zscore
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,r2_score
from pmdarima import auto_arima
from prophet import Prophet

#Read in Data
sales_vol = pd.read_csv("./Outlier_Detection/sales_volumes.csv",parse_dates=["date"],converters={'product_code':str, 'description':str, 'volume':int})
baseline = pd.read_csv("./Outlier_Detection/baseline.csv")

# Dealing with returns
def adjust_negative_volumes(df):
    # Split the DataFrame into positive and negative volumes
    negative_volumes = df[df['volume'] < 0].copy()
    positive_volumes = df[df['volume'] > 0].copy()

    # loop over negative volumes
    for idx, neg_row in negative_volumes.iterrows():
        # Find a corresponding positive volume row
        matching_rows = positive_volumes[
            (positive_volumes['product_code'] == neg_row['product_code']) &
            (positive_volumes['customer_id'] == neg_row['customer_id']) &
            (positive_volumes['volume'] > 0)
        ]

        if not matching_rows.empty:
            # modify the first matching positive row
            pos_idx = matching_rows.index[0]
            df.at[pos_idx, 'volume'] += neg_row['volume']  # Subtracting a negative is equivalent to adding

            # Set the negative volume to zero (We will be aggreating later anyway)
            df.at[idx, 'volume'] = 0

    return df

# Dealing with 2 for one (Assumed based on breif that all purchaces with 0 unit price is 2 for 1 special)
def double_volume_for_zero_unit_price(df):
    # Find rows where unit_price is 0
    zero_price_rows = df['unit_price'] == 0
    # Double the volume for those rows
    df.loc[zero_price_rows, 'volume'] *= 2

    return df

def detect_outliers_isolation_forest_separately(df, pos_contamination=0.05, neg_contamination=0.05):
    # Split the DataFrame into positive and negative volumes
    positive_volumes = df[df['volume'] >= 0].copy()
    negative_volumes = df[df['volume'] <= 0].copy()

    # Instantiate the Isolation Forest model
    iso_forest = IsolationForest(contamination=pos_contamination, random_state=42)
    iso_forest_neg = IsolationForest(contamination=neg_contamination, random_state=42) # bulk returns are far less likeley to be real
    
    # Fit the model and predict outliers for positive volumes
    positive_volumes['is_outlier'] = iso_forest.fit_predict(positive_volumes[['volume']])
    positive_volumes['is_outlier'] = positive_volumes['is_outlier'].apply(lambda x: True if x == -1 else False)
    
    # Fit the model and predict outliers for negative volumes
    negative_volumes['is_outlier'] = iso_forest_neg.fit_predict(negative_volumes[['volume']])
    negative_volumes['is_outlier'] = negative_volumes['is_outlier'].apply(lambda x: True if x == -1 else False)

    # Combine the results back into the original DataFrame
    df = pd.concat([positive_volumes, negative_volumes])
    outliers = df[df['is_outlier']]
    # print(f"\nNumber of Outliers Detected: {len(outliers)}")
    # print(outliers)

    # Ensure the DataFrame is sorted by the original index (optional)
    df = df.sort_index()

    return df

def replace_outliers_with_zero(df):
    # Replace volumes with 0 where is_outlier is True
    df.loc[df['is_outlier'] == True, 'volume'] = 0
    
    return df

def train_and_test_auto_arima(df):
    results = []
    for product in df['product_code'].unique():
        product_sales = df[df['product_code'] == product].set_index('date')['volume']
        
        # Split data into training and validation sets (e.g., last 4 days for validation)
        train = product_sales.iloc[:-4]
        test = product_sales.iloc[-4:]
        
        # Use auto_arima to find the best model
        model = auto_arima(train, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
        
        # Forecast next 4 days (the validation set period)
        forecast = model.predict(n_periods=4)
        
        # Calculate validation metrics
        mae = mean_absolute_error(test, forecast)
        mse = mean_squared_error(test, forecast)
        mape = mean_absolute_percentage_error(test, forecast)
        
        # Forecasting next 4 days
        future_forecast = model.predict(n_periods=4).sum()
        
        results.append({
            'product_code': product,
            'forecasted_volume': future_forecast,
            'mae': mae,
            'mse': mse,
            'mape': mape
        })

    results_df = pd.DataFrame(results)
    return results_df

def train_and_test_prophet(df):
    results =[]
    for product in df['product_code'].unique():
        product_sales = df[df['product_code'] == product].set_index('date')['volume']
        
        # Prep data for Prophet
        df_prophet = product_sales.reset_index().rename(columns={'date': 'ds', 'volume': 'y'})
        
        # Split data into training and validation sets (e.g., last 4 days for validation)
        train = df_prophet.iloc[:-4]
        test = df_prophet.iloc[-4:]
        
        # Init and fit Prophet model
        model = Prophet(changepoint_prior_scale=0.8)
        model.fit(train)
        
        # Create dataframe for future dates (including validation period)
        future = model.make_future_dataframe(periods=4)
        
        # Forecast
        forecast = model.predict(future)
        
        # Extract predictions for the validation period
        forecasted_values = forecast.iloc[-4:]['yhat'].values
        
        # Calculate validation metrics
        mae = mean_absolute_error(test['y'], forecasted_values)
        mse = mean_squared_error(test['y'], forecasted_values)
        mape = mean_absolute_percentage_error(test['y'], forecasted_values)
        
        # Forecasting next 4 days
        future_forecast = forecast.iloc[-4:]['yhat'].sum()
        
        results.append({
            'product_code': product,
            'forecasted_volume': future_forecast,
            'mae': mae,
            'mse': mse,
            'mape': mape
        })

    # Results to DataFrame
    prophet_df = pd.DataFrame(results)
    return prophet_df

def prophet_forecaster(df, n_days_to_forecast):
    results = []
    for product in df['product_code'].unique():
        product_sales = df[df['product_code'] == product].set_index('date')['volume']
        
        # Prep data for Prophet
        df_prophet = product_sales.reset_index().rename(columns={'date': 'ds', 'y': 'volume'})
        df_prophet = df_prophet.rename(columns={'volume': 'y'})
        
        # Init and fit Prophet model on the entire dataset
        model = Prophet(daily_seasonality=False,changepoint_prior_scale=0.8)
        model.fit(df_prophet)
        
        # Create a dataframe for future dates 
        future = model.make_future_dataframe(periods=n_days_to_forecast)
        
        # Forecast
        forecast = model.predict(future)
        
        # Sum forecasted values for the next n days
        next_n_days_forecast = forecast.iloc[-n_days_to_forecast:]['yhat'].sum()
        
        results.append({
            'product_code': product,
            'prophet_forecast': next_n_days_forecast
        })

    # Results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

if __name__ == '__main__':
    #Read in Data
    sales_vol = pd.read_csv("./Outlier_Detection/sales_volumes.csv",parse_dates=["date"],converters={'product_code':str, 'description':str, 'volume':int})
    baseline = pd.read_csv("./Outlier_Detection/baseline.csv")

    # Dealing With Returns (negative Volume Numbers)
    sales_vol = adjust_negative_volumes(sales_vol)

    # Dealing with 0 Unit Price (discounts)
    sales_vol = double_volume_for_zero_unit_price(sales_vol)

    # Perform outlier detection on individual transactions
    sales_vol = detect_outliers_isolation_forest_separately(sales_vol, pos_contamination=0.002, neg_contamination=0.02).drop_duplicates()
    sales_vol = sales_vol[sales_vol['is_outlier']==False]

    # Sort out missing dates
    # Get total volume of every product sold per day
    daily_sales = sales_vol.groupby(['product_code', 'date']).agg({'volume': 'sum'}).reset_index()

    # Reindex to include all dates for each product, filling missing dates with 0 volume
    all_dates = pd.date_range(start=daily_sales['date'].min(), end=daily_sales['date'].max(), freq='D')
    daily_sales.columns = ['product_code', 'date', 'volume']
    # Create a complete date range for each product and merge it back with the existing data
    products = daily_sales['product_code'].unique()
    # Generate a DataFrame with all dates for each product
    complete_index = pd.MultiIndex.from_product([products, all_dates], names=['product_code', 'date'])
    complete_daily_sales = pd.DataFrame(index=complete_index).reset_index()
    # Merge with the actual sales data and fill missing volumes with 0
    complete_daily_sales = complete_daily_sales.merge(daily_sales, on=['product_code', 'date'], how='left').fillna(0)

    # Now perform outlier removal on dailey product sale that might affect the model
    complete_daily_sales = detect_outliers_isolation_forest_separately(complete_daily_sales, pos_contamination=0.05, neg_contamination=0.0001).drop_duplicates()
    complete_daily_sales = replace_outliers_with_zero(complete_daily_sales) # replace with zero as we every product needs a value sold on every day even if 0
    
    # Training and test models 
    # Because we are going to be training 100 forcasting models I used automated approaches
    # used daily sales as weekley sale would not be enough to train models as weekley would only have 27 data points per product for training
    # auto_arima_train_results = train_and_test_auto_arima(complete_daily_sales) #commented out to save compute time
    # auto_arima work resonably well but did take some time a few minutes to tain and test
    
    # Train and test facebooks Prophet Forecaster
    prophet_train_results = train_and_test_prophet(complete_daily_sales)
    # prophet performed slightly better than auto_arima and was also faster
    
    # train final model on all data and forecast for the next 28 days (4 weeks)
    prophet_forecast = prophet_forecaster(complete_daily_sales, n_days_to_forecast=28)

    # merging basline and my forcasts together
    forecasts = prophet_forecast
    forecasts['baseline_forecast']= baseline['baseline_forecast']

    # Wind trending products
    forecasts['is_trending'] = forecasts['prophet_forecast'] >= (2*forecasts['baseline_forecast'])

    # Write results to a csv
    forecasts[['product_code', 'is_trending']].to_csv(path_or_buf='trending_products.csv',header=True,index=False)
