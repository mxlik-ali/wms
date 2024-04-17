from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from matplotlib import pyplot as plt
import numpy as np
from data import *
import pandas as pd

# Load data
sales_train = pd.read_csv('data/sales_train.csv')
items = pd.read_csv('data/items.csv')
shops = pd.read_csv('data/shops.csv')

# Merge datasets
merged_data = pd.merge(sales_train, items, on='item_id')
merged_data = pd.merge(merged_data, shops, on='shop_id')

# Separate data for each shop
shop_ids = merged_data['shop_id'].unique()

for shop_id in shop_ids:
    shop_data = merged_data[merged_data['shop_id'] == shop_id]
    
    # Feature Engineering specific to each shop
    # ...
    
    # Split data
    X = shop_data[['item_id', 'item_category_id', 'date_block_num']]  # Features specific to each shop
    y = shop_data['item_cnt_day']  # Target variable
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on validation set
    val_preds = model.predict(X_val)
    
    # Evaluate model
    mse = mean_squared_error(y_val, val_preds)
    print(f'Shop {shop_id} Mean Squared Error:', mse)
    
    # Plot actual vs predicted sales
    plt.figure(figsize=(10, 6))
    plt.scatter(X_val['date_block_num'], y_val, color='blue', label='Actual')
    plt.scatter(X_val['date_block_num'], val_preds, color='red', label='Predicted')
    plt.title(f'Shop {shop_id} Actual vs Predicted Sales')
    plt.xlabel('Date Block Number')
    plt.ylabel('Item Count')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Make predictions on test set
    # Preprocess test data similarly to training data
    # test_preds = model.predict(test_data)
    
    # Generate submission file
    # ...
