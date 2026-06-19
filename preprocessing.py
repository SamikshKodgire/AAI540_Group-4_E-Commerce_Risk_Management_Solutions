import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def main():
    base_dir = '/opt/ml/processing'
    input_dir = os.path.join(base_dir, 'input')
    
    # 1. Load Data
    orders = pd.read_csv(os.path.join(input_dir, 'olist_orders_dataset.csv'))
    payments = pd.read_csv(os.path.join(input_dir, 'olist_order_payments_dataset.csv'))
    reviews = pd.read_csv(os.path.join(input_dir, 'olist_order_reviews_dataset.csv'))
    customers = pd.read_csv(os.path.join(input_dir, 'olist_customers_dataset.csv'))
    items = pd.read_csv(os.path.join(input_dir, 'olist_order_items_dataset.csv'))
    
    # 2. Merge
    df = orders.merge(customers, on='customer_id', how='left') \
               .merge(payments, on='order_id', how='left') \
               .merge(reviews, on='order_id', how='left')
    
    freight_df = items.groupby('order_id')['freight_value'].sum().reset_index()
    df = df.merge(freight_df, on='order_id', how='left').fillna(0)
    
    # 3. Feature Engineering
    datetime_cols = ['order_purchase_timestamp', 'order_estimated_delivery_date', 'order_delivered_customer_date']
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
    df['logistical_failure'] = ((df['order_status'] != 'delivered') | 
                                (df['order_delivered_customer_date'] > df['order_estimated_delivery_date']))
    df['is_high_risk'] = np.where((df['review_score'] <= 2) & df['logistical_failure'], 1, 0)
    
    df['delivery_delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days.fillna(0)
    df['freight_ratio'] = np.where(df['payment_value'] > 0, df['freight_value'] / df['payment_value'], 0)
    
    # 4. Encoding
    le = LabelEncoder()
    df['payment_type'] = le.fit_transform(df['payment_type'].astype(str))
    df['customer_state'] = le.fit_transform(df['customer_state'].astype(str))
    
    # 5. Prepare for XGBoost
    cols = ['order_id', 'is_high_risk', 'payment_type', 'payment_installments', 'payment_value', 
            'freight_ratio', 'customer_state', 'delivery_delay_days']
    df_model = df[cols].copy().dropna()
    
    import time
    df_model['EventTime'] = pd.Series([time.time()] * len(df_model), dtype="float64")
    
    fs_dir = os.path.join(base_dir, 'fs_data')
    os.makedirs(fs_dir, exist_ok=True)
    df_model.to_csv(os.path.join(fs_dir, 'final_features.csv'), index=False)
    # -----------------------------------------------

    # Now strip the headers/IDs for XGBoost
    df_xgboost = df_model.drop(columns=['order_id', 'EventTime'])
    batch_data = df_xgboost.drop(columns=['is_high_risk']).tail(100)
    
    batch_dir = os.path.join(base_dir, 'batch')
    os.makedirs(batch_dir, exist_ok=True)
    
    # Save without headers/index as required by the XGBoost model
    batch_data.to_csv(os.path.join(batch_dir, 'batch.csv'), header=False, index=False)
    # ------------------------------------------

    train, test = train_test_split(df_xgboost, test_size=0.2, random_state=42, stratify=df_xgboost['is_high_risk'])
   
    # 6. Save
    os.makedirs(os.path.join(base_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test'), exist_ok=True)
    train.to_csv(os.path.join(base_dir, 'train', 'train.csv'), header=False, index=False)
    test.to_csv(os.path.join(base_dir, 'test', 'test.csv'), header=False, index=False)

if __name__ == "__main__":
    main()
